"""CPU-only tests for the inter-process GPU lease pool (`gpu_lease`).

No CUDA, no torch: the pool is pure flock bookkeeping, so "devices" here
are arbitrary indices whose lock files live under a tmp dir. Child
processes run via `subprocess` (not multiprocessing) so the tests exercise
the real cross-process semantics — including flock auto-release on SIGKILL.

The module (not its names) is imported because the integrations conftest
reloads `compilagent.integrations.*` per test — attribute access at call
time always hits the freshly reloaded objects.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from compilagent.integrations.triton_source._internal import gpu_lease, sandbox

REPO_ROOT = Path(__file__).resolve().parents[3]


def _child_env(lock_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["COMPILAGENT_GPU_LOCK_DIR"] = str(lock_dir)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    env["CUDA_VISIBLE_DEVICES"] = ""  # belt and braces: children never use GPUs
    return env


# ------------------------------------------------------------ acquire basics


def test_acquire_requires_a_configured_pool(monkeypatch):
    monkeypatch.delenv("COMPILAGENT_GPU_POOL", raising=False)
    assert gpu_lease.pool_devices() == ()
    with pytest.raises(ValueError, match="COMPILAGENT_GPU_POOL"):
        gpu_lease.acquire(timeout=0.1)


def test_pool_devices_parses_env(monkeypatch):
    monkeypatch.setenv("COMPILAGENT_GPU_POOL", " 1, 2,3 ,")
    assert gpu_lease.pool_devices() == ("1", "2", "3")


def test_acquire_times_out_with_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPILAGENT_GPU_LOCK_DIR", str(tmp_path))
    holder = gpu_lease.acquire(timeout=1.0, devices=("11",))
    try:
        started = time.monotonic()
        with pytest.raises(gpu_lease.GpuLeaseTimeoutError, match="every device stayed busy"):
            # Same process, second fd: flock is per-open-file-description,
            # so this contends exactly like another process would.
            gpu_lease.acquire(timeout=0.4, devices=("11",))
        assert time.monotonic() - started >= 0.4
    finally:
        holder.release()


def test_lease_is_reacquirable_after_context_exit(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPILAGENT_GPU_LOCK_DIR", str(tmp_path))
    with gpu_lease.acquire(timeout=1.0, devices=("12",)) as lease:
        assert lease.device == "12"
        assert lease.wait_seconds >= 0.0
    # Released on exit: an immediate re-acquire must succeed.
    gpu_lease.acquire(timeout=0.5, devices=("12",)).release()


# ------------------------------------- (a) two processes serialize on 1 GPU


_CRITICAL_CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from compilagent.integrations.triton_source._internal import gpu_lease

    marker_dir = Path(sys.argv[1]); idx = sys.argv[2]
    with gpu_lease.acquire(timeout=30.0, devices=("17",)):
        in_critical = marker_dir / "in_critical"
        if in_critical.exists():
            (marker_dir / f"overlap_{idx}").touch()  # another holder inside!
        in_critical.touch()
        time.sleep(0.4)
        in_critical.unlink()
    (marker_dir / f"done_{idx}").touch()
    """
)


def test_two_processes_contending_for_one_device_serialize(tmp_path):
    lock_dir = tmp_path / "locks"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CRITICAL_CHILD, str(marker_dir), str(i)],
            env=_child_env(lock_dir),
        )
        for i in range(2)
    ]
    for p in procs:
        assert p.wait(timeout=60) == 0
    assert (marker_dir / "done_0").exists() and (marker_dir / "done_1").exists()
    overlaps = list(marker_dir.glob("overlap_*"))
    assert not overlaps, f"critical sections overlapped: {overlaps}"


# ------------------------------------ (b) lease released on process SIGKILL


_HOLDER_CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from compilagent.integrations.triton_source._internal import gpu_lease

    lease = gpu_lease.acquire(timeout=10.0, devices=("19",))
    Path(sys.argv[1]).touch()  # signal: lease is held
    time.sleep(120)            # hold "forever" until the parent kills us
    """
)


# ----------------------------------- (c) sandbox CUDA_VISIBLE_DEVICES wiring


def _fake_sandbox_run(captured: dict):
    """A `subprocess.run` stand-in that records the env it was given and
    emits a well-formed sandbox result on stdout."""

    payload = {
        "compiled": True,
        "gates": {},
        "cand_ms": 1.0,
        "ref_ms": 2.0,
        "speedup_vs_ref": 2.0,
        "error": None,
    }

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=sandbox.MARKER + json.dumps(payload) + "\n",
            stderr="",
        )

    return fake_run


def test_sandbox_injects_leased_device_when_pool_set(monkeypatch, tmp_path):
    captured: dict = {}
    monkeypatch.setattr(sandbox.subprocess, "run", _fake_sandbox_run(captured))
    monkeypatch.setenv("COMPILAGENT_GPU_POOL", "5")
    monkeypatch.setenv("COMPILAGENT_GPU_LOCK_DIR", str(tmp_path / "locks"))
    # Inherited pinning MUST be overridden: pool indices are physical.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    result = sandbox.run_sandboxed_eval(
        reference_source="ref",
        candidate_source="cand",
        artifact_dir=tmp_path / "artifacts",
    )

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "5"
    assert result["compiled"] is True
    assert result["gpu_lease"]["device"] == "5"
    assert result["gpu_lease"]["lease_wait_s"] >= 0.0
    # The lease was released with the subprocess: re-acquire must be instant.
    gpu_lease.acquire(timeout=0.5, devices=("5",)).release()


def test_sandbox_inherits_parent_env_when_pool_unset(monkeypatch, tmp_path):
    captured: dict = {}
    monkeypatch.setattr(sandbox.subprocess, "run", _fake_sandbox_run(captured))
    monkeypatch.delenv("COMPILAGENT_GPU_POOL", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    result = sandbox.run_sandboxed_eval(
        reference_source="ref",
        candidate_source="cand",
        artifact_dir=tmp_path / "artifacts",
    )

    # env=None → the child inherits the parent environment verbatim, i.e.
    # the historical CUDA_VISIBLE_DEVICES pinning flows through untouched.
    assert captured["env"] is None
    assert "gpu_lease" not in result


def test_sandbox_folds_lease_timeout_into_failure_dict(monkeypatch, tmp_path):
    def never_run(cmd, **kwargs):  # the subprocess must NOT be spawned
        raise AssertionError("sandbox subprocess spawned without a lease")

    monkeypatch.setattr(sandbox.subprocess, "run", never_run)
    monkeypatch.setenv("COMPILAGENT_GPU_POOL", "6")
    monkeypatch.setenv("COMPILAGENT_GPU_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv(sandbox.LEASE_TIMEOUT_ENV, "0.2")

    holder = gpu_lease.acquire(timeout=1.0, devices=("6",))
    try:
        result = sandbox.run_sandboxed_eval(
            reference_source="ref",
            candidate_source="cand",
            artifact_dir=tmp_path / "artifacts",
        )
    finally:
        holder.release()
    assert result["compiled"] is False
    assert "GPU lease unavailable" in result["error"]


def test_lease_released_when_holder_is_sigkilled(monkeypatch, tmp_path):
    lock_dir = tmp_path / "locks"
    held_flag = tmp_path / "held"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_CHILD, str(held_flag)],
        env=_child_env(lock_dir),
    )
    try:
        deadline = time.monotonic() + 30.0
        while not held_flag.exists():
            assert proc.poll() is None, "holder child died before leasing"
            assert time.monotonic() < deadline, "holder never acquired"
            time.sleep(0.05)
        # While the child lives, the device must be unavailable.
        monkeypatch.setenv("COMPILAGENT_GPU_LOCK_DIR", str(lock_dir))
        with pytest.raises(gpu_lease.GpuLeaseTimeoutError):
            gpu_lease.acquire(timeout=0.3, devices=("19",))
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=30)
        # flock auto-releases on fd close at process death: re-acquire works.
        gpu_lease.acquire(timeout=5.0, devices=("19",)).release()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
