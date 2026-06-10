"""Parent-side driver for the `triton_source` subprocess sandbox.

`run_sandboxed_eval` writes the evaluation payload next to the candidate
artifacts, spawns `sandbox_runner` in a fresh interpreter (inheriting the
parent environment — so `CUDA_VISIBLE_DEVICES` pinning flows through), and
parses the marker-prefixed JSON the runner prints.

GPU lease pool (episode parallelism): when ``COMPILAGENT_GPU_POOL`` is set,
EVERY sandbox run (baseline included) first leases one pool device via
`gpu_lease.acquire`, holds the lease around the entire subprocess
invocation, and injects ``CUDA_VISIBLE_DEVICES=<leased device>`` into the
child env — overriding any inherited pinning, since pool indices are
physical. The sandbox is the only GPU-bound unit (compile + gates +
CUDA-event timing of candidate AND reference happen inside it), so this
mutual exclusion is all that timing validity requires. Lease wait happens
*before* `subprocess.run`, so it never counts against the sandbox hard
timeout; the leased device and wait seconds are recorded under the
``gpu_lease`` key of the result so suite rows can report GPU wait. With the
env var unset, behavior is exactly the historical one (inherited pinning).

Hard timeout: `subprocess.run(timeout=...)` kills the child on expiry, so a
candidate that hangs (`while True: pass`), deadlocks, or OOMs can never take
the session down — the failure comes back as an error dict the backend folds
into `CompileResult(ok=False, ...)`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .gpu_lease import GpuLeaseTimeoutError, acquire, pool_devices
from .sandbox_runner import (
    DEFAULT_REPETITIONS,
    DEFAULT_TRIAL_SEEDS,
    DEFAULT_WARMUP,
    MARKER,
)

DEFAULT_TIMEOUT_SECONDS = 240.0

#: How long a sandbox may wait for a pool device before giving up (the wait
#: is queueing, not compute, hence the generous default); override with the
#: env var below when worker:GPU ratios make longer queues legitimate.
DEFAULT_LEASE_TIMEOUT_SECONDS = 3600.0
LEASE_TIMEOUT_ENV = "COMPILAGENT_GPU_LEASE_TIMEOUT"

_RUNNER_MODULE = "compilagent.integrations.triton_source._internal.sandbox_runner"


def _failure(error: str, *, timed_out: bool = False) -> dict[str, Any]:
    return {
        "compiled": False,
        "gates": {},
        "cand_ms": None,
        "ref_ms": None,
        "speedup_vs_ref": None,
        "error": error,
        "timed_out": timed_out,
    }


def run_sandboxed_eval(
    *,
    reference_source: str,
    candidate_source: str | None,
    artifact_dir: Path,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    warmup: int = DEFAULT_WARMUP,
    repetitions: int = DEFAULT_REPETITIONS,
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
) -> dict[str, Any]:
    """Evaluate one candidate (or the baseline, when `candidate_source` is
    None) in a sandboxed subprocess; never raises for expected failures."""

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_source": reference_source,
        "candidate_source": candidate_source,
        "workdir": str(artifact_dir),
        "atol": atol,
        "rtol": rtol,
        "warmup": warmup,
        "repetitions": repetitions,
        "trial_seeds": list(trial_seeds),
    }
    payload_path = artifact_dir / "sandbox_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not pool_devices():
        # No pool configured: historical behavior — the child inherits the
        # parent environment, i.e. whatever CUDA_VISIBLE_DEVICES pinning the
        # worker was launched with.
        return _invoke_runner(payload_path, artifact_dir, timeout_seconds, env=None)

    # Pool mode: hold an exclusive device lease around the ENTIRE subprocess
    # invocation. Acquiring before `subprocess.run` keeps lease (queue) wait
    # out of the sandbox hard timeout.
    try:
        lease = acquire(timeout=_lease_timeout_seconds())
    except (GpuLeaseTimeoutError, ValueError, OSError) as exc:
        return _failure(f"GPU lease unavailable: {exc}")
    with lease:
        # Override, never compose: pool entries are physical device indices,
        # so the child must see exactly the leased device regardless of any
        # inherited pinning.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": lease.device}
        result = _invoke_runner(payload_path, artifact_dir, timeout_seconds, env=env)
    result["gpu_lease"] = {
        "device": lease.device,
        "lease_wait_s": round(lease.wait_seconds, 3),
    }
    return result


def _lease_timeout_seconds() -> float:
    raw = os.environ.get(LEASE_TIMEOUT_ENV, "") or ""
    try:
        return float(raw) if raw else DEFAULT_LEASE_TIMEOUT_SECONDS
    except ValueError:
        return DEFAULT_LEASE_TIMEOUT_SECONDS


def _invoke_runner(
    payload_path: Path,
    artifact_dir: Path,
    timeout_seconds: float,
    *,
    env: dict[str, str] | None,
) -> dict[str, Any]:
    """Spawn the sandbox runner and parse its marker-prefixed JSON.

    `env=None` inherits the parent environment verbatim; pool mode passes a
    copy with `CUDA_VISIBLE_DEVICES` rewritten to the leased device.
    """

    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", _RUNNER_MODULE, str(payload_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return _failure(
            f"sandbox subprocess exceeded the {timeout_seconds:.0f}s hard "
            "timeout (hang, deadlock, or pathological kernel) and was killed.",
            timed_out=True,
        )
    except Exception as exc:  # noqa: BLE001
        return _failure(f"sandbox subprocess failed to launch: {exc!r}")

    (artifact_dir / "sandbox_stdout.log").write_text(
        proc.stdout or "", encoding="utf-8"
    )
    (artifact_dir / "sandbox_stderr.log").write_text(
        proc.stderr or "", encoding="utf-8"
    )

    for line in (proc.stdout or "").splitlines():
        if line.startswith(MARKER):
            try:
                result = json.loads(line[len(MARKER):])
            except json.JSONDecodeError as exc:
                return _failure(f"sandbox emitted unparseable JSON: {exc}")
            result.setdefault("timed_out", False)
            return result

    stderr_tail = (proc.stderr or "")[-2000:]
    return _failure(
        "sandbox produced no result JSON "
        f"(exit code {proc.returncode}); stderr tail: {stderr_tail}"
    )
