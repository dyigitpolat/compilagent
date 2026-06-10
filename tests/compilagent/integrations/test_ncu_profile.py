"""Unit tests for the NCU profiling runner (mocked ncu — no GPU, no ncu).

Covers the CSV parser, dominant-kernel curation, the plain-then-`sudo -n`
permission fallback, graceful degradation to ``metrics={}`` +
``ncu_unavailable``, and GPU-lease wrapping (the lease is held around the
whole invocation and released afterwards). The real `ncu` binary is never
invoked: every test injects a fake `invoke`.
"""

from __future__ import annotations

import subprocess

from compilagent.integrations.triton_source._internal import gpu_lease
from compilagent.integrations.triton_source._internal.ncu_profile import (
    CURATED_METRICS,
    NVTX_RANGE,
    build_candidate_snippet,
    curate_metrics,
    format_metrics_block,
    parse_ncu_csv,
    profile_candidate,
    profile_snippet,
)

_HEADER = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context",'
    '"Stream","Block Size","Grid Size","Device","CC","Section Name",'
    '"Metric Name","Metric Unit","Metric Value"'
)


def _row(launch_id, kernel, metric, unit, value):
    return (
        f'"{launch_id}","1234","python","127.0.0.1","{kernel}","1","7",'
        f'"(256, 1, 1)","(16, 1, 1)","0","12.0","Command line profiler metrics",'
        f'"{metric}","{unit}","{value}"'
    )


#: Two kernels: softmax_kernel dominates (longer total duration, 2 launches).
MOCK_CSV = "\n".join(
    [
        "==PROF== Connected to process 1234",
        "==PROF== Profiling \"softmax_kernel\": 0%....50%....100%",
        _HEADER,
        _row(0, "softmax_kernel", "gpu__time_duration.sum", "ns", "10,240.00"),
        _row(0, "softmax_kernel",
             "sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", "20.00"),
        _row(0, "softmax_kernel",
             "dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "80.00"),
        _row(0, "softmax_kernel", "lts__t_sector_hit_rate.pct", "%", "33.00"),
        _row(1, "softmax_kernel", "gpu__time_duration.sum", "ns", "10,260.00"),
        _row(1, "softmax_kernel",
             "sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", "22.00"),
        _row(1, "softmax_kernel",
             "dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "82.00"),
        _row(1, "softmax_kernel", "lts__t_sector_hit_rate.pct", "%", "35.00"),
        _row(2, "helper_kernel", "gpu__time_duration.sum", "ns", "100.00"),
        _row(2, "helper_kernel",
             "sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", "1.00"),
    ]
)


def _ok_invoke(record=None):
    def invoke(cmd, *, env=None, timeout=None):
        if record is not None:
            record.append({"cmd": list(cmd), "env": env, "timeout": timeout})
        return 0, MOCK_CSV, ""

    return invoke


# ------------------------------------------------------------------- parser


def test_parse_ncu_csv_skips_banner_lines_and_keeps_metric_rows():
    rows = parse_ncu_csv(MOCK_CSV)
    assert len(rows) == 10
    assert rows[0]["Kernel Name"] == "softmax_kernel"
    assert rows[0]["Metric Name"] == "gpu__time_duration.sum"
    assert rows[0]["Metric Value"] == "10,240.00"


def test_parse_ncu_csv_returns_empty_without_a_table():
    assert parse_ncu_csv("==PROF== something went wrong\n") == []
    assert parse_ncu_csv("") == []


# ----------------------------------------------------------------- curation


def test_curate_metrics_picks_dominant_kernel_and_averages_launches():
    curated = curate_metrics(parse_ncu_csv(MOCK_CSV))
    assert curated["dominant_kernel"] == "softmax_kernel"
    assert curated["kernel_names"] == ["helper_kernel", "softmax_kernel"]
    metrics = curated["metrics"]
    # Thousands separators stripped; values averaged across the 2 launches.
    assert metrics["gpu__time_duration.sum"]["value"] == 10250.0
    assert metrics["gpu__time_duration.sum"]["unit"] == "ns"
    sm = metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    assert sm["value"] == 21.0
    assert metrics["lts__t_sector_hit_rate.pct"]["value"] == 34.0
    # Only curated metrics that appeared are reported.
    assert "l1tex__t_sector_hit_rate.pct" not in metrics


def test_format_metrics_block_lists_descriptions_and_flags_other_kernels():
    profile = {
        "ncu_unavailable": False,
        **curate_metrics(parse_ncu_csv(MOCK_CSV)),
    }
    block = format_metrics_block(profile)
    assert "softmax_kernel" in block
    assert "helper_kernel" in block  # other kernels named
    assert "DRAM throughput, % of peak" in block
    assert "81.0 %" in block


def test_format_metrics_block_is_empty_when_degraded():
    assert format_metrics_block({"ncu_unavailable": True, "metrics": {}}) == ""
    assert format_metrics_block({"ncu_unavailable": False, "metrics": {}}) == ""


# ------------------------------------------------------------------ snippet


def test_candidate_snippet_embeds_sources_device_argv_and_nvtx_range():
    snippet = build_candidate_snippet(
        reference_source="class Model: pass",
        candidate_source="class ModelNew: pass",
    )
    assert "class Model: pass" in snippet
    assert "class ModelNew: pass" in snippet
    # sudo strips the env, so the device must travel as argv[1].
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]' in snippet
    assert NVTX_RANGE in snippet
    compile(snippet, "ncu_target.py", "exec")  # must be valid python


# ---------------------------------------------------------------- invocation


def test_profile_snippet_success_with_plain_ncu(tmp_path, monkeypatch):
    monkeypatch.delenv(gpu_lease.POOL_ENV, raising=False)
    record = []
    result = profile_snippet(
        snippet_source="print('hi')",
        artifact_dir=tmp_path,
        invoke=_ok_invoke(record),
    )
    assert result["ncu_unavailable"] is False
    assert result["invocation"] == "ncu"
    assert result["dominant_kernel"] == "softmax_kernel"
    assert len(result["metrics"]) == 4
    assert "gpu_lease" not in result  # no pool configured
    # Exactly one invocation: plain ncu, never sudo.
    assert len(record) == 1
    cmd = record[0]["cmd"]
    assert cmd[0].endswith("ncu")
    assert "sudo" not in cmd
    assert "--csv" in cmd
    metrics_arg = cmd[cmd.index("--metrics") + 1]
    assert metrics_arg == ",".join(name for name, _ in CURATED_METRICS)
    assert f"{NVTX_RANGE}/" in cmd
    # The target script was materialized for ncu to run.
    assert (tmp_path / "ncu_target.py").read_text() == "print('hi')"
    assert str(tmp_path / "ncu_target.py") in cmd


def test_permission_failure_falls_back_to_sudo_n(tmp_path, monkeypatch):
    monkeypatch.delenv(gpu_lease.POOL_ENV, raising=False)
    record = []

    def invoke(cmd, *, env=None, timeout=None):
        record.append(list(cmd))
        if cmd[0] == "sudo":
            return 0, MOCK_CSV, ""
        return 1, "", "==ERROR== ERR_NVGPUCTRPERM - insufficient permissions"

    result = profile_snippet(
        snippet_source="x", artifact_dir=tmp_path, invoke=invoke
    )
    assert result["ncu_unavailable"] is False
    assert result["invocation"] == "sudo -n ncu"
    assert len(record) == 2
    assert record[1][:2] == ["sudo", "-n"]


def test_both_attempts_failing_degrades_with_permission_reason(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(gpu_lease.POOL_ENV, raising=False)

    def invoke(cmd, *, env=None, timeout=None):
        if cmd[0] == "sudo":
            return 1, "", "sudo: a password is required"
        return 1, "", "==ERROR== ERR_NVGPUCTRPERM - insufficient permissions"

    result = profile_snippet(
        snippet_source="x", artifact_dir=tmp_path, invoke=invoke
    )
    assert result["ncu_unavailable"] is True
    assert result["metrics"] == {}
    assert result["reason"] == "permission"
    assert "ERR_NVGPUCTRPERM" in result["error"]
    assert "password is required" in result["error"]
    assert result["invocation"] is None


def test_missing_binary_degrades_with_binary_missing_reason(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(gpu_lease.POOL_ENV, raising=False)

    def invoke(cmd, *, env=None, timeout=None):
        raise FileNotFoundError("ncu")

    result = profile_snippet(
        snippet_source="x", artifact_dir=tmp_path, invoke=invoke
    )
    assert result["ncu_unavailable"] is True
    assert result["reason"] == "binary_missing"
    assert result["metrics"] == {}


def test_timeout_degrades_without_trying_sudo(tmp_path, monkeypatch):
    monkeypatch.delenv(gpu_lease.POOL_ENV, raising=False)
    calls = []

    def invoke(cmd, *, env=None, timeout=None):
        calls.append(list(cmd))
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout or 0)

    result = profile_snippet(
        snippet_source="x", artifact_dir=tmp_path, invoke=invoke
    )
    assert result["ncu_unavailable"] is True
    assert result["reason"] == "timeout"
    assert len(calls) == 1  # ncu WAS running; sudo retry would double wallclock


def test_empty_csv_from_a_zero_exit_degrades_as_no_kernel_rows(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(gpu_lease.POOL_ENV, raising=False)

    def invoke(cmd, *, env=None, timeout=None):
        return 0, "==PROF== nothing profiled\n", ""

    result = profile_snippet(
        snippet_source="x", artifact_dir=tmp_path, invoke=invoke
    )
    assert result["ncu_unavailable"] is True
    assert result["reason"] == "no_kernel_rows"
    assert result["metrics"] == {}


# -------------------------------------------------------------------- lease


def test_pool_mode_holds_one_lease_around_the_whole_invocation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(gpu_lease.POOL_ENV, "3")
    monkeypatch.setenv(gpu_lease.LOCK_DIR_ENV, str(tmp_path / "locks"))
    monkeypatch.setenv(gpu_lease.BUSY_CHECK_ENV, "0")
    seen = []

    def invoke(cmd, *, env=None, timeout=None):
        seen.append({"cmd": list(cmd), "env": dict(env or {})})
        # The lease must be HELD while ncu runs (replays charged to it):
        # a second non-blocking acquire on the 1-device pool must fail.
        try:
            gpu_lease.acquire(timeout=0).release()
            seen.append("LEASE WAS FREE")
        except gpu_lease.GpuLeaseTimeoutError:
            pass
        if cmd[0] == "sudo":
            return 0, MOCK_CSV, ""
        return 1, "", "ERR_NVGPUCTRPERM"

    result = profile_snippet(
        snippet_source="x", artifact_dir=tmp_path / "art", invoke=invoke
    )
    assert result["ncu_unavailable"] is False
    assert result["gpu_lease"]["device"] == "3"
    assert result["gpu_lease"]["lease_wait_s"] >= 0.0
    # Both attempts ran under the SAME (still-held) lease...
    assert [s for s in seen if s == "LEASE WAS FREE"] == []
    attempts = [s for s in seen if isinstance(s, dict)]
    assert len(attempts) == 2
    # ...with the leased device pinned in env AND passed as argv[-1]
    # (argv survives sudo's env reset).
    for attempt in attempts:
        assert attempt["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert attempt["cmd"][-1] == "3"
    # The lease is released after profile_snippet returns.
    gpu_lease.acquire(timeout=0).release()


def test_pool_mode_lease_timeout_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv(gpu_lease.POOL_ENV, "5")
    monkeypatch.setenv(gpu_lease.LOCK_DIR_ENV, str(tmp_path / "locks"))
    monkeypatch.setenv(gpu_lease.BUSY_CHECK_ENV, "0")
    monkeypatch.setenv("COMPILAGENT_GPU_LEASE_TIMEOUT", "0.05")
    blocker = gpu_lease.acquire(timeout=1)  # hold the only device
    try:
        result = profile_snippet(
            snippet_source="x",
            artifact_dir=tmp_path / "art",
            invoke=_ok_invoke(),
        )
    finally:
        blocker.release()
    assert result["ncu_unavailable"] is True
    assert result["reason"] == "lease_unavailable"
    assert result["metrics"] == {}


# ----------------------------------------------------------------- candidate


def test_profile_candidate_builds_snippet_and_profiles_it(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(gpu_lease.POOL_ENV, raising=False)
    result = profile_candidate(
        reference_source="class Model: pass",
        candidate_source="class ModelNew: pass",
        artifact_dir=tmp_path,
        invoke=_ok_invoke(),
    )
    assert result["ncu_unavailable"] is False
    written = (tmp_path / "ncu_target.py").read_text()
    assert "class ModelNew: pass" in written
    assert NVTX_RANGE in written
