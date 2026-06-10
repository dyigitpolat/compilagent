"""NCU (Nsight Compute) runner: profile ONE kernel-bearing python snippet.

Backend-side profiling support for the `archetype_ma` (H-MA,
CudaForge-inspired) harness: a gate-passing candidate is re-run under
`ncu` with a curated ~12-metric subset (SM/DRAM throughput % of peak,
achieved occupancy, L1/L2 hit rates, global load/store efficiency, the top
warp-stall reasons, registers per thread, kernel duration), the `--csv`
output is parsed, and the dominant kernel's metrics come back as a dict the
harness can fold into a Judge prompt.

GPU lease (pool mode): when ``COMPILAGENT_GPU_POOL`` is set, the lease is
held around the ENTIRE ncu invocation — NCU's kernel replays multiply the
snippet's GPU time, and all of it is charged to the lease, exactly like the
sandbox charges its compile+gates+timing. The leased device is injected
into the child env as ``CUDA_VISIBLE_DEVICES`` AND passed as ``argv[1]`` to
the target script (which re-exports it before importing torch), because
``sudo`` resets the environment and would otherwise drop the pinning.

Permission fallback (profiling is admin-gated on this machine —
``RmProfilingAdminOnly=1`` — and the sudoers grant may be pending): the
runner first tries plain ``ncu``, then ``sudo -n ncu`` (non-interactive; a
password prompt fails immediately instead of hanging). When both fail, it
DEGRADES GRACEFULLY: ``metrics={}`` plus an ``ncu_unavailable=True`` flag
and a `reason` (`permission` / `binary_missing` / `timeout` /
`no_kernel_rows` / `lease_unavailable` / `ncu_failed`), so the harness can
keep running as a no-NCU H-MA (mirroring CudaForge's own no-NCU ablation)
and the suite row records the degradation.

Only the candidate's measured forward passes are profiled: the snippet
wraps them in an NVTX range (``compilagent_ncu``) and the runner passes
``--nvtx --nvtx-include`` so setup kernels (input randn, state-dict copies,
Triton JIT warmup) neither pollute the metrics nor pay replay cost.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .gpu_lease import GpuLeaseTimeoutError, acquire, pool_devices
from .sandbox import _lease_timeout_seconds

#: Curated metric subset (name → short description for the Judge prompt).
#: ~12 robust counters chosen a priori — a documented deviation from
#: CudaForge's offline Pearson-derived 24-metric whitelist.
CURATED_METRICS: tuple[tuple[str, str], ...] = (
    ("gpu__time_duration.sum", "kernel duration per launch"),
    (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "SM (compute) throughput, % of peak",
    ),
    (
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "DRAM throughput, % of peak",
    ),
    (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "achieved occupancy, % of peak active warps",
    ),
    ("l1tex__t_sector_hit_rate.pct", "L1/TEX cache hit rate, %"),
    ("lts__t_sector_hit_rate.pct", "L2 cache hit rate, %"),
    (
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
        "global load efficiency (bytes used / transferred), %",
    ),
    (
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
        "global store efficiency (bytes used / transferred), %",
    ),
    (
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "warp stall: long scoreboard (global-memory latency), %",
    ),
    (
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "warp stall: short scoreboard (shared-memory), %",
    ),
    (
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "warp stall: barrier, %",
    ),
    (
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "warp stall: wait (fixed-latency dependency), %",
    ),
    ("launch__registers_per_thread", "registers per thread"),
)

DURATION_METRIC = "gpu__time_duration.sum"
NVTX_RANGE = "compilagent_ncu"

#: NCU replays each profiled kernel many times; generous but bounded.
DEFAULT_NCU_TIMEOUT_SECONDS = 600.0
SNIPPET_WARMUP = 2
SNIPPET_ITERATIONS = 3

#: Test seam: (cmd, env, timeout) → (returncode, stdout, stderr).
InvokeFn = Callable[..., tuple[int, str, str]]

_SNIPPET_TEMPLATE = '''"""compilagent NCU profile target — generated, do not edit."""
import os
import sys

# Device pinning must survive `sudo -n ncu` (sudo resets the environment),
# so the leased device travels as argv[1] and is applied pre-torch-import.
if len(sys.argv) > 1 and sys.argv[1]:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

import importlib.util
import pathlib
import tempfile

REFERENCE_SOURCE = {reference!r}
CANDIDATE_SOURCE = {candidate!r}

_workdir = pathlib.Path(tempfile.mkdtemp(prefix="compilagent-ncu-modules-"))


def _load(name, source):
    # Triton 3.x requires @triton.jit functions in a real .py file.
    path = _workdir / (name + ".py")
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


import torch

_ref = _load("compilagent_ncu_reference", REFERENCE_SOURCE)
_cand = _load("compilagent_ncu_candidate", CANDIDATE_SOURCE)

torch.manual_seed(0)
_model_ref = _ref.Model().cuda().eval()
torch.manual_seed(0)
_model = _cand.ModelNew().cuda().eval()
try:
    _model.load_state_dict(_model_ref.state_dict(), strict=False)
except Exception:
    pass

torch.manual_seed(0)
_inputs = _ref.get_inputs()
with torch.no_grad():
    # Warmup OUTSIDE the NVTX range: Triton JIT compile + cache warm
    # launches are neither profiled nor replayed.
    for _ in range({warmup}):
        _model(*_inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push({nvtx_range!r})
    for _ in range({iterations}):
        _model(*_inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
'''


def build_candidate_snippet(
    *,
    reference_source: str,
    candidate_source: str,
    warmup: int = SNIPPET_WARMUP,
    iterations: int = SNIPPET_ITERATIONS,
) -> str:
    """One self-contained kernel-bearing snippet: loads the reference
    (`Model` + `get_inputs`) and candidate (`ModelNew`) as real files, then
    runs `iterations` measured forward passes inside the NVTX range."""

    return _SNIPPET_TEMPLATE.format(
        reference=reference_source,
        candidate=candidate_source,
        warmup=int(warmup),
        iterations=int(iterations),
        nvtx_range=NVTX_RANGE,
    )


# ----------------------------------------------------------------- parsing


def parse_ncu_csv(stdout: str) -> list[dict[str, str]]:
    """Rows of ncu's ``--csv`` output (one row per kernel-launch × metric).

    ncu mixes ``==PROF==``/``==WARNING==`` banner lines into stdout; the CSV
    proper starts at the header row carrying both an ``ID`` and a
    ``Metric Name`` column. Returns ``[]`` when no such table exists.
    """

    lines = stdout.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if "Metric Name" in line and line.lstrip().startswith(('"ID"', "ID,"))
        ),
        None,
    )
    if start is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return [
        row
        for row in reader
        if (row.get("Metric Name") or "").strip()
        and (row.get("Kernel Name") or "").strip()
    ]


def _parse_value(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip().strip('"').replace(",", "")  # "1,234.56" → 1234.56
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def curate_metrics(
    rows: Sequence[dict[str, str]],
    metrics: Sequence[tuple[str, str]] = CURATED_METRICS,
) -> dict[str, Any]:
    """Reduce raw rows to the curated block for the DOMINANT kernel.

    A Triton candidate may launch several distinct kernels per forward;
    metrics are reported for the kernel with the largest total
    ``gpu__time_duration.sum`` across its launches (falling back to the
    most-often-launched kernel when the duration metric is absent), each
    metric averaged over that kernel's launches.
    """

    per_kernel: dict[str, dict[str, list[float]]] = {}
    units: dict[str, str] = {}
    for row in rows:
        kernel = (row.get("Kernel Name") or "").strip()
        name = (row.get("Metric Name") or "").strip()
        value = _parse_value(row.get("Metric Value"))
        if not kernel or not name or value is None:
            continue
        per_kernel.setdefault(kernel, {}).setdefault(name, []).append(value)
        unit = (row.get("Metric Unit") or "").strip()
        if unit:
            units.setdefault(name, unit)
    if not per_kernel:
        return {"metrics": {}, "dominant_kernel": None, "kernel_names": []}

    def _weight(kernel: str) -> tuple[float, int]:
        samples = per_kernel[kernel]
        return (
            sum(samples.get(DURATION_METRIC, ())),
            sum(len(v) for v in samples.values()),
        )

    dominant = max(per_kernel, key=_weight)
    curated: dict[str, Any] = {}
    for name, description in metrics:
        values = per_kernel[dominant].get(name)
        if not values:
            continue
        curated[name] = {
            "value": round(sum(values) / len(values), 4),
            "unit": units.get(name, ""),
            "description": description,
        }
    return {
        "metrics": curated,
        "dominant_kernel": dominant,
        "kernel_names": sorted(per_kernel),
    }


def format_metrics_block(profile: dict[str, Any]) -> str:
    """Human-readable curated-metrics block for the Judge prompt; empty
    string when the profile is degraded (``ncu_unavailable``) or empty."""

    metrics = profile.get("metrics") or {}
    if profile.get("ncu_unavailable") or not metrics:
        return ""
    kernels = profile.get("kernel_names") or []
    others = (
        f" (other kernels seen: {', '.join(k for k in kernels if k != profile.get('dominant_kernel'))})"
        if len(kernels) > 1
        else ""
    )
    lines = [
        "NCU profile of the dominant kernel "
        f"`{profile.get('dominant_kernel')}`{others}:"
    ]
    for name, entry in metrics.items():
        unit = f" {entry['unit']}" if entry.get("unit") else ""
        lines.append(f"  - {entry['description']} [{name}]: {entry['value']}{unit}")
    return "\n".join(lines)


# ----------------------------------------------------------------- running


def _degraded(reason: str, error: str) -> dict[str, Any]:
    return {
        "ncu_unavailable": True,
        "reason": reason,
        "error": error,
        "invocation": None,
        "metrics": {},
        "dominant_kernel": None,
        "kernel_names": [],
    }


def _invoke(
    cmd: list[str],
    *,
    env: dict[str, str] | None,
    timeout: float,
) -> tuple[int, str, str]:
    proc = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, timeout=timeout, env=env
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _classify(errors: list[str]) -> str:
    joined = " ".join(errors)
    if "ERR_NVGPUCTRPERM" in joined or "password is required" in joined:
        return "permission"
    if errors and all("binary not found" in e for e in errors):
        return "binary_missing"
    if "no metric rows" in joined:
        return "no_kernel_rows"
    return "ncu_failed"


def _profile_target(
    target: Path,
    metric_names: Sequence[str],
    *,
    env: dict[str, str] | None,
    device: str,
    timeout_seconds: float,
    invoke: InvokeFn,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Try plain ``ncu`` then ``sudo -n ncu`` on one target script."""

    ncu_path = shutil.which("ncu") or "ncu"
    base = [
        "--csv",
        "--metrics",
        ",".join(metric_names),
        "--target-processes",
        "all",
        "--nvtx",
        "--nvtx-include",
        f"{NVTX_RANGE}/",
        sys.executable,
        str(target),
        device,
    ]
    attempts: tuple[tuple[str, list[str]], ...] = (
        ("ncu", [ncu_path, *base]),
        # -n: non-interactive — fail fast instead of prompting when the
        # sudoers grant hasn't landed.
        ("sudo -n ncu", ["sudo", "-n", ncu_path, *base]),
    )
    errors: list[str] = []
    for index, (label, cmd) in enumerate(attempts):
        try:
            returncode, stdout, stderr = invoke(
                cmd, env=env, timeout=timeout_seconds
            )
        except FileNotFoundError as exc:
            errors.append(f"{label}: binary not found ({exc})")
            continue
        except subprocess.TimeoutExpired:
            # A timeout means ncu WAS running (replays in flight) — retrying
            # under sudo would only double the wallclock.
            return _degraded(
                "timeout",
                f"{label}: ncu exceeded the {timeout_seconds:.0f}s hard "
                "timeout and was killed.",
            )
        _write_log(artifact_dir, f"ncu_attempt{index}_stdout.log", stdout)
        _write_log(artifact_dir, f"ncu_attempt{index}_stderr.log", stderr)
        if returncode != 0:
            tail = (stderr or stdout)[-1000:]
            errors.append(f"{label}: exit {returncode}: {tail}")
            continue
        rows = parse_ncu_csv(stdout)
        if not rows:
            errors.append(
                f"{label}: ncu succeeded but emitted no metric rows "
                "(NVTX range matched no kernel launches?)"
            )
            continue
        curated = curate_metrics(rows)
        return {
            "ncu_unavailable": False,
            "reason": None,
            "error": None,
            "invocation": label,
            **curated,
        }
    return _degraded(_classify(errors), " | ".join(errors))


def _write_log(artifact_dir: Path, name: str, text: str) -> None:
    # Logging is best-effort; never fail the profile over it.
    with contextlib.suppress(OSError):
        (artifact_dir / name).write_text(text, encoding="utf-8")


def profile_snippet(
    *,
    snippet_source: str,
    artifact_dir: Path,
    metrics: Sequence[tuple[str, str]] = CURATED_METRICS,
    timeout_seconds: float = DEFAULT_NCU_TIMEOUT_SECONDS,
    invoke: InvokeFn | None = None,
) -> dict[str, Any]:
    """Profile one kernel-bearing snippet under ncu; never raises.

    Returns a dict with ``ncu_unavailable`` / ``reason`` / ``error`` /
    ``invocation`` / ``metrics`` / ``dominant_kernel`` / ``kernel_names``
    (+ ``gpu_lease`` in pool mode). Degraded results always carry
    ``metrics == {}`` and ``ncu_unavailable == True``.
    """

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / "ncu_target.py"
    target.write_text(snippet_source, encoding="utf-8")
    metric_names = [name for name, _ in metrics]
    run = invoke or _invoke

    if not pool_devices():
        # No pool: inherit the parent's pinning (env untouched, argv empty).
        return _profile_target(
            target,
            metric_names,
            env=None,
            device="",
            timeout_seconds=timeout_seconds,
            invoke=run,
            artifact_dir=artifact_dir,
        )

    # Pool mode: ONE lease around the whole invocation — including the
    # sudo fallback and every NCU replay — so no other sandbox/profile can
    # share the device while replays run.
    try:
        lease = acquire(timeout=_lease_timeout_seconds())
    except (GpuLeaseTimeoutError, ValueError, OSError) as exc:
        return _degraded("lease_unavailable", f"GPU lease unavailable: {exc}")
    with lease:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": lease.device}
        result = _profile_target(
            target,
            metric_names,
            env=env,
            device=lease.device,
            timeout_seconds=timeout_seconds,
            invoke=run,
            artifact_dir=artifact_dir,
        )
    result["gpu_lease"] = {
        "device": lease.device,
        "lease_wait_s": round(lease.wait_seconds, 3),
    }
    return result


def profile_candidate(
    *,
    reference_source: str,
    candidate_source: str,
    artifact_dir: Path,
    timeout_seconds: float = DEFAULT_NCU_TIMEOUT_SECONDS,
    invoke: InvokeFn | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: build the candidate snippet and profile it."""

    snippet = build_candidate_snippet(
        reference_source=reference_source, candidate_source=candidate_source
    )
    return profile_snippet(
        snippet_source=snippet,
        artifact_dir=artifact_dir,
        timeout_seconds=timeout_seconds,
        invoke=invoke,
    )
