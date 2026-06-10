"""Import-smoke for the 24 KernelBench-derived workloads (D8).

Two stages per workload:

  1. import smoke — the manifest-backed spec registered cleanly (checked by
     importing the integration on this interpreter, no GPU),
  2. baseline smoke — `TritonSourceBackend.compile(spec, Plan())` runs the
     reference module through the subprocess sandbox: compile + CUDA-event
     timing, ONE pool-device lease per workload (the sandbox leases via
     ``COMPILAGENT_GPU_POOL``, defaulting to "1" here — never pin a busy
     device directly).

Appends one JSONL row per workload (resumable: existing non-error rows are
skipped) with the reference timing, torch/device info, and lease wait.

    COMPILAGENT_GPU_POOL=1 env/bin/python -m scripts.smoke_kernelbench \
        --out scripts/results/kb24_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_pilot import _append_row, _completed_keys  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "scripts" / "results" / "kb24_smoke.jsonl")
    )
    parser.add_argument("--ids", default="", help="comma-separated subset of workload ids")
    args = parser.parse_args(argv)

    os.environ.setdefault("COMPILAGENT_GPU_POOL", "1")

    import compilagent.integrations.triton_source  # noqa: F401
    from compilagent.core.backend import backend_registry
    from compilagent.core.plan import Plan
    from compilagent.core.workload_registry import workload_registry
    from compilagent.integrations.triton_source.kernelbench_workloads import (
        KB_WORKLOAD_IDS,
    )

    ids = [i.strip() for i in args.ids.split(",") if i.strip()] or list(KB_WORKLOAD_IDS)
    backend = backend_registry.get("triton_source")
    capability = backend.device_capability()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(out_path)

    print(
        f"{len(ids)} workload(s); device {capability.name} ({capability.arch}); "
        f"pool [{os.environ['COMPILAGENT_GPU_POOL']}] → {out_path}",
        flush=True,
    )
    failures = 0
    for workload_id in ids:
        if workload_id in done:
            print(f"skip (done): {workload_id}", flush=True)
            continue
        started = time.perf_counter()
        spec = workload_registry.get_spec(workload_id)
        with tempfile.TemporaryDirectory(prefix=f"kb-smoke-{workload_id}-") as tmp:
            compile_result = backend.compile(spec, Plan(), artifact_dir=Path(tmp))
        evaluation = (compile_result.metadata or {}).get("evaluation") or {}
        lease = evaluation.get("gpu_lease") or {}
        row = {
            "key": workload_id,
            "workload": workload_id,
            "kb_file": (spec.metadata.get("kernelbench") or {}).get("kb_file"),
            "family": (spec.metadata.get("kernelbench") or {}).get("family"),
            "compiled": bool(compile_result.ok),
            "ref_trimmed_mean_ms": evaluation.get("ref_ms"),
            "gpu_device": lease.get("device"),
            "gpu_lease_wait_s": lease.get("lease_wait_s"),
            "torch_arch": capability.arch,
            "device_name": capability.name,
            "wallclock_s": round(time.perf_counter() - started, 1),
            "timestamp": time.time(),
        }
        if not compile_result.ok:
            row["error"] = compile_result.diagnostics or "baseline compile failed"
            failures += 1
        _append_row(out_path, row)
        print(
            f"{'✓' if compile_result.ok else '✗'} {workload_id} "
            f"ref_ms={row['ref_trimmed_mean_ms']} gpu={row['gpu_device']} "
            f"wait={row['gpu_lease_wait_s']}s wall={row['wallclock_s']}s"
            + (f" error={row.get('error', '')[:200]}" if not compile_result.ok else ""),
            flush=True,
        )
    print(json.dumps({"total": len(ids), "failures": failures}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
