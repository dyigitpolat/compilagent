"""Run the 12-candidate Triton + Inductor suite under pydantic_ai + Mistral.

For each of the 6 PyTorch primitives (`scripts/modules.py`) and the 6
Triton primitives (`scripts/kernels/*.py`), drive an
`OptimizationSession` through the `pydantic_ai` harness backed by
`mistral:mistral-large-latest`, and record:

  - baseline_median_ms
  - best_median_ms
  - best_speedup
  - correctness_ok

Workloads are *probed* first (cheap eager forward / kernel launch) and
sorted ascending by baseline cost; the fastest workloads run first and
get a mini-report (`fast_tier_report.json`) emitted as soon as the
bottom half lands. Final per-row results stream into
`scripts/results/suite_results.json` after every candidate so a partial
suite still produces a usable artifact.

Run:
    python -m scripts.run_suite                # all 12
    python -m scripts.run_suite rmsnorm gelu_kernel   # subset
    SUITE_MAX_CANDIDATES=8 python -m scripts.run_suite   # cap proposals
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_imports() -> None:
    import compilagent.integrations.python  # noqa: F401  (registers entry points)
    import compilagent.integrations.torch_inductor  # noqa: F401
    import compilagent.integrations.triton  # noqa: F401
    import compilagent.integrations.pydantic_ai  # noqa: F401


def _result_row(name: str, kind: str, result: Any) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "baseline_median_ms": getattr(result, "baseline_median_ms", None),
        "best_median_ms": getattr(result, "best_median_ms", None),
        "best_speedup": getattr(result, "best_speedup", None),
        "best_candidate_id": getattr(result, "best_candidate_id", None),
        "correctness_ok": getattr(result, "correctness_ok", None),
        "improved": bool(getattr(result, "improved", False)),
        "elapsed_ms": getattr(result, "elapsed_ms", None),
    }


def _error_row(name: str, kind: str, exc: BaseException) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "error": f"{type(exc).__name__}: {exc}",
        "baseline_median_ms": None,
        "best_median_ms": None,
        "best_speedup": None,
        "improved": False,
    }


def _print_row(row: dict[str, Any]) -> None:
    name = row["name"]
    kind = row["kind"]
    if "error" in row:
        print(f"  [{kind:6s}] {name:20s}  ERROR: {row['error']}", flush=True)
        return
    base = row.get("baseline_median_ms")
    best = row.get("best_median_ms")
    sp = row.get("best_speedup")
    ok = row.get("correctness_ok")
    base_s = f"{base:7.3f}" if isinstance(base, (int, float)) else "   n/a "
    best_s = f"{best:7.3f}" if isinstance(best, (int, float)) else "   n/a "
    sp_s = f"{sp:5.3f}x" if isinstance(sp, (int, float)) else "  n/a "
    print(
        f"  [{kind:6s}] {name:20s}  base={base_s} ms  best={best_s} ms  "
        f"speedup={sp_s}  correct={ok}",
        flush=True,
    )


def _run_one_module(name: str, max_candidates: int, model_id: str) -> dict[str, Any]:
    from scripts.modules import MODULE_BUILDERS
    from compilagent.integrations.python import optimize_module

    module, inputs = MODULE_BUILDERS[name]()
    try:
        result = optimize_module(
            module,
            inputs,
            max_candidates=max_candidates,
            harness="pydantic_ai",
            model_id=model_id,
        )
        return _result_row(name, "module", result)
    finally:
        del module
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _run_one_kernel(name: str, max_candidates: int, model_id: str) -> dict[str, Any]:
    from scripts.kernel_specs import KERNEL_BUILDERS
    from compilagent.integrations.python import optimize_kernel

    spec = KERNEL_BUILDERS[name]()
    try:
        result = optimize_kernel(
            spec["kernel"],
            args=spec["args"],
            grid=spec["grid"],
            constexpr=spec["constexpr"],
            max_candidates=max_candidates,
            harness="pydantic_ai",
            model_id=model_id,
        )
        return _result_row(name, "kernel", result)
    finally:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _selected_candidates(argv: list[str]) -> list[tuple[str, str]]:
    from scripts.kernel_specs import KERNEL_BUILDERS
    from scripts.modules import MODULE_BUILDERS

    all_candidates: list[tuple[str, str]] = (
        [(n, "module") for n in MODULE_BUILDERS]
        + [(n, "kernel") for n in KERNEL_BUILDERS]
    )
    if not argv:
        return all_candidates
    wanted = set(argv)
    return [c for c in all_candidates if c[0] in wanted]


def _probe_baseline_ms(name: str, kind: str, *, reps: int = 5) -> float:
    """Cheap eager-mode probe used to rank workloads from fastest → slowest.

    Modules: forward under `inference_mode`. Kernels: direct launch + sync.
    Returns the median over `reps` reps in ms; falls back to `inf` on
    error so failing workloads sink to the bottom of the queue.
    """

    import time
    import torch

    try:
        if kind == "module":
            from scripts.modules import MODULE_BUILDERS
            module, inputs = MODULE_BUILDERS[name]()
            with torch.inference_mode():
                for _ in range(2):
                    module(*inputs)
                torch.cuda.synchronize()
                samples: list[float] = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    module(*inputs)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - t0) * 1000.0)
            del module
        else:
            from scripts.kernel_specs import KERNEL_BUILDERS
            spec = KERNEL_BUILDERS[name]()
            grid = spec["grid"](spec["constexpr"])
            for _ in range(2):
                spec["kernel"][grid](*spec["args"], **spec["constexpr"])
            torch.cuda.synchronize()
            samples = []
            for _ in range(reps):
                t0 = time.perf_counter()
                spec["kernel"][grid](*spec["args"], **spec["constexpr"])
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - t0) * 1000.0)
        torch.cuda.empty_cache()
        return sorted(samples)[len(samples) // 2]
    except BaseException:  # noqa: BLE001
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return float("inf")


def _write_partial(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    _ensure_imports()

    import torch
    if not torch.cuda.is_available():
        print("CUDA not available — Triton + Inductor backends both require GPU.", file=sys.stderr)
        return 2

    max_candidates = int(os.environ.get("SUITE_MAX_CANDIDATES", "12"))
    model_id = os.environ.get("SUITE_MODEL_ID", "mistral:mistral-large-latest")
    out_dir = REPO_ROOT / "scripts" / "results"
    out_path = Path(
        os.environ.get("SUITE_RESULTS_PATH", str(out_dir / "suite_results.json"))
    )
    fast_path_env = os.environ.get("SUITE_FAST_REPORT_PATH")
    if fast_path_env:
        fast_path = Path(fast_path_env)
    else:
        fast_path = out_path.parent / f"{out_path.stem}_fast_tier.json"

    candidates = _selected_candidates(argv)
    if not candidates:
        print("No candidates matched.", file=sys.stderr)
        return 2

    print(
        f"Probing baselines for {len(candidates)} workloads to rank fast → slow…",
        flush=True,
    )
    probed = [(n, k, _probe_baseline_ms(n, k)) for n, k in candidates]
    probed.sort(key=lambda t: t[2])
    for n, k, ms in probed:
        ms_s = f"{ms:8.3f} ms" if ms != float("inf") else "    failed"
        print(f"  probe {k:6s} {n:20s} → {ms_s}", flush=True)

    print(
        f"\nRunning {len(probed)} workloads with pydantic_ai + "
        f"{model_id}, max_candidates={max_candidates}, fast-first.",
        flush=True,
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Output: {out_path}\n", flush=True)

    fast_tier_size = max(1, len(probed) // 2)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    def _payload(rows: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
        return {
            "harness": "pydantic_ai",
            "model_id": model_id,
            "max_candidates_per_workload": max_candidates,
            "elapsed_seconds": elapsed_s,
            "gpu": torch.cuda.get_device_name(0),
            "ordering": [
                {"name": n, "kind": k, "probe_baseline_ms": (None if ms == float("inf") else ms)}
                for n, k, ms in probed
            ],
            "rows": rows,
        }

    for idx, (name, kind, _ms) in enumerate(probed, 1):
        print(f"→ [{idx}/{len(probed)}] {kind}: {name}", flush=True)
        try:
            if kind == "module":
                row = _run_one_module(name, max_candidates, model_id)
            else:
                row = _run_one_kernel(name, max_candidates, model_id)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            row = _error_row(name, kind, exc)
        results.append(row)
        _print_row(row)

        elapsed = time.perf_counter() - started
        _write_partial(out_path, _payload(results, elapsed))
        if idx == fast_tier_size:
            _write_partial(fast_path, _payload(results, elapsed))
            print(
                f"  ↳ fast-tier report ({idx} fastest workloads) → {fast_path}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    _write_partial(out_path, _payload(results, elapsed))
    print(f"\nSuite finished in {elapsed:.1f}s. Results → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
