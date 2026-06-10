"""Unseen-config holdout evaluation over a pilot results JSONL (D10).

For every results row carrying a gate-passing winning candidate
(`best_candidate_id` — `run_pilot` only promotes candidates that passed all
E2a gates with speedup > 1), this driver:

  1. reads the candidate module source from the run's workspace artifacts
     (``.compilagent/workloads/<wl>/runs/<run>/candidates/<cid>/candidate_module.py``),
  2. generates the workload's ~6 held-out input configs
     (`triton_source.holdout`, one per category),
  3. re-evaluates the candidate on each config in the SAME subprocess
     sandbox used during search — full correctness gates (g1/g2/g3/g5) +
     CUDA-event timing of candidate AND reference — GPU-only, one pool lease
     per sandbox run (``COMPILAGENT_GPU_POOL``, defaulting to "1" here),
  4. appends one JSONL row per candidate with
     ``{seen_speedup, unseen_correct (per config), unseen_speedup_geomean,
     delta_g}`` where ``delta_g = unseen_speedup_geomean - seen_speedup``
     (negative ⇒ the seen-config speedup does not generalize).

Resumable: rows whose key is already present (without error) are skipped.

Example:

    COMPILAGENT_GPU_POOL=1 env/bin/python -m scripts.eval_holdout \
        --results scripts/results/t1lite_or.jsonl \
        --out scripts/results/holdout_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_pilot import _append_row, _completed_keys  # noqa: E402

_GATE_ORDER = (
    "g1_shape_dtype",
    "g2_allclose",
    "g3_no_alias_no_mutation",
    "g5_determinism",
)


def _geomean(values: list[float]) -> float | None:
    positive = [v for v in values if v and v > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def _winner_rows(results_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("best_candidate_id") and row.get("run_id"):
            rows.append(row)
    return rows


def _candidate_source_path(
    workspace_root: Path, workload: str, run_id: str, candidate_id: str
) -> Path:
    return (
        workspace_root
        / "workloads"
        / workload
        / "runs"
        / run_id
        / "candidates"
        / candidate_id
        / "candidate_module.py"
    )


def _eval_candidate_on_config(
    *,
    spec: Any,
    candidate_source: str,
    config: Any,
    artifact_dir: Path,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    from compilagent.integrations.triton_source._internal.sandbox import (
        DEFAULT_TIMEOUT_SECONDS,
        run_sandboxed_eval,
    )
    from compilagent.integrations.triton_source.holdout import apply_to_reference

    reference = str(spec.metadata.get("reference_module_source", "") or "")
    result = run_sandboxed_eval(
        reference_source=apply_to_reference(reference, config),
        candidate_source=candidate_source,
        artifact_dir=artifact_dir,
        atol=spec.tolerance.atol,
        rtol=spec.tolerance.rtol,
        timeout_seconds=float(
            spec.metadata.get("sandbox_timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        ),
        warmup=warmup,
        repetitions=repetitions,
    )
    gates = result.get("gates") or {}
    correct = bool(gates) and all(
        gates.get(g, {}).get("ok", False) for g in _GATE_ORDER
    )
    lease = result.get("gpu_lease") or {}
    return {
        "category": config.category,
        "input_shapes": config.input_shapes,
        "correct": correct,
        "gates": {
            g: gates.get(g, {}).get("ok") for g in _GATE_ORDER if g in gates
        },
        "speedup_vs_ref": result.get("speedup_vs_ref"),
        "cand_ms": result.get("cand_ms"),
        "ref_ms": result.get("ref_ms"),
        "error": result.get("error"),
        "timed_out": bool(result.get("timed_out")),
        "gpu_device": lease.get("device"),
        "gpu_lease_wait_s": lease.get("lease_wait_s"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="pilot results JSONL (e.g. scripts/results/t1lite_or.jsonl)")
    parser.add_argument(
        "--workspace", default=str(REPO_ROOT / ".compilagent"),
        help="workspace root holding the runs' candidate artifacts",
    )
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "scripts" / "results" / "holdout_results.jsonl"),
    )
    parser.add_argument(
        "--artifacts",
        default=str(REPO_ROOT / "scripts" / "results" / "holdout_artifacts"),
        help="where per-config sandbox payloads/logs land",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate at most N candidates (0 = all)")
    args = parser.parse_args(argv)

    # GPU-only via leases: every sandbox run leases one pool device. GPU 0
    # is reserved on this machine; default the pool to GPU 1.
    os.environ.setdefault("COMPILAGENT_GPU_POOL", "1")

    import compilagent.integrations.triton_source  # noqa: F401  (registers specs)
    from compilagent.core.workload_registry import workload_registry
    from compilagent.integrations.triton_source.holdout import generate_holdout_configs

    results_path = Path(args.results)
    workspace_root = Path(args.workspace)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_root = Path(args.artifacts)
    done = _completed_keys(out_path)

    rows = _winner_rows(results_path)
    print(f"{len(rows)} winner row(s) in {results_path}", flush=True)
    evaluated = 0
    for row in rows:
        workload = row["workload"]
        run_id = row["run_id"]
        candidate_id = row["best_candidate_id"]
        key = f"{workload}|{run_id}|{candidate_id}"
        if key in done:
            print(f"skip (done): {key}", flush=True)
            continue
        if args.limit and evaluated >= args.limit:
            break

        base = {
            "key": key,
            "workload": workload,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "harness": row.get("harness"),
            "budget": row.get("budget"),
            "seed": row.get("seed"),
            "model_id": row.get("model_id"),
            "seen_speedup": row.get("best_speedup"),
            "timestamp": time.time(),
        }
        source_path = _candidate_source_path(
            workspace_root, workload, run_id, candidate_id
        )
        if not source_path.exists():
            _append_row(out_path, {**base, "error": f"missing artifact {source_path}"})
            continue
        try:
            spec = workload_registry.get_spec(workload)
            configs = generate_holdout_configs(spec)
        except (KeyError, ValueError) as exc:
            _append_row(out_path, {**base, "error": f"{type(exc).__name__}: {exc}"})
            continue

        candidate_source = source_path.read_text(encoding="utf-8")
        config_rows = [
            _eval_candidate_on_config(
                spec=spec,
                candidate_source=candidate_source,
                config=config,
                artifact_dir=(
                    artifacts_root / workload / run_id / candidate_id / config.category
                ),
                warmup=args.warmup,
                repetitions=args.reps,
            )
            for config in configs
        ]
        speedups = [
            c["speedup_vs_ref"]
            for c in config_rows
            if c["correct"] and isinstance(c["speedup_vs_ref"], (int, float))
        ]
        geomean = _geomean(speedups)
        seen = base["seen_speedup"]
        out_row = {
            **base,
            "unseen_correct": {c["category"]: c["correct"] for c in config_rows},
            "unseen_correct_fraction": (
                sum(c["correct"] for c in config_rows) / len(config_rows)
            ),
            "unseen_speedup_geomean": geomean,
            "delta_g": (
                geomean - seen
                if geomean is not None and isinstance(seen, (int, float))
                else None
            ),
            "configs": config_rows,
            "timestamp": time.time(),
        }
        _append_row(out_path, out_row)
        evaluated += 1
        print(
            f"✓ {key} seen={seen if seen is None else round(seen, 3)} "
            f"unseen_geomean={geomean if geomean is None else round(geomean, 3)} "
            f"correct={out_row['unseen_correct_fraction']:.2f}",
            flush=True,
        )
    print(f"holdout results → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
