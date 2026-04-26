"""Reproducible effectiveness study.

Runs the optimizer across a 3-axis grid and records JSON results suitable for
plotting. Each cell is run 3 times (different RNG seeds) so the plot can show
mean ± stddev:

  - harness:      {pydantic_ai, claude_agent_sdk}
  - workload:     {vit_block (pytorch / inductor), vector_add (triton)}
  - max_candidates: {4, 8, 12, 16, 20}
  - seed:         {0, 1, 2}

Total: 2 × 2 × 5 × 3 = 60 runs.

Each run records:

  - baseline_median_ms
  - best_speedup, best_median_ms
  - best_correctness_ok, best_max_abs_diff
  - successful_count, failed_attempts
  - elapsed_ms
  - final_text (the agent's report)
  - candidates: list of {id, description, changes, speedup, median_ms}

Output: `runs/study/<timestamp>/results.jsonl` (one line per run).
Plotting: `python scripts/experiments/plot_study.py <results.jsonl>`.

Usage:
    env/bin/python scripts/experiments/run_study.py
    env/bin/python scripts/experiments/run_study.py \\
        --harnesses pydantic_ai \\
        --workloads vit_block \\
        --trials 4 8

Both harnesses use the same Mistral model unless `--model` is overridden.
Failed runs are recorded with `error` set; the study continues to the next
cell so a transient failure doesn't trash the whole grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch  # noqa: F401  — must precede triton/inductor imports

from compilagent_triton.backends import import_backend_packages
from compilagent_triton.study import CellResult
from compilagent_triton.workloads.registry import import_workload_packages


# ---------------------------------------------------------------------------
# Grid + cell driver
# ---------------------------------------------------------------------------


WORKLOADS = ("vit_block", "vector_add")
HARNESSES = ("pydantic_ai", "claude_agent_sdk")
TRIALS = (4, 8, 12, 16, 20)
SEEDS = (0, 1, 2)




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--harnesses", nargs="+", default=list(HARNESSES),
        choices=HARNESSES,
        help="Which harnesses to run.",
    )
    parser.add_argument(
        "--workloads", nargs="+", default=list(WORKLOADS),
        help="Workload ids to optimize.",
    )
    parser.add_argument(
        "--trials", nargs="+", type=int, default=list(TRIALS),
        help="max_candidates values to sweep.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS),
        help="RNG seeds to run for each cell.",
    )
    parser.add_argument(
        "--model", default="mistral:mistral-large-latest",
        help="LLM provider:model string for pydantic_ai cells. Default Mistral large.",
    )
    parser.add_argument(
        "--sdk-model", default="anthropic:claude-opus-4-7",
        help=(
            "LLM provider:model string for claude_agent_sdk cells. The SDK is "
            "the `claude` CLI under the hood and only routes to Anthropic "
            "models, so this is automatically substituted when `--model` is "
            "non-Anthropic."
        ),
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory. Default: runs/study/<timestamp>/.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke run: 1 seed × 1 trial setting.",
    )
    parser.add_argument(
        "--parallelism", type=int, default=1,
        help=(
            "How many cells to run concurrently. Each cell runs in its own "
            "subprocess pinned to a specific GPU (round-robin via "
            "CUDA_VISIBLE_DEVICES) so cells don't fight for memory."
        ),
    )
    parser.add_argument(
        "--gpus", default=None,
        help=(
            "Comma-separated list of GPU indices to schedule cells onto "
            "(e.g. `0,1,2,3`). Defaults to every visible GPU reported by "
            "nvidia-smi."
        ),
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds = [0]
        args.trials = [4]

    if not torch.cuda.is_available():
        print("CUDA is required to run the study.", file=sys.stderr)
        return 1

    # Self-register backends + workloads.
    import_backend_packages()
    import_workload_packages()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_root = Path(args.out or f"runs/study/{timestamp}")
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.jsonl"

    cells = [
        (h, w, t, s)
        for h in args.harnesses
        for w in args.workloads
        for t in args.trials
        for s in args.seeds
    ]
    print(f"Running {len(cells)} cells. Results: {results_path}")
    started = time.perf_counter()

    # Each cell runs in its own subprocess so a torch / triton / CUDA-cleanup
    # segfault in one cell doesn't take down the rest of the sweep. With
    # `--parallelism > 1`, multiple subprocesses run concurrently, each
    # pinned to one GPU via `CUDA_VISIBLE_DEVICES` so they don't fight for
    # memory.
    import os as _os
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    else:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            )
            gpu_ids = [int(line.strip()) for line in res.stdout.splitlines()
                       if line.strip().isdigit()]
        except Exception:  # noqa: BLE001
            gpu_ids = [0]
    if not gpu_ids:
        gpu_ids = [0]
    print(f"Scheduling cells across GPUs {gpu_ids} with parallelism={args.parallelism}")

    def _run_one(idx_cell: tuple[int, tuple[str, str, int, int]]) -> tuple[int, dict, float]:
        idx, (h, w, t, s) = idx_cell
        gpu = gpu_ids[idx % len(gpu_ids)]
        cell_t0 = time.perf_counter()
        cell_result_path = (
            out_root / "cells" /
            f"{h}__{w}__t{t}__s{s}" / "cell_result.json"
        )
        cell_result_path.parent.mkdir(parents=True, exist_ok=True)
        cell_result_path.unlink(missing_ok=True)
        cmd = [
            sys.executable, "-u", "-m",
            "compilagent_triton._study_cell_runner",
            "--harness", h, "--workload", w,
            "--trials", str(t), "--seed", str(s),
            "--model", args.model, "--sdk-model", args.sdk_model,
            "--out", str(out_root),
            "--result-path", str(cell_result_path),
        ]
        env = {**_os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        try:
            rc = subprocess.run(
                cmd, check=False, timeout=1800, env=env,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = 124
        cell_elapsed = time.perf_counter() - cell_t0
        if cell_result_path.exists():
            with cell_result_path.open() as f:
                row = json.loads(f.read())
        else:
            row = asdict(CellResult(
                harness=h, workload_id=w, backend_id="",
                max_candidates=t, seed=s, model_name=args.model,
                baseline_median_ms=None, best_speedup=None,
                best_candidate_id=None, best_median_ms=None,
                best_correctness_ok=None, best_max_abs_diff=None,
                successful_count=0, failed_attempts=0,
                elapsed_ms=cell_elapsed * 1000, final_text=None,
                candidates=[],
                correctness_recheck_ok=None,
                correctness_recheck_max_abs_diff=None,
                error=f"subprocess returncode={rc} (no result.json written; "
                      "likely segfault or OOM)",
                timestamp=datetime.now(UTC).isoformat(),
            ))
        row["_gpu"] = gpu
        return idx, row, cell_elapsed

    rows_by_idx: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as pool:
        futures = {
            pool.submit(_run_one, (i, cell)): (i, cell)
            for i, cell in enumerate(cells, 1)
        }
        # Announce launches as we submit so the user sees the schedule.
        for i, cell in enumerate(cells, 1):
            h, w, t, s = cell
            print(f"[{i}/{len(cells)}] queued: harness={h} workload={w} "
                  f"trials={t} seed={s} -> gpu {gpu_ids[(i-1) % len(gpu_ids)]}",
                  flush=True)
        for fut in as_completed(futures):
            idx, row, cell_elapsed = fut.result()
            rows_by_idx[idx] = row
            h, w, t, s = futures[fut][1]
            if row.get("error"):
                first_line = (row["error"] or "").splitlines()[0]
                print(f"[{idx}/{len(cells)}] DONE ERROR ({cell_elapsed:.1f}s) "
                      f"{w} gpu={row.get('_gpu')}: {first_line}", flush=True)
            else:
                sp = f"{row['best_speedup']:.4f}x" if row.get("best_speedup") else "n/a"
                ok = row.get("correctness_recheck_ok")
                ok_str = "OK" if ok is True else "FAIL" if ok is False else "n/a"
                print(f"[{idx}/{len(cells)}] DONE {w} gpu={row.get('_gpu')}: "
                      f"speedup={sp} recheck={ok_str} "
                      f"successful={row.get('successful_count', 0)}/{t} "
                      f"({cell_elapsed:.1f}s)", flush=True)
    # Write rows in submission order so results.jsonl matches the cells grid.
    with results_path.open("w", encoding="utf-8") as out:
        for idx in sorted(rows_by_idx):
            out.write(json.dumps(rows_by_idx[idx], default=str) + "\n")

    elapsed = time.perf_counter() - started
    print(f"\nStudy complete. {len(cells)} cells in {elapsed/60:.1f} min.")
    print(f"Results: {results_path}")
    print(f"Plot:    env/bin/python scripts/experiments/plot_study.py {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
