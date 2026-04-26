"""Subprocess entry point for one study cell.

Invoked by `scripts/experiments/run_study.py` per cell so a torch / triton /
CUDA-cleanup segfault is contained — the parent reads the per-cell result
JSON if it exists, synthesises an error row otherwise.

Cleanup-hang protection:
  - The cell logic runs in a daemon worker thread that writes the result
    file as soon as `run_cell` returns.
  - The main thread polls for the result file. As soon as it appears, the
    main thread calls `os._exit(0)` immediately, skipping every Python
    finalizer and torch / CUDA cleanup that has been observed to deadlock or
    segfault on this host (sm_120, torch 2.x, triton).
  - If the worker hits an exception before writing the file, the watchdog
    times out after `--watchdog-seconds` and exits with code 99 — the parent
    then synthesises an error row.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import torch  # noqa: F401  — must precede triton/inductor imports

from .backends import import_backend_packages
from .study import run_cell
from .workloads.registry import import_workload_packages


def _worker(args, error_path: Path) -> None:
    """Run the cell. The result file is written by `run_cell` itself via the
    `early_write_path` arg — that happens *before* any cleanup runs, so even
    if torch / triton finalizers segfault on the way out, the parent already
    has the row on disk."""

    try:
        run_cell(
            harness=args.harness, workload_id=args.workload,
            max_candidates=args.trials, seed=args.seed,
            model_name=args.model, sdk_model_name=args.sdk_model,
            out_root=args.out,
            early_write_path=args.result_path,
        )
    except Exception:  # noqa: BLE001
        with error_path.open("w", encoding="utf-8") as f:
            f.write(traceback.format_exc())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--harness", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--trials", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sdk-model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--result-path", type=Path, required=True)
    p.add_argument(
        "--watchdog-seconds", type=int, default=900,
        help="Hard upper bound on cell wall-clock; the main thread force-exits past this.",
    )
    args = p.parse_args()

    import_backend_packages()
    import_workload_packages()

    error_path = args.result_path.with_suffix(".error.txt")
    worker = threading.Thread(target=_worker, args=(args, error_path), daemon=True)
    worker.start()

    deadline = time.time() + max(60, args.watchdog_seconds)
    poll = 0.5
    # Spin until the worker writes the result file OR the watchdog fires.
    # Once the file is present, the main thread calls `_exit(0)` immediately —
    # we never let Python's atexit / torch / CUDA finalizers run because they
    # deadlock on this host after some workloads.
    while time.time() < deadline:
        if args.result_path.exists():
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0)
        if error_path.exists():
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(2)
        if not worker.is_alive():
            # Worker died without writing a file or an error — treat as error.
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(3)
        time.sleep(poll)
    print(f"CELL: watchdog fired at {args.watchdog_seconds}s", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(99)


if __name__ == "__main__":
    main()
