"""Pilot grid driver: (harness × workload × budget × seed) cells (E-ticket DRIVER).

Extends scripts/run_archetype_smoke.py to a resumable multi-GPU grid:

  - pinned mode (``--gpus "1,2,3"``): one session at a time per GPU — a
    simple work queue feeds one worker process per device; each worker
    pins itself with CUDA_VISIBLE_DEVICES *before* importing torch.
  - episode-parallel pool mode (``--gpu-pool "1,2,3" --episode-workers 9``,
    mutually exclusive with --gpus): episodes are LLM-latency-dominated
    (GPUs idle at ~0-3% while pinned workers wait on chat completions), so
    N >> #GPUs unpinned workers drain the same queue and the ONLY GPU-bound
    unit — the triton_source sandbox subprocess — leases one pool device
    per run via fcntl.flock (`triton_source._internal.gpu_lease`); workers
    export COMPILAGENT_GPU_POOL and never set CUDA_VISIBLE_DEVICES.
  - one suite-row JSON per cell, appended to a JSONL results file
    (speedup, per-candidate gate verdicts, tokens in/out, $-estimate at
    mistral-large pricing, wallclock, llm_calls, E/V counts, GPU lease
    waits in pool mode, judge metadata for cascade cells). Appends are
    serialized across processes with flock on a sidecar lock file.
  - resumable: cells whose key already has a non-error row in the JSONL
    are skipped; errored cells are retried on the next invocation.
  - Mistral rate limits: the process-global throttle in
    archetype_harnesses._llm (1.2 s min-interval between request starts,
    ≤2 in-flight) applies inside every worker process; both knobs are
    forwarded from --llm-min-interval / --llm-max-concurrent. NB: in pool
    mode the throttle stays per-process, so the aggregate request rate
    scales with --episode-workers — the 429-retry inside DirectChatLLM
    absorbs the overflow.

Examples (GPU 0 is reserved on this machine — never include it):

    MISTRAL_API_KEY=... python -m scripts.run_pilot \
        --harnesses archetype_evo,archetype_band,cascade \
        --workloads softmax_4096 --budgets 4 --seeds 13 --gpus 1 \
        --out results/pilot.jsonl

    OPENROUTER_API_KEY=... python -m scripts.run_pilot \
        --harnesses archetype_sr,cascade --workloads softmax_4096 \
        --budgets 8 --seeds 13,42 --gpu-pool 1,2,3 --episode-workers 9 \
        --out results/pilot.jsonl
"""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: USD per 1M tokens, mistral-large-latest (la Plateforme list price).
MISTRAL_LARGE_PRICING_PER_1M = {"input": 2.0, "output": 6.0}


def cell_key(cell: dict[str, Any]) -> str:
    return (
        f"{cell['harness']}|{cell['workload']}|{cell['budget']}"
        f"|{cell['seed']}|{cell['model_id']}|{cell.get('cascade_disable', '')}"
    )


def estimate_cost_usd(tokens_in: int, tokens_out: int) -> float:
    return (
        tokens_in / 1e6 * MISTRAL_LARGE_PRICING_PER_1M["input"]
        + tokens_out / 1e6 * MISTRAL_LARGE_PRICING_PER_1M["output"]
    )


def _candidate_rows(session: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cid, c in session.candidates.items():
        compile_outcome = c.get("compile")
        timing = c.get("timing")
        correctness = c.get("correctness")
        compile_meta = getattr(compile_outcome, "metadata", {}) or {}
        lease = _gpu_lease_of(compile_meta)
        rows.append(
            {
                "candidate_id": cid,
                "description": c.get("description", ""),
                "compile_ok": getattr(compile_outcome, "ok", None),
                "gates": compile_meta.get("gates"),
                "median_ms": getattr(timing, "median_ms", None),
                "speedup_vs_baseline": c.get("speedup"),
                "correctness_ok": getattr(correctness, "ok", None),
                "diagnostics": getattr(compile_outcome, "diagnostics", None),
                # Pool mode only (None/absent wait otherwise): which device
                # the sandbox leased and how long it queued for it.
                "gpu_device": lease.get("device"),
                "gpu_lease_wait_s": lease.get("lease_wait_s"),
            }
        )
    return rows


def _gpu_lease_of(compile_metadata: dict[str, Any]) -> dict[str, Any]:
    evaluation = compile_metadata.get("evaluation") or {}
    return evaluation.get("gpu_lease") or {}


def _gate_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    passed = 0
    failed = 0
    compile_failed = 0
    failures_by_gate: dict[str, int] = {}
    for row in candidates:
        gates = row.get("gates") or {}
        failing = [k for k, v in gates.items() if not (v or {}).get("ok", False)]
        if failing:
            failed += 1
            for gate in failing:
                failures_by_gate[gate] = failures_by_gate.get(gate, 0) + 1
        elif row.get("compile_ok") is True:
            passed += 1
        elif row.get("compile_ok") is False:
            # Crashed before any gate verdict (e.g. Triton CompilationError
            # inside the sandbox) — not a gate pass.
            compile_failed += 1
    return {
        "gate_passing": passed,
        "gate_failing": failed,
        "compile_failed": compile_failed,
        "failures_by_gate": failures_by_gate,
    }


def _best_validated(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    qualifying = [
        r
        for r in rows
        if isinstance(r.get("speedup_vs_baseline"), (int, float))
        and r["speedup_vs_baseline"] > 1.0
        and (r.get("correctness_ok") is None or r["correctness_ok"])
    ]
    qualifying.sort(key=lambda r: r["speedup_vs_baseline"], reverse=True)
    return qualifying[0] if qualifying else None


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Run one (harness, workload, budget, seed) session and build its row."""

    import asyncio

    import compilagent.integrations.archetype_harnesses  # noqa: F401
    import compilagent.integrations.pydantic_ai  # noqa: F401  (model resolution)
    import compilagent.integrations.triton_source  # noqa: F401
    from compilagent.harness.base import HarnessRunRequest
    from compilagent.harness.registry import harness_registry
    from compilagent.integrations.archetype_harnesses import ExperimentLogPolicy
    from compilagent.session.session import OptimizationSession, run_session
    from compilagent.storage.trace_store import TraceStore
    from compilagent.storage.workspace import OptimizationWorkspace

    started = time.perf_counter()
    workspace = OptimizationWorkspace(session_cwd=Path.cwd()).ensure()
    sink = TraceStore(workspace.root).ensure()

    cascade_disable = [
        d for d in str(cell.get("cascade_disable", "")).split(",") if d.strip()
    ]
    # C6 lives half in the policy: attach ExperimentLogPolicy for cascade
    # cells unless the memory bundle / c6 is disabled.
    policy = None
    if cell["harness"] == "cascade" and not (
        {"c6", "memory"} & {d.strip().lower() for d in cascade_disable}
    ):
        policy = ExperimentLogPolicy(workspace.root)

    session = OptimizationSession(
        workload_id=cell["workload"],
        workspace=workspace,
        sink=sink,
        max_candidates=int(cell["budget"]),
        policy=policy,
    )

    extra: dict[str, Any] = {
        "max_candidates": int(cell["budget"]),
        "seed": int(cell["seed"]),
        "llm_min_interval_s": float(cell["llm_min_interval_s"]),
        "llm_max_concurrent": int(cell["llm_max_concurrent"]),
    }
    if cascade_disable:
        extra["cascade"] = {"disable": cascade_disable}
    key_env = {
        "mistral": "MISTRAL_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    provider = str(cell["model_id"]).split(":", 1)[0]
    env_name = key_env.get(provider)
    if env_name and os.environ.get(env_name):
        extra[f"{provider}_api_key"] = os.environ[env_name]

    request = HarnessRunRequest(
        toolset=session.toolset,
        system_instructions="",
        user_prompt="Optimize the registered workload.",
        model_id=cell["model_id"],
        max_turns=int(cell["max_turns"]),
        extra=extra,
    )
    harness = harness_registry.get(cell["harness"])
    harness_result = asyncio.run(
        run_session(
            session=session,
            harness=harness,
            request=request,
            max_continuations=int(cell["max_continuations"]),
        )
    )
    session.finalize()

    candidates = _candidate_rows(session)
    best = _best_validated(candidates)
    baseline_lease = _gpu_lease_of(session.baseline_compile.metadata or {})
    gpu_wait_total = (baseline_lease.get("lease_wait_s") or 0.0) + sum(
        r.get("gpu_lease_wait_s") or 0.0 for r in candidates
    )
    usage = harness_result.metadata.get("usage") or {}
    tokens_in = int(usage.get("request_tokens", 0) or 0)
    tokens_out = int(usage.get("response_tokens", 0) or 0)
    ran = [c for c in candidates if c.get("compile_ok") is not None]
    timed_e = sum(1 for c in ran if c.get("median_ms") is not None)

    row: dict[str, Any] = {
        "key": cell_key(cell),
        "harness": cell["harness"],
        "workload": cell["workload"],
        "budget": int(cell["budget"]),
        "seed": int(cell["seed"]),
        "model_id": cell["model_id"],
        "gpu": cell.get("gpu"),
        "run_id": session.run_id,
        "baseline_median_ms": session.baseline_time.median_ms,
        "best_candidate_id": best["candidate_id"] if best else None,
        "best_median_ms": best["median_ms"] if best else None,
        "best_speedup": best["speedup_vs_baseline"] if best else None,
        "successful_count": session.budget_state["successful_count"],
        "failed_attempts": session.budget_state["failed_attempts"],
        # Budget-ledger counters: E = timing invocations (gate-passing,
        # timed); V = validation-only executions (ran, never timed).
        "timed_evals_E": timed_e,
        "validation_only_V": len(ran) - timed_e,
        "gates": _gate_summary(candidates),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd_est": round(estimate_cost_usd(tokens_in, tokens_out), 6),
        "llm_calls": harness_result.metadata.get("llm_calls"),
        "wallclock_s": round(time.perf_counter() - started, 1),
        # Total seconds sandbox runs (baseline + candidates) queued for a
        # GPU lease; 0.0 in pinned mode.
        "gpu_lease_wait_s_total": round(gpu_wait_total, 3),
        "completion_reason": harness_result.metadata.get("completion_reason"),
        "iterations": harness_result.metadata.get("iterations"),
        "policy": getattr(policy, "name", "null") if policy else "null",
        "candidates": candidates,
        "timestamp": time.time(),
    }
    if cell.get("cascade_disable"):
        row["cascade_disable"] = cell["cascade_disable"]
    for extra_key in ("judge", "cascade_config", "arms", "incumbent_speedup"):
        if extra_key in harness_result.metadata:
            row[extra_key] = harness_result.metadata[extra_key]
    return row


def _execute_cell(cell: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    """`run_cell`, or — under ``--dry-run`` — a 0.1 s stub row, so driver
    plumbing (queue, append safety, resumability) is testable without GPUs,
    torch, or an LLM key."""

    if not dry_run:
        return run_cell(cell)
    time.sleep(0.1)
    return {
        "key": cell_key(cell),
        "harness": cell["harness"],
        "workload": cell["workload"],
        "budget": int(cell["budget"]),
        "seed": int(cell["seed"]),
        "model_id": cell["model_id"],
        "gpu": cell.get("gpu"),
        "best_speedup": None,
        "wallclock_s": 0.1,
        "completion_reason": "dry_run",
        "timestamp": time.time(),
    }


def _append_row(out_path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL row, serialized ACROSS PROCESSES with flock on a
    sidecar lock file: single-line write + flush + fsync under the lock, so
    concurrent workers can never tear or interleave lines. (An mp.Lock only
    covers workers of one driver invocation; the flock sidecar also covers
    any other process appending to the same file.)"""

    line = json.dumps(row, default=str) + "\n"
    lock_path = out_path.with_name(out_path.name + ".lock")
    with open(lock_path, "a", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _worker(
    label: str,
    env: dict[str, str],
    gpu_field: str,
    queue: Any,
    out_path: str,
    dry_run: bool,
) -> None:
    """Drain the cell queue one session at a time.

    `env` is applied BEFORE importing torch. Pinned mode passes
    ``CUDA_VISIBLE_DEVICES=<idx>`` (one worker per device, historical
    behavior); pool mode passes ``COMPILAGENT_GPU_POOL`` and pops any
    inherited ``CUDA_VISIBLE_DEVICES`` — the worker itself never pins a
    device (pool indices are physical), it only runs the LLM-bound episode
    while the backend leases a device around each sandbox subprocess.
    """

    os.environ.update(env)
    if "COMPILAGENT_GPU_POOL" in env:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    out = Path(out_path)
    while True:
        try:
            # Short timeout (not get_nowait) so a worker that starts before
            # the queue feeder has flushed every cell doesn't exit early.
            cell = queue.get(timeout=1.0)
        except Empty:
            return
        cell = dict(cell)
        cell["gpu"] = gpu_field
        key = cell_key(cell)
        print(f"[{label}] → {key}", flush=True)
        try:
            row = _execute_cell(cell, dry_run)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            row = {
                "key": key,
                "harness": cell["harness"],
                "workload": cell["workload"],
                "budget": int(cell["budget"]),
                "seed": int(cell["seed"]),
                "model_id": cell["model_id"],
                "gpu": gpu_field,
                "error": f"{type(exc).__name__}: {exc}",
                "timestamp": time.time(),
            }
        _append_row(out, row)
        print(
            f"[{label}] ✓ {key} speedup={row.get('best_speedup')} "
            f"tokens={row.get('tokens_in')}/{row.get('tokens_out')} "
            f"E/V={row.get('timed_evals_E')}/{row.get('validation_only_V')} "
            f"wall={row.get('wallclock_s')}s",
            flush=True,
        )


def _completed_keys(out_path: Path) -> set[str]:
    """Keys of cells already present with a NON-ERROR row (errored cells
    are retried on the next invocation)."""

    done: set[str] = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("key") and "error" not in row:
            done.add(str(row["key"]))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harnesses", required=True,
                        help="comma-separated harness ids")
    parser.add_argument("--workloads", default="softmax_4096",
                        help="comma-separated workload ids")
    parser.add_argument("--budgets", default="4",
                        help="comma-separated max_candidates budgets")
    parser.add_argument("--seeds", default="13",
                        help="comma-separated seeds")
    parser.add_argument("--model", default="mistral:mistral-large-latest")
    parser.add_argument(
        "--gpus", default="",
        help='pinned mode: comma-separated CUDA device indices, e.g. "1,2,3", '
             "one worker per device (GPU 0 is reserved on this machine — "
             "never include it); mutually exclusive with --gpu-pool",
    )
    parser.add_argument(
        "--gpu-pool", default="",
        help='pool mode: comma-separated CUDA device indices, e.g. "1,2,3", '
             "shared by --episode-workers unpinned workers; each sandbox "
             "subprocess leases one device (flock) for its lifetime; "
             "mutually exclusive with --gpus",
    )
    parser.add_argument(
        "--episode-workers", type=int, default=0,
        help="pool mode: number of concurrent episode worker processes "
             "(default 2 × pool size; episodes are LLM-latency-dominated, "
             "so N >> #GPUs is the point)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="replace episode execution with a 0.1 s stub row — exercises "
             "queue/append/resume plumbing without GPUs, torch, or LLM keys",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-continuations", type=int, default=2)
    parser.add_argument(
        "--cascade-disable", default="",
        help="comma-separated cascade ingredients/bundles to disable "
             "(c1..c10 or proposal/feedback/budget/memory)",
    )
    parser.add_argument("--llm-min-interval", type=float, default=1.2)
    parser.add_argument("--llm-max-concurrent", type=int, default=2)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "scripts" / "results" / "pilot.jsonl"),
    )
    args = parser.parse_args(argv)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    pool = [g.strip() for g in args.gpu_pool.split(",") if g.strip()]
    if gpus and pool:
        print("--gpus (pinned) and --gpu-pool (leased) are mutually "
              "exclusive; pick one.", file=sys.stderr)
        return 2
    if args.episode_workers and not pool:
        print("--episode-workers only applies to pool mode; pass --gpu-pool.",
              file=sys.stderr)
        return 2
    if not gpus and not pool:
        print(
            "Refusing to run without an explicit --gpus or --gpu-pool list "
            "(GPU 0 is reserved). Example: --gpus 1  or  --gpu-pool 1,2,3",
            file=sys.stderr,
        )
        return 2
    if "0" in gpus or "0" in pool:
        print("GPU 0 is reserved on this machine; remove it from "
              "--gpus/--gpu-pool.", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(out_path)

    cells: list[dict[str, Any]] = []
    for harness in [h.strip() for h in args.harnesses.split(",") if h.strip()]:
        for workload in [w.strip() for w in args.workloads.split(",") if w.strip()]:
            for budget in [int(b) for b in args.budgets.split(",") if b.strip()]:
                for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
                    cell = {
                        "harness": harness,
                        "workload": workload,
                        "budget": budget,
                        "seed": seed,
                        "model_id": args.model,
                        "max_turns": args.max_turns,
                        "max_continuations": args.max_continuations,
                        "cascade_disable": (
                            args.cascade_disable if harness == "cascade" else ""
                        ),
                        "llm_min_interval_s": args.llm_min_interval,
                        "llm_max_concurrent": args.llm_max_concurrent,
                    }
                    if cell_key(cell) in done:
                        print(f"skip (done): {cell_key(cell)}", flush=True)
                        continue
                    cells.append(cell)

    if not cells:
        print("Nothing to do — every cell is already in the results file.")
        return 0

    if pool:
        n_workers = args.episode_workers or 2 * len(pool)
        pool_csv = ",".join(pool)
        worker_specs = [
            (f"w{i}", {"COMPILAGENT_GPU_POOL": pool_csv}, f"pool:{pool_csv}")
            for i in range(n_workers)
        ]
        print(
            f"{len(cells)} cell(s) across {n_workers} episode worker(s) "
            f"leasing GPUs [{pool_csv}] → {out_path}",
            flush=True,
        )
    else:
        worker_specs = [
            (f"gpu {gpu}", {"CUDA_VISIBLE_DEVICES": str(gpu)}, str(gpu))
            for gpu in gpus
        ]
        print(f"{len(cells)} cell(s) across {len(gpus)} GPU(s) → {out_path}",
              flush=True)

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    for cell in cells:
        queue.put(cell)
    workers = [
        ctx.Process(
            target=_worker,
            args=(label, env, gpu_field, queue, str(out_path), args.dry_run),
            daemon=False,
        )
        for (label, env, gpu_field) in worker_specs
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    failed = [w for w in workers if w.exitcode not in (0, None)]
    if failed:
        print(f"{len(failed)} worker(s) exited non-zero.", file=sys.stderr)
        return 1
    print(f"Results → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
