"""Pilot grid driver: (harness × workload × budget × seed) cells (E-ticket DRIVER).

Extends scripts/run_archetype_smoke.py to a resumable multi-GPU grid:

  - one session at a time per GPU: a simple work queue feeds one worker
    process per device given via ``--gpus "1,2,3"``; each worker pins
    itself with CUDA_VISIBLE_DEVICES *before* importing torch.
  - one suite-row JSON per cell, appended to a JSONL results file
    (speedup, per-candidate gate verdicts, tokens in/out, $-estimate at
    mistral-large pricing, wallclock, llm_calls, E/V counts, judge
    metadata for cascade cells).
  - resumable: cells whose key already has a non-error row in the JSONL
    are skipped; errored cells are retried on the next invocation.
  - Mistral rate limits: the process-global throttle in
    archetype_harnesses._llm (1.2 s min-interval between request starts,
    ≤2 in-flight) applies inside every worker process; both knobs are
    forwarded from --llm-min-interval / --llm-max-concurrent.

Example (GPU 0 is reserved on this machine — never include it):

    MISTRAL_API_KEY=... python -m scripts.run_pilot \
        --harnesses archetype_evo,archetype_band,cascade \
        --workloads softmax_4096 --budgets 4 --seeds 13 --gpus 1 \
        --out results/pilot.jsonl
"""

from __future__ import annotations

import argparse
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
        rows.append(
            {
                "candidate_id": cid,
                "description": c.get("description", ""),
                "compile_ok": getattr(compile_outcome, "ok", None),
                "gates": (getattr(compile_outcome, "metadata", {}) or {}).get(
                    "gates"
                ),
                "median_ms": getattr(timing, "median_ms", None),
                "speedup_vs_baseline": c.get("speedup"),
                "correctness_ok": getattr(correctness, "ok", None),
                "diagnostics": getattr(compile_outcome, "diagnostics", None),
            }
        )
    return rows


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


def _worker(gpu: str, queue: Any, out_path: str, lock: Any) -> None:
    """One worker per GPU: pin the device BEFORE importing torch, then
    drain the cell queue one session at a time."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = Path(out_path)
    while True:
        try:
            cell = queue.get_nowait()
        except Empty:
            return
        cell = dict(cell)
        cell["gpu"] = str(gpu)
        label = cell_key(cell)
        print(f"[gpu {gpu}] → {label}", flush=True)
        try:
            row = run_cell(cell)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            row = {
                "key": label,
                "harness": cell["harness"],
                "workload": cell["workload"],
                "budget": int(cell["budget"]),
                "seed": int(cell["seed"]),
                "model_id": cell["model_id"],
                "gpu": str(gpu),
                "error": f"{type(exc).__name__}: {exc}",
                "timestamp": time.time(),
            }
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(
            f"[gpu {gpu}] ✓ {label} speedup={row.get('best_speedup')} "
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
        help='comma-separated CUDA device indices, e.g. "1,2,3" '
             "(GPU 0 is reserved on this machine — never include it)",
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
    if not gpus:
        print(
            "Refusing to run without an explicit --gpus list "
            "(GPU 0 is reserved). Example: --gpus 1",
            file=sys.stderr,
        )
        return 2
    if "0" in gpus:
        print("GPU 0 is reserved on this machine; remove it from --gpus.",
              file=sys.stderr)
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
    print(f"{len(cells)} cell(s) across {len(gpus)} GPU(s) → {out_path}",
          flush=True)

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    for cell in cells:
        queue.put(cell)
    lock = ctx.Lock()
    workers = [
        ctx.Process(
            target=_worker, args=(gpu, queue, str(out_path), lock), daemon=False
        )
        for gpu in gpus
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
