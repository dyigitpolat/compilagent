"""Drive archetype-harness OptimizationSessions on triton_source workloads.

The T0/E-ticket smoke driver: one row per (workload, harness) cell, in a
suite-row-style JSON that also carries the per-candidate gate verdicts and
the run-level token accounting the archetype harnesses report.

Run (GPU 0 is reserved — always pin explicitly):

    CUDA_VISIBLE_DEVICES=1 MISTRAL_API_KEY=... python -m scripts.run_archetype_smoke \
        --workload softmax_4096 --harnesses archetype_bon,archetype_sr \
        --max-candidates 2 --out results/e_ticket_smoke.json
"""

from __future__ import annotations

import argparse
import asyncio
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
    import compilagent.integrations.triton_source  # noqa: F401
    import compilagent.integrations.archetype_harnesses  # noqa: F401
    import compilagent.integrations.pydantic_ai  # noqa: F401  (model resolution)


def _harness_extra(model_id: str, max_candidates: int) -> dict[str, Any]:
    extra: dict[str, Any] = {"max_candidates": max_candidates}
    key_env = {
        "mistral": "MISTRAL_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    provider = model_id.split(":", 1)[0]
    env_name = key_env.get(provider)
    if env_name and os.environ.get(env_name):
        extra[f"{provider}_api_key"] = os.environ[env_name]
    return extra


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


def run_one(
    *,
    workload_id: str,
    harness_id: str,
    model_id: str,
    max_candidates: int,
    max_turns: int,
    max_continuations: int,
) -> dict[str, Any]:
    from compilagent.harness.base import HarnessRunRequest
    from compilagent.harness.registry import harness_registry
    from compilagent.session.session import OptimizationSession, run_session
    from compilagent.storage.trace_store import TraceStore
    from compilagent.storage.workspace import OptimizationWorkspace

    started = time.perf_counter()
    workspace = OptimizationWorkspace(session_cwd=Path.cwd()).ensure()
    sink = TraceStore(workspace.root).ensure()
    session = OptimizationSession(
        workload_id=workload_id,
        workspace=workspace,
        sink=sink,
        max_candidates=max_candidates,
    )
    request = HarnessRunRequest(
        toolset=session.toolset,
        system_instructions="",
        user_prompt="Optimize the registered workload.",
        model_id=model_id,
        max_turns=max_turns,
        extra=_harness_extra(model_id, max_candidates),
    )
    harness = harness_registry.get(harness_id)
    harness_result = asyncio.run(
        run_session(
            session=session,
            harness=harness,
            request=request,
            max_continuations=max_continuations,
        )
    )
    session.finalize()

    candidates = _candidate_rows(session)
    best = _best_validated(candidates)
    elapsed_s = time.perf_counter() - started
    return {
        "workload": workload_id,
        "harness": harness_id,
        "model_id": model_id,
        "max_candidates": max_candidates,
        "run_id": session.run_id,
        "baseline_median_ms": session.baseline_time.median_ms,
        "best_candidate_id": best["candidate_id"] if best else None,
        "best_median_ms": best["median_ms"] if best else None,
        "best_speedup": best["speedup_vs_baseline"] if best else None,
        "correctness_ok": best["correctness_ok"] if best else None,
        "successful_count": session.budget_state["successful_count"],
        "failed_attempts": session.budget_state["failed_attempts"],
        "candidates": candidates,
        "tokens": harness_result.metadata.get("usage"),
        "llm_calls": harness_result.metadata.get("llm_calls"),
        "completion_reason": harness_result.metadata.get("completion_reason"),
        "iterations": harness_result.metadata.get("iterations"),
        "elapsed_s": round(elapsed_s, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="softmax_4096")
    parser.add_argument(
        "--harnesses", default="archetype_bon,archetype_sr",
        help="comma-separated harness ids, run in order",
    )
    parser.add_argument("--model", default="mistral:mistral-large-latest")
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-continuations", type=int, default=2)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "scripts" / "results" / "e_ticket_smoke.json"),
    )
    args = parser.parse_args(argv)

    # GPU 0 is reserved for another experiment on this machine: refuse to
    # run unless the caller pinned a device explicitly.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(
            "Refusing to run without an explicit CUDA_VISIBLE_DEVICES "
            "(GPU 0 is reserved). Example: CUDA_VISIBLE_DEVICES=1 ...",
            file=sys.stderr,
        )
        return 2

    _ensure_imports()

    rows: list[dict[str, Any]] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _payload() -> dict[str, Any]:
        return {
            "model_id": args.model,
            "workload": args.workload,
            "max_candidates": args.max_candidates,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "rows": rows,
        }

    for harness_id in [h.strip() for h in args.harnesses.split(",") if h.strip()]:
        print(f"→ {harness_id} on {args.workload} "
              f"(max_candidates={args.max_candidates})", flush=True)
        try:
            row = run_one(
                workload_id=args.workload,
                harness_id=harness_id,
                model_id=args.model,
                max_candidates=args.max_candidates,
                max_turns=args.max_turns,
                max_continuations=args.max_continuations,
            )
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            row = {
                "workload": args.workload,
                "harness": harness_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        out_path.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")
        print(
            f"  speedup={row.get('best_speedup')} "
            f"tokens={row.get('tokens')} elapsed={row.get('elapsed_s')}s",
            flush=True,
        )

    print(f"Results → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
