"""Per-cell driver used by the study script.

Lives as a proper module under `compilagent_triton/` so subprocess workers
can `from compilagent_triton.study import run_cell` cleanly. The script in
`scripts/experiments/run_study.py` is just a CLI wrapper around `run_cell`.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch  # noqa: F401  — must precede triton/inductor imports

from .backends import backend_registry
from .settings import CompilagentSettings
from .trace_store import TraceStore
from .workload_runner import _run_claude_sdk, _run_pydantic_ai
from .workloads.registry import workload_registry


@dataclass(slots=True)
class CellResult:
    harness: str
    workload_id: str
    backend_id: str
    max_candidates: int
    seed: int
    model_name: str
    baseline_median_ms: float | None
    best_speedup: float | None
    best_candidate_id: str | None
    best_median_ms: float | None
    best_correctness_ok: bool | None
    best_max_abs_diff: float | None
    successful_count: int
    failed_attempts: int
    elapsed_ms: float
    final_text: str | None
    candidates: list[dict]
    correctness_recheck_ok: bool | None
    correctness_recheck_max_abs_diff: float | None
    error: str | None
    timestamp: str


def _recheck_correctness(
    *, workload_id: str, workspace_root: Path, run_id: str,
) -> tuple[bool | None, float | None]:
    """Re-run the best candidate's compile + correctness check.

    Disabled by default in the study driver — torch + triton CUDA cleanup
    state can segfault when we re-compile in the same process. Re-enable by
    setting `COMPILAGENT_STUDY_RECHECK=1` in the env.
    """

    trace_store = TraceStore(workspace_root)
    events = [
        e for e in trace_store.read_events()
        if (e.payload or {}).get("run_id") == run_id
    ]
    proposals: dict[str, dict] = {}
    benchmarks: dict[str, dict] = {}
    for ev in events:
        kind = ev.kind
        payload = ev.payload or {}
        if kind == "candidate.proposed":
            for c in payload.get("candidates", []) or []:
                proposals[c["id"]] = c
        elif kind == "benchmark.completed":
            cid = payload.get("candidate_id")
            if cid:
                benchmarks[cid] = payload
    best_id = None
    best_sp = 1.0
    for cid, b in benchmarks.items():
        sp = b.get("speedup_vs_baseline")
        if isinstance(sp, (int, float)) and sp > best_sp:
            best_sp = sp
            best_id = cid
    if best_id is None or best_id not in proposals:
        return None, None

    from .backends import Plan
    from .backends.base import Intervention, Target, ToleranceConfig
    interventions: list[Intervention] = []
    for kind_str, group in (proposals[best_id].get("changes") or {}).items():
        if not isinstance(group, dict):
            continue
        for selector, payload in group.items():
            interventions.append(Intervention(
                target=Target(kind=str(kind_str), selector=str(selector)),
                payload=payload, rationale="recheck",
            ))
    if not interventions:
        return None, None

    spec = workload_registry.get_spec(workload_id)
    backend = backend_registry.get(spec.backend_id)
    plan = Plan(interventions=tuple(interventions))
    try:
        torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        recheck_root = workspace_root / "recheck" / run_id
        recheck_root.mkdir(parents=True, exist_ok=True)
        b_dir = recheck_root / "baseline"; b_dir.mkdir(parents=True, exist_ok=True)
        c_dir = recheck_root / "candidate"; c_dir.mkdir(parents=True, exist_ok=True)
        b_compile = backend.compile(spec, Plan(), artifact_dir=b_dir)
        c_compile = backend.compile(spec, plan, artifact_dir=c_dir)
        if not (getattr(b_compile, "ok", False) and getattr(c_compile, "ok", False)):
            return None, None
        tol = ToleranceConfig(
            atol=spec.tolerance.atol, rtol=spec.tolerance.rtol,
            notes="independent recheck",
        )
        result = backend.validate_correctness(spec, b_compile, c_compile, tol)
        return result.ok, result.max_abs_diff
    except Exception:  # noqa: BLE001
        return None, None


def run_cell(
    *,
    harness: str,
    workload_id: str,
    max_candidates: int,
    seed: int,
    model_name: str,
    sdk_model_name: str,
    out_root: Path,
    early_write_path: Path | None = None,
) -> CellResult:
    """Drive one cell of the study grid; subprocess-safe.

    If `early_write_path` is set we serialise the (so-far-known) result there
    as soon as the agent finishes — before any `asyncio.run` cleanup or torch
    CUDA finalizers run. This protects the result against the segfaults that
    routinely hit Python's interpreter shutdown on this host (sm_120).
    """

    timestamp = datetime.now(UTC).isoformat()
    spec = workload_registry.get_spec(workload_id)

    cell_dir = (
        out_root / "cells" /
        f"{harness}__{workload_id}__t{max_candidates}__s{seed}"
    )
    cell_dir.mkdir(parents=True, exist_ok=True)
    trace_store = TraceStore(cell_dir).ensure()
    run_id = f"study-{harness}-{workload_id}-t{max_candidates}-s{seed}"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    settings = CompilagentSettings.from_env(project_root=Path.cwd())
    effective_model = (
        sdk_model_name
        if harness == "claude_agent_sdk" and not model_name.startswith("anthropic:")
        else model_name
    )
    settings = settings.model_copy(update={"model_name": effective_model})

    runner = _run_claude_sdk if harness == "claude_agent_sdk" else _run_pydantic_ai

    # Use a hand-managed event loop instead of `asyncio.run`. `asyncio.run`
    # closes the loop on exit, which on this host (torch/triton + sm_120)
    # routinely hangs or segfaults during torch.compile finalizers. With a
    # bare `run_until_complete` the loop is left open and we `os._exit` once
    # the result is written — the OS cleans up.
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        summary = loop.run_until_complete(runner(
            workload_id=workload_id, run_id=run_id,
            workspace_root=cell_dir, trace_store=trace_store,
            settings=settings, max_candidates=max_candidates,
        ))
        error = None
    except Exception:  # noqa: BLE001
        summary = {}
        error = traceback.format_exc()

    candidates_list: list[dict] = []
    for ev in trace_store.read_events():
        kind = ev.kind
        payload = ev.payload or {}
        if payload.get("run_id") != run_id:
            continue
        if kind == "candidate.proposed":
            for c in payload.get("candidates", []) or []:
                if c.get("kind") == "search_space_summary":
                    continue
                candidates_list.append({
                    "id": c.get("id"),
                    "description": c.get("description"),
                    "changes": c.get("changes", {}),
                })
        elif kind == "benchmark.completed":
            cid = payload.get("candidate_id")
            if not cid:
                continue
            for c in candidates_list:
                if c.get("id") == cid:
                    c["median_ms"] = payload.get("median_ms")
                    c["speedup_vs_baseline"] = payload.get("speedup_vs_baseline")
                    break

    if (
        not error and summary.get("best_speedup")
        and os.environ.get("COMPILAGENT_STUDY_RECHECK") == "1"
    ):
        recheck_ok, recheck_diff = _recheck_correctness(
            workload_id=workload_id, workspace_root=cell_dir, run_id=run_id,
        )
    else:
        recheck_ok = summary.get("best_correctness_ok")
        recheck_diff = summary.get("best_max_abs_diff")

    # Snapshot every scalar we need into local variables BEFORE building the
    # CellResult — so we don't keep `summary` (which transitively holds the
    # WorkloadSession and every torch.compile'd callable) alive longer than
    # necessary. The CellResult itself only contains scalars + the candidates
    # list (which is plain dicts of strings/numbers).
    s_baseline_median_ms = summary.get("baseline_median_ms")
    s_best_speedup = summary.get("best_speedup")
    s_best_candidate_id = summary.get("best_candidate_id")
    s_best_median_ms = summary.get("best_median_ms")
    s_best_correctness_ok = summary.get("best_correctness_ok")
    s_best_max_abs_diff = summary.get("best_max_abs_diff")
    s_successful_count = int(summary.get("successful_count") or 0)
    s_failed_attempts = int(summary.get("failed_attempts") or 0)
    s_elapsed_ms = float(summary.get("elapsed_ms") or 0.0)
    s_final_text = summary.get("final_text")

    result = CellResult(
        harness=harness,
        workload_id=workload_id,
        backend_id=spec.backend_id,
        max_candidates=max_candidates,
        seed=seed,
        model_name=effective_model,
        baseline_median_ms=s_baseline_median_ms,
        best_speedup=s_best_speedup,
        best_candidate_id=s_best_candidate_id,
        best_median_ms=s_best_median_ms,
        best_correctness_ok=s_best_correctness_ok,
        best_max_abs_diff=s_best_max_abs_diff,
        successful_count=s_successful_count,
        failed_attempts=s_failed_attempts,
        elapsed_ms=s_elapsed_ms,
        final_text=s_final_text,
        candidates=candidates_list,
        correctness_recheck_ok=recheck_ok,
        correctness_recheck_max_abs_diff=recheck_diff,
        error=error,
        timestamp=timestamp,
    )
    # Persist the result NOW, before any GC pass touches the still-live
    # WorkloadSession references inside `summary`. Once this file is on disk
    # the parent has its data even if the rest of this process segfaults
    # during cleanup.
    if early_write_path is not None:
        from dataclasses import asdict as _asdict
        try:
            early_write_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = early_write_path.with_suffix(early_write_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(_asdict(result), f, default=str)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(early_write_path)
        except Exception:  # noqa: BLE001
            pass
    return result
