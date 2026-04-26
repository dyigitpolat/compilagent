"""Backend-agnostic agent runner driven by a `WorkloadSpec`.

The optimization session is encapsulated in `WorkloadSession`, which owns the
baseline compile + analysis + search space and exposes the **tool surface** the
agent works against:

  - `inspect_workload`, `inspect_search_space`,
  - `propose_intervention`, `propose_candidate`, `propose_candidates`,
  - `run_candidate`, `run_candidates`,
  - `synthesize_findings`, `compare_runs`.

The same session can be driven by either harness:

  - **pydantic-ai** (`run_workload_optimization` → `_run_pydantic_ai`) — methods
    are registered as `@agent.tool_plain` so docstrings flow through directly.
  - **Claude Agent SDK** (`build_workload_mcp_server`) — methods are wrapped as
    MCP tools via `claude_agent_sdk.tool` with the same docstrings as
    descriptions.

Streams thinking, text, and tool events to the observer's TraceStore so the UI
sees them live, regardless of which harness drove the session.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
)

from .backends import Plan, backend_registry
from .backends.base import Intervention, Target, ToleranceConfig
from .core.workload import WorkloadSpec
from .llm import model_for_settings, model_settings_for_settings
from .settings import CompilagentSettings
from .trace_store import TraceStore
from .workloads.registry import import_workload_packages, workload_registry


# Names of `WorkloadSession` methods that are exposed as agent tools. Used by
# both the pydantic-ai loop (registers each on the Agent) and the SDK MCP
# server (registers each as a tool). Keeping a single list keeps the two
# harnesses on the same generic tool surface.
TOOL_METHOD_NAMES: tuple[str, ...] = (
    "inspect_workload",
    "inspect_search_space",
    "propose_intervention",
    "propose_candidate",
    "propose_candidates",
    "run_candidate",
    "run_candidates",
    "synthesize_findings",
    "compare_runs",
)


def _retry_on_value_error(fn):
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    return wrapped


def _loads_lenient(text: str) -> Any:
    """Permissive JSON parser used by the propose_* tools.

    LLMs (especially Mistral on long batched arrays) routinely emit JSON with
    minor mistakes the strict parser rejects: trailing commas, smart quotes,
    backslash-newlines inside strings, lone control characters. We try strict
    first; on failure we sanitise and retry, then fall back to ast.literal_eval
    as a last resort. The strict-mode error message bubbles up if everything
    fails so the agent's retry sees the precise complaint.
    """

    text = (text or "").strip()
    if not text:
        raise ValueError("empty JSON input")
    try:
        return json.loads(text)
    except json.JSONDecodeError as strict_err:
        cleaned = (
            text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"')
        )
        # Strip trailing commas before } or ]: ", }" or ", ]".
        import re as _re
        cleaned = _re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        try:
            import ast as _ast
            return _ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            pass
        raise ValueError(
            f"could not parse JSON (strict and lenient parsers both failed): {strict_err}"
        ) from strict_err


class WorkloadSession:
    """Per-run state + tool surface for a backend-agnostic workload optimization.

    Construction performs the deterministic baseline phase eagerly: compile
    baseline, run analysis, build derived search space, time the baseline.
    After construction the session is read-only from the agent's POV; the only
    mutating tools are `propose_*` and `run_*`, which append to the candidate
    dict.
    """

    def __init__(
        self,
        *,
        workload_id: str,
        run_id: str,
        workspace_root: Path,
        trace_store: TraceStore,
        settings: CompilagentSettings,
        max_candidates: int,
    ) -> None:
        # Backend + workload registries are dynamic — see backends/__init__.py
        # and workloads/registry.py. New backend / workload added on disk shows
        # up here without editing this code.
        from .backends import import_backend_packages
        import_backend_packages()
        import_workload_packages()

        self.workload_id = workload_id
        self.run_id = run_id
        self.trace_store = trace_store
        self.settings = settings
        self.max_candidates = max_candidates
        self.spec = workload_registry.get_spec(workload_id)
        self.backend = backend_registry.get(self.spec.backend_id)

        self.artifact_root = (
            workspace_root / "workloads" / workload_id / "runs" / run_id
        ).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        baseline_dir = self.artifact_root / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        # Per-run part counter for monotonic UI card keys. TraceStore is a
        # `slots=True` dataclass so we keep this as a plain dict.
        self.part_counters: dict[str, int] = {"thinking": 0, "text": 0}
        self.budget_state: dict[str, int] = {
            "successful_count": 0,
            "failed_attempts": 0,
            "max_failed_attempts": max(8, max_candidates * 3),
        }
        self.candidates: dict[str, dict[str, Any]] = {}

        cap = self.backend.device_capability()
        self.arch = cap.arch
        trace_store.emit(
            "agent.run_started",
            payload={
                "run_id": run_id,
                "workload_id": workload_id,
                "backend_id": self.backend.id,
                "model": settings.model_name,
                "reasoning_effort": settings.reasoning_effort,
                "max_candidates": max_candidates,
            },
        )

        # ---- baseline compile + analysis ----
        self.baseline_compile = self.backend.compile(
            self.spec, Plan(),
            artifact_dir=baseline_dir,
            pass_callback=self._make_pass_callback("baseline"),
        )
        trace_store.emit(
            "compile.completed",
            payload={
                "run_id": run_id,
                "compile_id": "baseline",
                "ok": getattr(self.baseline_compile, "ok", False),
                "elapsed_ms": getattr(self.baseline_compile, "elapsed_ms", None),
            },
        )
        self._emit_artifacts(self.baseline_compile, candidate_id="baseline")

        baseline_artifacts = list(getattr(self.baseline_compile, "artifacts", ()) or ())
        for attr in ("fx_graph_path", "output_code_path", "schedule_log_path"):
            p = getattr(self.baseline_compile, attr, None)
            if p:
                baseline_artifacts.append(p)
        self.analysis = self.backend.analyze(
            self.spec, baseline_artifacts=baseline_artifacts,
        )
        self.baseline_time = self.backend.time_workload(
            self.spec, Plan(),
            warmup=self.spec.budget.warmup,
            repetitions=self.spec.budget.repetitions,
            max_seconds=self.spec.budget.max_seconds,
        )
        trace_store.emit(
            "benchmark.completed",
            payload={
                "run_id": run_id,
                "candidate_id": "baseline",
                "median_ms": self.baseline_time.median_ms,
                "p20_ms": self.baseline_time.p20_ms,
                "p80_ms": self.baseline_time.p80_ms,
                "best": {
                    "candidate_id": "baseline",
                    "median_ms": self.baseline_time.median_ms,
                    "speedup_vs_baseline": 1.0,
                },
                "results": [{
                    "candidate_id": "baseline",
                    "median_ms": self.baseline_time.median_ms,
                }],
            },
        )
        self.search_space = self.backend.derive_search_space(self.spec, self.analysis)
        trace_store.emit(
            "candidate.proposed",
            payload={
                "run_id": run_id,
                "kind": "search_space_summary",
                "lever_count": len(self.search_space.levers),
                "candidates": [],
            },
        )

    # ---- artifact + per-pass plumbing -------------------------------------

    def _emit_artifacts(self, compile_outcome: Any, *, candidate_id: str) -> None:
        """Emit `artifact.created` for every artifact a backend produced."""

        seen: set[str] = set()
        for attr in ("output_code_path", "schedule_log_path", "fx_graph_path"):
            p = getattr(compile_outcome, attr, None)
            if p and str(p) not in seen:
                seen.add(str(p))
                self.trace_store.emit(
                    "artifact.created",
                    payload={
                        "run_id": candidate_id,
                        "kernel_id": self.spec.id,
                        "stage": attr.replace("_path", ""),
                        "path": str(p),
                    },
                    artifact_paths=[str(p)],
                )
        for p in getattr(compile_outcome, "artifacts", ()) or ():
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            stage = Path(sp).suffix.lstrip(".").lower() or "artifact"
            self.trace_store.emit(
                "artifact.created",
                payload={
                    "run_id": candidate_id,
                    "kernel_id": self.spec.id,
                    "stage": stage,
                    "path": sp,
                },
                artifact_paths=[sp],
            )

    def _make_pass_callback(self, candidate_id: str):
        """Build a backend-agnostic `compiler.pass` event emitter."""

        def _cb(event: Any) -> None:
            payload = {
                "run_id": self.run_id,
                "candidate_id": candidate_id,
                "stage": getattr(event, "stage", "") or "",
                "name": getattr(event, "name", "") or "",
                "duration_ms": getattr(event, "duration_ms", 0.0) or 0.0,
                "action": getattr(event, "action", "run") or "run",
                "ir_after_size": getattr(event, "ir_after_size", None),
                "error": getattr(event, "error", None),
            }
            self.trace_store.emit("compiler.pass", payload=payload)

        return _cb

    def _candidate_id(self) -> str:
        return f"cand-{uuid.uuid4().hex[:10]}"

    def _register_candidate(
        self,
        interventions: list[Intervention],
        *,
        rationale: str | None,
        description: str | None,
    ) -> str:
        cid = self._candidate_id()
        plan = Plan(interventions=tuple(interventions))
        self.candidates[cid] = {
            "id": cid,
            "plan": plan,
            "rationale": rationale or "",
            "description": description or "",
        }
        changes: dict[str, Any] = {}
        for iv in interventions:
            bucket = changes.setdefault(iv.target.kind, {})
            bucket[iv.target.selector] = iv.payload
        readable = description or rationale or (
            ", ".join(
                f"{iv.target.kind}({iv.target.selector})={iv.payload!r}"
                for iv in interventions
            )
        )
        self.trace_store.emit(
            "candidate.proposed",
            payload={
                "run_id": self.run_id,
                "candidates": [{
                    "id": cid,
                    "kernel_id": self.spec.id,
                    "kind": "multi_intervention" if len(interventions) > 1 else "intervention",
                    "description": readable,
                    "expected_effect": rationale or "",
                    "changes": changes,
                    "status": "proposed",
                }],
            },
        )
        return json.dumps({
            "id": cid,
            "intervention_count": len(interventions),
            "interventions": [iv.serialize() for iv in interventions],
            "description": description or "",
        }, indent=2)

    # ---- agent-facing tools (return JSON strings; raise ValueError on bad input) ----

    def inspect_workload(self) -> str:
        """Return the workload spec + baseline timing + analysis summary."""

        return json.dumps({
            "workload": self.spec.serialize(),
            "backend_id": self.backend.id,
            "device": {
                "arch": self.backend.device_capability().arch,
                "name": self.backend.device_capability().name,
                "capability_int": self.backend.device_capability().capability_int,
            },
            "baseline_timing": {
                "median_ms": self.baseline_time.median_ms,
                "p20_ms": self.baseline_time.p20_ms,
                "p80_ms": self.baseline_time.p80_ms,
            },
            "analysis_summary": self.analysis.summary,
        }, indent=2, default=str)

    def inspect_search_space(self) -> str:
        """Return the **derived** lever catalog with evidence."""

        return json.dumps({
            "workload_id": self.spec.id,
            "backend_id": self.backend.id,
            "lever_count": len(self.search_space.levers),
            "levers": [lev.serialize() for lev in self.search_space.levers],
        }, indent=2, default=str)

    def propose_intervention(
        self,
        target_kind: str,
        target_selector: str,
        payload_json: str,
        rationale: str = "",
    ) -> str:
        """Propose a single-lever candidate (legacy single-knob path).

        Prefer `propose_candidate` (multi-intervention) for any non-trivial
        hypothesis — combining 2-4 levers (e.g. shape_padding + max_autotune +
        coordinate_descent_tuning) is usually how meaningful wins surface.
        """

        try:
            payload = _loads_lenient(payload_json) if payload_json.strip() else None
        except ValueError as exc:
            raise ValueError(f"payload_json is not valid JSON: {exc}") from exc
        intervention = Intervention(
            target=Target(kind=target_kind, selector=target_selector),
            payload=payload,
            rationale=rationale or "",
        )
        return self._register_candidate([intervention], rationale=rationale, description=None)

    def propose_candidate(
        self,
        interventions_json: str,
        description: str,
        expected_effect: str = "",
    ) -> str:
        """Propose a **multi-intervention** candidate — combine several levers.

        `interventions_json` is a JSON list. Each entry is
        `{"target_kind": "...", "target_selector": "...", "payload": <value>,
          "rationale": "<optional one-liner>"}`. All interventions in the list
        are applied **together** to a single compile + benchmark.

        Use this when your hypothesis spans multiple knobs / levers (e.g.
        "enabling shape_padding alongside max_autotune should expose more
        Tensor-Core-eligible matmuls"). One-knob interventions are still fine
        via `propose_intervention`, but most non-trivial ML wins come from
        combining levers.
        """

        try:
            entries = _loads_lenient(interventions_json)
        except ValueError as exc:
            raise ValueError(f"interventions_json is not valid JSON: {exc}") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError("interventions_json must be a non-empty JSON list")
        ivs: list[Intervention] = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"intervention #{i} must be a dict")
            kind = entry.get("target_kind")
            selector = entry.get("target_selector", "")
            payload = entry.get("payload")
            sub_rationale = entry.get("rationale", "")
            if not kind:
                raise ValueError(f"intervention #{i} is missing `target_kind`")
            ivs.append(Intervention(
                target=Target(kind=str(kind), selector=str(selector)),
                payload=payload,
                rationale=str(sub_rationale or ""),
            ))
        return self._register_candidate(ivs, rationale=expected_effect, description=description)

    def propose_candidates(self, plans_json: str) -> str:
        """Propose **multiple** multi-intervention candidates in one shot.

        `plans_json` is a JSON list. Each entry has the same shape as
        `propose_candidate`: `{"description": str, "expected_effect"?: str,
         "interventions": [{"target_kind": "...", "target_selector": "...",
                            "payload": <value>, "rationale"?: "..."}, ...]}`.

        Useful for setting up a small batch (e.g. 3 hypotheses) before running
        them. Returns the list of registered candidate ids.
        """

        try:
            entries = _loads_lenient(plans_json)
        except ValueError as exc:
            raise ValueError(f"plans_json is not valid JSON: {exc}") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError("plans_json must be a non-empty JSON list of plan dicts")
        out: list[dict[str, Any]] = []
        for i, plan_dict in enumerate(entries):
            if not isinstance(plan_dict, dict):
                raise ValueError(f"plan #{i} must be a dict")
            ivs_raw = plan_dict.get("interventions") or []
            if not isinstance(ivs_raw, list) or not ivs_raw:
                raise ValueError(f"plan #{i} is missing a non-empty `interventions` list")
            ivs: list[Intervention] = []
            for j, iv in enumerate(ivs_raw):
                if not isinstance(iv, dict):
                    raise ValueError(f"plan #{i} intervention #{j} must be a dict")
                kind = iv.get("target_kind")
                if not kind:
                    raise ValueError(f"plan #{i} intervention #{j} missing `target_kind`")
                ivs.append(Intervention(
                    target=Target(kind=str(kind), selector=str(iv.get("target_selector", ""))),
                    payload=iv.get("payload"),
                    rationale=str(iv.get("rationale") or ""),
                ))
            registered = json.loads(self._register_candidate(
                ivs,
                rationale=plan_dict.get("expected_effect", ""),
                description=plan_dict.get("description", ""),
            ))
            out.append({
                "id": registered["id"],
                "description": registered.get("description", ""),
                "intervention_count": registered.get("intervention_count", len(ivs)),
            })
        return json.dumps({"registered": len(out), "candidates": out}, indent=2)

    def run_candidate(self, candidate_id: str) -> str:
        """Compile + time a previously proposed candidate."""

        if candidate_id not in self.candidates:
            raise ValueError(
                f"unknown candidate `{candidate_id}`; "
                f"known: {list(self.candidates.keys())[-5:]}"
            )
        try:
            import torch  # type: ignore[import-not-found]
            torch._dynamo.reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        c = self.candidates[candidate_id]
        plan = c.get("plan")
        if plan is None and "intervention" in c:
            plan = Plan(interventions=(c["intervention"],))
        cdir = self.artifact_root / "candidates" / candidate_id
        cdir.mkdir(parents=True, exist_ok=True)
        compile_outcome = self.backend.compile(
            self.spec, plan,
            artifact_dir=cdir,
            pass_callback=self._make_pass_callback(candidate_id),
        )
        self._emit_artifacts(compile_outcome, candidate_id=candidate_id)
        ok = getattr(compile_outcome, "ok", False)
        timing = self.backend.time_workload(
            self.spec, plan,
            warmup=self.spec.budget.warmup,
            repetitions=self.spec.budget.repetitions,
            max_seconds=self.spec.budget.max_seconds,
        ) if ok else None

        correctness = None
        if ok and getattr(self.baseline_compile, "ok", False):
            try:
                tol = ToleranceConfig(
                    atol=self.spec.tolerance.atol,
                    rtol=self.spec.tolerance.rtol,
                    notes=self.spec.tolerance.notes,
                )
                correctness = self.backend.validate_correctness(
                    self.spec, self.baseline_compile, compile_outcome, tol,
                )
            except Exception as exc:  # noqa: BLE001
                correctness = None
                self.trace_store.emit(
                    "log.line",
                    payload={"level": "warn",
                             "message": f"correctness check failed: {exc!r}"},
                )

        speedup = (
            (self.baseline_time.median_ms / timing.median_ms)
            if timing and timing.median_ms and self.baseline_time.median_ms
            else None
        )
        c.update({
            "compile": compile_outcome,
            "timing": timing,
            "correctness": correctness,
            "speedup": speedup,
        })

        self.trace_store.emit(
            "benchmark.completed",
            payload={
                "run_id": self.run_id,
                "candidate_id": candidate_id,
                "median_ms": timing.median_ms if timing else None,
                "p20_ms": timing.p20_ms if timing else None,
                "p80_ms": timing.p80_ms if timing else None,
                "speedup_vs_baseline": speedup,
                "best": {
                    "candidate_id": candidate_id,
                    "median_ms": timing.median_ms if timing else None,
                    "speedup_vs_baseline": speedup,
                },
                "results": [{
                    "candidate_id": candidate_id,
                    "median_ms": timing.median_ms if timing else None,
                    "speedup_vs_baseline": speedup,
                }],
            },
        )
        compile_diagnostics = getattr(compile_outcome, "diagnostics", None)
        compile_warnings = list(getattr(compile_outcome, "warnings", []) or [])
        successful = bool(
            ok and timing and timing.median_ms is not None
            and (correctness is None or correctness.ok)
        )
        if successful:
            self.budget_state["successful_count"] += 1
        else:
            self.budget_state["failed_attempts"] += 1
        slots_remaining = max(0, self.max_candidates - self.budget_state["successful_count"])
        self.trace_store.emit(
            "run.progress",
            payload={
                "run_id": self.run_id,
                "successful_count": self.budget_state["successful_count"],
                "failed_attempts": self.budget_state["failed_attempts"],
                "max_candidates": self.max_candidates,
                "slots_remaining": slots_remaining,
            },
        )
        hint = None
        if not ok:
            joined_warnings = "; ".join(compile_warnings[:3])
            if "shared_mem" in (compile_diagnostics or "").lower() or any(
                "Hardware limit" in w for w in compile_warnings
            ):
                hint = (
                    "compile failed because the chosen knob combination produced "
                    "Triton-template configs that exceed shared-memory budget. "
                    "Try a different knob mix (e.g. drop coordinate_descent_tuning "
                    "or pair max_autotune with shape_padding=False), or pick a "
                    "different intervention class."
                )
            elif joined_warnings:
                hint = (
                    "compile produced runtime warnings: " + joined_warnings
                    + " — adjust knobs accordingly."
                )
            else:
                hint = (
                    "compile failed; see `compile_diagnostics`. Adjust the plan "
                    "and propose a new candidate. This attempt did NOT count "
                    "against your trial budget."
                )
        elif correctness is not None and not correctness.ok:
            hint = (
                "compile + run succeeded but the candidate output drifted "
                f"outside tolerance (max_abs_diff={correctness.max_abs_diff}). "
                "Pick interventions that don't change numerics, or relax the "
                "tolerance hypothesis. This attempt did NOT count against "
                "your trial budget."
            )
        return json.dumps({
            "candidate_id": candidate_id,
            "compile_ok": ok,
            "median_ms": timing.median_ms if timing else None,
            "speedup_vs_baseline": speedup,
            "correctness_ok": correctness.ok if correctness else None,
            "max_abs_diff": correctness.max_abs_diff if correctness else None,
            "compile_diagnostics": compile_diagnostics,
            "compile_warnings": compile_warnings[:6],
            "successful": successful,
            "successful_count": self.budget_state["successful_count"],
            "failed_attempts": self.budget_state["failed_attempts"],
            "slots_remaining": slots_remaining,
            "hint": hint,
        }, indent=2, default=str)

    def run_candidates(self, candidate_ids_json: str) -> str:
        """Run a batch of previously proposed candidates in sequence.

        `candidate_ids_json` is a JSON list of candidate ids. Each is compiled
        and timed; the result of every attempt is collected (whether it
        succeeded, failed, or hit OOM probes). Returns a per-candidate result
        list plus an aggregate summary so the agent can synthesise across
        feedback sources without juggling individual `run_candidate` calls.
        """

        try:
            ids = _loads_lenient(candidate_ids_json)
        except ValueError as exc:
            raise ValueError(f"candidate_ids_json is not valid JSON: {exc}") from exc
        if not isinstance(ids, list) or not ids:
            raise ValueError("candidate_ids_json must be a non-empty list")
        results: list[dict[str, Any]] = []
        for cid in ids:
            try:
                results.append(json.loads(self.run_candidate(str(cid))))
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001
                results.append({"candidate_id": str(cid), "error": repr(exc)})
        successes = [r for r in results if r.get("successful")]
        return json.dumps({
            "ran": len(results),
            "successful": len(successes),
            "failed": len(results) - len(successes),
            "best": (
                max(successes, key=lambda r: r.get("speedup_vs_baseline") or 0.0)
                if successes else None
            ),
            "results": results,
            "slots_remaining": max(
                0, self.max_candidates - self.budget_state["successful_count"]
            ),
        }, indent=2, default=str)

    def synthesize_findings(self) -> str:
        """Aggregate the current run's candidate results into a structured
        synthesis the agent can read at any point.

        Returns:
          - per-target_kind speedup distributions (avg, best, count) — which
            intervention class is correlating with wins.
          - per-(target_kind, target_selector) wins/losses.
          - a list of "co-occurring lever pairs" that appear in winning
            candidates (so the agent can spot e.g. `max_autotune` × `shape_padding`).
          - failure modes: which interventions appear in failed candidates.
        """

        successes = [c for c in self.candidates.values() if c.get("speedup")]
        failures = [
            c for c in self.candidates.values()
            if c.get("compile") and not getattr(c.get("compile"), "ok", True)
        ]
        per_kind: dict[str, dict[str, Any]] = {}
        per_target: dict[str, dict[str, Any]] = {}
        pair_wins: dict[tuple[str, str], int] = {}
        for c in successes:
            sp = c.get("speedup") or 0.0
            ivs = list(c.get("plan").interventions) if c.get("plan") else []
            keys = [(iv.target.kind, iv.target.selector) for iv in ivs]
            for kind, selector in keys:
                pk = per_kind.setdefault(kind, {"count": 0, "best": 0.0, "speedups": []})
                pk["count"] += 1
                pk["best"] = max(pk["best"], sp)
                pk["speedups"].append(sp)
                target_key = f"{kind}:{selector}"
                pt = per_target.setdefault(target_key, {"count": 0, "best": 0.0})
                pt["count"] += 1
                pt["best"] = max(pt["best"], sp)
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    pair_key = tuple(sorted([f"{a[0]}:{a[1]}", f"{b[0]}:{b[1]}"]))
                    pair_wins[pair_key] = pair_wins.get(pair_key, 0) + 1
        for stats in per_kind.values():
            ss = stats.pop("speedups")
            stats["avg"] = (sum(ss) / len(ss)) if ss else 0.0
        failure_targets: dict[str, int] = {}
        for c in failures:
            ivs = list(c.get("plan").interventions) if c.get("plan") else []
            for iv in ivs:
                key = f"{iv.target.kind}:{iv.target.selector}"
                failure_targets[key] = failure_targets.get(key, 0) + 1
        return json.dumps({
            "successful_count": len(successes),
            "failed_count": len(failures),
            "per_target_kind": per_kind,
            "per_target": dict(
                sorted(per_target.items(), key=lambda kv: -kv[1]["best"])[:12]
            ),
            "co_occurring_lever_pairs": [
                {"pair": list(k), "count": v}
                for k, v in sorted(pair_wins.items(), key=lambda kv: -kv[1])[:8]
            ],
            "failure_targets": dict(
                sorted(failure_targets.items(), key=lambda kv: -kv[1])[:8]
            ),
        }, indent=2, default=str)

    def compare_runs(self) -> str:
        """Return a leaderboard of (baseline + judged candidates) by median_ms."""

        rows: list[dict[str, Any]] = [{
            "candidate_id": "baseline",
            "median_ms": self.baseline_time.median_ms,
            "speedup_vs_baseline": 1.0,
        }]
        for cid, c in self.candidates.items():
            timing = c.get("timing")
            if timing and timing.median_ms:
                rows.append({
                    "candidate_id": cid,
                    "median_ms": timing.median_ms,
                    "speedup_vs_baseline": c.get("speedup"),
                    "rationale": c.get("rationale", ""),
                    "correctness_ok": c.get("correctness").ok if c.get("correctness") else None,
                })
        rows.sort(key=lambda r: r["median_ms"] if r["median_ms"] is not None else float("inf"))
        return json.dumps(rows, indent=2, default=str)

    # ---- agent prompts ----------------------------------------------------

    def system_instructions(self, *, harness_label: str = "pydantic-ai") -> str:
        return _instructions(
            spec=self.spec,
            baseline_median_ms=self.baseline_time.median_ms,
            harness_label=harness_label,
        )

    def starter_prompt(self) -> str:
        return _starter_prompt(
            spec=self.spec,
            run_id=self.run_id,
            baseline_median_ms=self.baseline_time.median_ms,
            search_space_size=len(self.search_space.levers),
            max_candidates=self.max_candidates,
        )

    def best_speedup(self) -> float | None:
        best: float | None = None
        for c in self.candidates.values():
            sp = c.get("speedup")
            if sp is None:
                continue
            if best is None or sp > best:
                best = sp
        return best

    def summary(self) -> dict[str, Any]:
        """Structured rollup of the run, used by the API and exit handlers."""

        best = self._best_validated_candidate()
        best_id = best["id"] if best else None
        best_sp = best.get("speedup") if best else None
        timing = best.get("timing") if best else None
        best_med = timing.median_ms if timing else None
        corr = best.get("correctness") if best else None
        best_corr = corr.ok if corr else None
        best_diff = corr.max_abs_diff if corr else None
        return {
            "run_id": self.run_id,
            "workload_id": self.spec.id,
            "backend_id": self.backend.id,
            "baseline_median_ms": self.baseline_time.median_ms,
            "best_speedup": best_sp,
            "best_candidate_id": best_id,
            "best_median_ms": best_med,
            "best_correctness_ok": best_corr,
            "best_max_abs_diff": best_diff,
            "successful_count": self.budget_state["successful_count"],
            "failed_attempts": self.budget_state["failed_attempts"],
            "max_candidates": self.max_candidates,
        }

    def _best_validated_candidate(self) -> dict[str, Any] | None:
        """Return the best candidate that **actually beat baseline** AND passed
        correctness (when checkable). Sub-1.0× and tolerance-failing candidates
        are excluded — the API surface should never report a regression as
        "best", and the optimized callable should never come from a candidate
        that drifted outside tolerance.
        """

        best: dict[str, Any] | None = None
        best_sp: float | None = None
        for c in self.candidates.values():
            sp = c.get("speedup")
            if not isinstance(sp, (int, float)) or sp <= 1.0:
                continue
            corr = c.get("correctness")
            if corr is not None and not corr.ok:
                continue
            if best_sp is None or sp > best_sp:
                best_sp = sp
                best = c
        return best

    def best_compiled_callable(self) -> Any | None:
        """Return the compiled callable from the best validated candidate.

        For TorchInductor this is the `torch.compile`d module/function; the
        user can call it directly with the same example_inputs the optimization
        ran against. Returns `None` if no candidate beat baseline within
        tolerance.
        """

        best = self._best_validated_candidate()
        if best is None:
            return None
        compile_outcome = best.get("compile")
        return getattr(compile_outcome, "compiled_callable", None)

    def best_plan(self) -> Any | None:
        """Return the `Plan` of the best validated candidate."""

        best = self._best_validated_candidate()
        return best.get("plan") if best else None


# ---------------------------------------------------------------------------
# pydantic-ai harness
# ---------------------------------------------------------------------------


def run_workload_optimization(
    *,
    workload_id: str,
    run_id: str,
    workspace_root: Path,
    trace_store: TraceStore,
    settings: CompilagentSettings | None = None,
    max_candidates: int = 4,
    harness: str = "pydantic_ai",
) -> dict[str, Any]:
    """Sync entry point for FastAPI background tasks.

    `harness` selects the agent driver:
      - "pydantic_ai" (default) — local pydantic-ai loop with @tool_plain.
      - "claude_agent_sdk" — Claude Agent SDK with an MCP server backed by
        the same `WorkloadSession` tool surface.
    """

    sett = settings or CompilagentSettings.from_env(project_root=workspace_root.parent)
    if harness == "claude_agent_sdk":
        return asyncio.run(_run_claude_sdk(
            workload_id=workload_id, run_id=run_id,
            workspace_root=workspace_root, trace_store=trace_store,
            settings=sett, max_candidates=max_candidates,
        ))
    return asyncio.run(_run_pydantic_ai(
        workload_id=workload_id, run_id=run_id,
        workspace_root=workspace_root, trace_store=trace_store,
        settings=sett, max_candidates=max_candidates,
    ))


async def _run_pydantic_ai(
    *,
    workload_id: str,
    run_id: str,
    workspace_root: Path,
    trace_store: TraceStore,
    settings: CompilagentSettings,
    max_candidates: int,
) -> dict[str, Any]:
    session = WorkloadSession(
        workload_id=workload_id, run_id=run_id,
        workspace_root=workspace_root, trace_store=trace_store,
        settings=settings, max_candidates=max_candidates,
    )
    agent: Agent[None, str] = Agent(
        model_for_settings(settings),
        name="compilagent-workload-optimizer",
        instructions=session.system_instructions(harness_label="pydantic-ai"),
    )

    # Register every WorkloadSession tool method on the agent. The retry
    # decorator turns any ValueError the method raises into a `ModelRetry`,
    # which pydantic-ai reflects back to the model. We bump `retries=5` so
    # weaker models (Mistral, smaller Anthropic) get multiple chances to
    # self-correct malformed JSON or invalid intervention payloads — the
    # default of 1 retry is too tight when one batch contains 4-5 candidates
    # worth of JSON.
    for method_name in TOOL_METHOD_NAMES:
        method = getattr(session, method_name)
        agent.tool_plain(_retry_on_value_error(method), retries=5)
    # Backend-supplied introspection tools (e.g. list_inductor_knobs).
    for extra in session.backend.list_introspection_tools():
        agent.tool_plain(extra.fn, retries=5)

    started_at = time.perf_counter()
    final_text: str | None = None
    try:
        async with agent.iter(
            session.starter_prompt(),
            model_settings=model_settings_for_settings(settings),
        ) as agent_run:
            async for node in agent_run:
                if Agent.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        await _stream_model_response(
                            stream=stream, trace_store=trace_store, run_id=run_id,
                            counters=session.part_counters,
                        )
                elif Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        await _stream_tool_events(
                            stream=stream, trace_store=trace_store, run_id=run_id,
                        )
        final_text = (
            str(agent_run.result.output) if agent_run.result is not None else None
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        trace_store.emit(
            "agent.run_failed",
            payload={
                "run_id": run_id,
                "elapsed_ms": elapsed_ms,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    trace_store.emit(
        "agent.run_completed",
        payload={
            "run_id": run_id,
            "elapsed_ms": elapsed_ms,
            "final_text": final_text,
            "best_speedup": session.best_speedup(),
        },
    )
    return {
        **session.summary(),
        "elapsed_ms": elapsed_ms,
        "final_text": final_text,
        # Live references the API uses to hand back a drop-in replacement
        # callable. NOT JSON-serializable; consumers that persist the dict
        # must drop these keys.
        "_session": session,
        "_compiled_callable": session.best_compiled_callable(),
        "_best_plan": session.best_plan(),
    }


# ---------------------------------------------------------------------------
# Claude Agent SDK harness
# ---------------------------------------------------------------------------


# JSON schemas for each WorkloadSession tool. Mirrors the docstrings; the SDK
# uses these to validate inputs and to surface argument names to the model.
_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "inspect_workload": {"type": "object", "properties": {}, "required": []},
    "inspect_search_space": {"type": "object", "properties": {}, "required": []},
    "propose_intervention": {
        "type": "object",
        "properties": {
            "target_kind": {"type": "string"},
            "target_selector": {"type": "string"},
            "payload_json": {"type": "string"},
            "rationale": {"type": "string", "default": ""},
        },
        "required": ["target_kind", "target_selector", "payload_json"],
    },
    "propose_candidate": {
        "type": "object",
        "properties": {
            "interventions_json": {"type": "string"},
            "description": {"type": "string"},
            "expected_effect": {"type": "string", "default": ""},
        },
        "required": ["interventions_json", "description"],
    },
    "propose_candidates": {
        "type": "object",
        "properties": {"plans_json": {"type": "string"}},
        "required": ["plans_json"],
    },
    "run_candidate": {
        "type": "object",
        "properties": {"candidate_id": {"type": "string"}},
        "required": ["candidate_id"],
    },
    "run_candidates": {
        "type": "object",
        "properties": {"candidate_ids_json": {"type": "string"}},
        "required": ["candidate_ids_json"],
    },
    "synthesize_findings": {"type": "object", "properties": {}, "required": []},
    "compare_runs": {"type": "object", "properties": {}, "required": []},
}


def build_workload_mcp_server(session: WorkloadSession, *, server_name: str = "compilagent_workload") -> Any:
    """Expose `WorkloadSession`'s tool surface as a Claude Agent SDK MCP server.

    Each `TOOL_METHOD_NAMES` entry is registered as an MCP tool; the
    description is the method's docstring (single source of truth across
    pydantic-ai and the SDK), the schema comes from `_TOOL_SCHEMAS`, and
    backend introspection tools (e.g. `list_inductor_knobs`) are also
    registered.
    """

    sdk = _import_sdk()
    tool = sdk.tool
    create_sdk_mcp_server = sdk.create_sdk_mcp_server
    tool_annotations = sdk.ToolAnnotations

    def _response(text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}]}

    def _error(exc: Exception) -> dict[str, Any]:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
        }

    tools: list[Any] = []
    # Methods that mutate session state (run / propose) need destructiveHint=True;
    # pure inspectors are read-only. Anything not in this set defaults to read-only.
    _MUTATING = {
        "propose_intervention", "propose_candidate", "propose_candidates",
        "run_candidate", "run_candidates",
    }

    def _add_tool(method_name: str) -> None:
        method = getattr(session, method_name)
        description = (method.__doc__ or method_name).strip()
        schema = _TOOL_SCHEMAS[method_name]
        is_read_only = method_name not in _MUTATING

        @tool(
            method_name,
            description,
            schema,
            annotations=tool_annotations(
                readOnlyHint=is_read_only,
                destructiveHint=not is_read_only,
                openWorldHint=False,
            ),
        )
        async def handler(args: dict[str, Any], *, _name: str = method_name) -> dict[str, Any]:
            try:
                bound = getattr(session, _name)
                return _response(bound(**args))
            except Exception as exc:  # noqa: BLE001
                return _error(exc)

        tools.append(handler)

    for name in TOOL_METHOD_NAMES:
        _add_tool(name)

    # Backend-supplied introspection tools. Each `IntrospectionTool` has
    # `name`, `description`, and a callable `fn` whose signature is the agent
    # tool surface — we reflect on it to derive a permissive JSON schema.
    import inspect as _inspect

    for extra in session.backend.list_introspection_tools():
        sig = _inspect.signature(extra.fn)
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            props[pname] = {"type": "string"}
            if param.default is _inspect.Parameter.empty:
                required.append(pname)
        extra_schema: dict[str, Any] = {"type": "object", "properties": props, "required": required}

        def _make_handler(fn: Any, _name: str = extra.name):
            @tool(
                _name,
                (fn.__doc__ or _name).strip(),
                extra_schema,
                annotations=tool_annotations(
                    readOnlyHint=True, destructiveHint=False, openWorldHint=False,
                ),
            )
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                try:
                    return _response(fn(**args))
                except Exception as exc:  # noqa: BLE001
                    return _error(exc)
            return handler

        tools.append(_make_handler(extra.fn))

    return create_sdk_mcp_server(name=server_name, version="1.0.0", tools=tools)


def _import_sdk() -> Any:
    try:
        import claude_agent_sdk
    except ImportError as exc:
        raise RuntimeError(
            "Claude Agent SDK harness requires `claude-agent-sdk`. "
            "Install with `env/bin/pip install claude-agent-sdk` before "
            "selecting this harness."
        ) from exc
    return claude_agent_sdk


async def _run_claude_sdk(
    *,
    workload_id: str,
    run_id: str,
    workspace_root: Path,
    trace_store: TraceStore,
    settings: CompilagentSettings,
    max_candidates: int,
) -> dict[str, Any]:
    """Drive the `WorkloadSession` through the Claude Agent SDK + MCP."""

    # The Claude Agent SDK is the `claude` CLI under the hood and only routes
    # to Anthropic models. Reject other providers up front with a clear
    # message — earlier the SDK would silently fail and produce an
    # incomprehensible error mid-session.
    model_name = settings.model_name or ""
    if not model_name.startswith("anthropic:") and "claude" not in model_name.lower():
        raise RuntimeError(
            f"Claude Agent SDK harness only supports Anthropic models; got "
            f"`{model_name}`. Either switch `harness` to `pydantic_ai` (which "
            "supports Mistral / OpenAI / Anthropic) or pass an `anthropic:...` "
            "model name."
        )

    session = WorkloadSession(
        workload_id=workload_id, run_id=run_id,
        workspace_root=workspace_root, trace_store=trace_store,
        settings=settings, max_candidates=max_candidates,
    )
    sdk = _import_sdk()
    server_name = "compilagent_workload"
    server = build_workload_mcp_server(session, server_name=server_name)

    started_at = time.perf_counter()
    allowed_prefix = f"mcp__{server_name}__"
    allowed_tools = [f"{allowed_prefix}{name}" for name in TOOL_METHOD_NAMES]
    for extra in session.backend.list_introspection_tools():
        allowed_tools.append(f"{allowed_prefix}{extra.name}")

    options = sdk.ClaudeAgentOptions(
        tools=[],
        allowed_tools=allowed_tools,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": session.system_instructions(harness_label="Claude Agent SDK"),
        },
        mcp_servers={server_name: server},
        permission_mode=settings.claude_sdk_permission_mode,
        cwd=str(workspace_root),
        model=settings.claude_sdk_model_name(),
        max_turns=settings.claude_sdk_max_turns,
        max_budget_usd=settings.claude_sdk_max_budget_usd,
        effort=settings.claude_sdk_effort_value(),
        setting_sources=["project"],
    )

    final_text: str | None = None
    final_chunks: list[str] = []
    client = sdk.ClaudeSDKClient(options=options)
    try:
        if hasattr(client, "connect"):
            await client.connect()
        await client.query(session.starter_prompt())
        async for message in client.receive_response():
            if hasattr(message, "result"):
                result = getattr(message, "result", None)
                if isinstance(result, str):
                    final_text = result
                trace_store.emit(
                    "agent.sdk_result",
                    payload={
                        "run_id": run_id,
                        "subtype": getattr(message, "subtype", None),
                        "session_id": getattr(message, "session_id", None),
                        "total_cost_usd": getattr(message, "total_cost_usd", None),
                    },
                )
            else:
                for chunk in _sdk_message_text_chunks(message):
                    if chunk:
                        final_chunks.append(chunk)
                        # Stream the assistant text into the UI as a single growing
                        # card so the user sees live output identical to the
                        # pydantic-ai path. Text/thinking events are emitted via
                        # incrementing part_id derived from the per-run counter.
                        session.part_counters["text"] += 1
                        pid = session.part_counters["text"]
                        trace_store.emit(
                            "agent.text_started",
                            payload={"run_id": run_id, "index": pid, "part_id": pid},
                        )
                        trace_store.emit(
                            "agent.text_delta",
                            payload={
                                "run_id": run_id, "index": pid, "part_id": pid,
                                "delta": chunk,
                            },
                        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        trace_store.emit(
            "agent.run_failed",
            payload={
                "run_id": run_id,
                "elapsed_ms": elapsed_ms,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if hasattr(client, "disconnect"):
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    if final_text is None and final_chunks:
        final_text = "\n".join(final_chunks).strip()
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    trace_store.emit(
        "agent.run_completed",
        payload={
            "run_id": run_id,
            "elapsed_ms": elapsed_ms,
            "final_text": final_text,
            "best_speedup": session.best_speedup(),
        },
    )
    return {
        **session.summary(),
        "elapsed_ms": elapsed_ms,
        "final_text": final_text,
        "_session": session,
        "_compiled_callable": session.best_compiled_callable(),
        "_best_plan": session.best_plan(),
    }


def _sdk_message_text_chunks(message: Any) -> list[str]:
    content = getattr(message, "content", None)
    if content is None and hasattr(message, "message"):
        content = getattr(message.message, "content", None)
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    chunks: list[str] = []
    for block in content:
        text = getattr(block, "text", None) or getattr(block, "content", None)
        if isinstance(text, str):
            chunks.append(text)
    return chunks


# ---------------------------------------------------------------------------
# Streaming + prompts
# ---------------------------------------------------------------------------


async def _stream_model_response(*, stream, trace_store, run_id, counters):
    """Translate pydantic-ai stream events into trace events the UI consumes."""

    part_ids: dict[tuple[str, int], int] = {}

    def _assign(kind: str, idx: int) -> int:
        key = (kind, idx)
        if key not in part_ids:
            counters[kind] += 1
            part_ids[key] = counters[kind]
        return part_ids[key]

    async for event in stream:
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, ThinkingPart):
                pid = _assign("thinking", event.index)
                trace_store.emit(
                    "agent.thinking_started",
                    payload={"run_id": run_id, "index": event.index, "part_id": pid},
                )
                if part.content:
                    trace_store.emit(
                        "agent.thinking_delta",
                        payload={
                            "run_id": run_id, "index": event.index, "part_id": pid,
                            "delta": part.content,
                        },
                    )
            elif isinstance(part, TextPart):
                pid = _assign("text", event.index)
                trace_store.emit(
                    "agent.text_started",
                    payload={"run_id": run_id, "index": event.index, "part_id": pid},
                )
                if part.content:
                    trace_store.emit(
                        "agent.text_delta",
                        payload={
                            "run_id": run_id, "index": event.index, "part_id": pid,
                            "delta": part.content,
                        },
                    )
            elif isinstance(part, ToolCallPart):
                continue
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, ThinkingPartDelta):
                content = getattr(delta, "content_delta", None)
                if content:
                    pid = _assign("thinking", event.index)
                    trace_store.emit(
                        "agent.thinking_delta",
                        payload={
                            "run_id": run_id, "index": event.index, "part_id": pid,
                            "delta": content,
                        },
                    )
            elif isinstance(delta, TextPartDelta):
                content = getattr(delta, "content_delta", None)
                if content:
                    pid = _assign("text", event.index)
                    trace_store.emit(
                        "agent.text_delta",
                        payload={
                            "run_id": run_id, "index": event.index, "part_id": pid,
                            "delta": content,
                        },
                    )


async def _stream_tool_events(*, stream, trace_store, run_id):
    async for event in stream:
        if isinstance(event, FunctionToolCallEvent):
            part = event.part
            try:
                args = part.args_as_dict()
            except Exception:  # noqa: BLE001
                args = getattr(part, "args", None)
            trace_store.emit(
                "agent.tool_call",
                payload={
                    "run_id": run_id, "tool": part.tool_name,
                    "tool_call_id": part.tool_call_id, "args": args,
                },
            )
        elif isinstance(event, FunctionToolResultEvent):
            result_part = getattr(event, "result", None)
            tool_call_id = getattr(result_part, "tool_call_id", None)
            content = getattr(result_part, "content", None)
            preview: str | None = None
            if isinstance(content, str):
                preview = content[:600]
            trace_store.emit(
                "agent.tool_result",
                payload={"run_id": run_id, "tool_call_id": tool_call_id, "preview": preview},
            )
        elif isinstance(event, (BuiltinToolCallEvent, BuiltinToolResultEvent)):
            continue


def _instructions(
    *,
    spec: WorkloadSpec,
    baseline_median_ms: float | None,
    harness_label: str = "pydantic-ai",
) -> str:
    return "\n".join((
        "You are a compiler-heuristic researcher. Your job is to REPLACE the",
        "compiler's hand-coded heuristics with experimentally validated",
        "decisions — **not** to autotune user-provided constants.",
        f"Active harness: {harness_label}.",
        "",
        "Forbidden surface (do NOT propose these — they are user inputs, not",
        "compiler decisions):",
        "  - Triton: `BLOCK_SIZE`, `num_warps`, `num_stages`, `num_ctas`,",
        "    `maxnreg`, `LOAD_CACHE_MODIFIER`, `eviction_policy` — all are",
        "    `tl.constexpr` / launch attributes the user already controls.",
        "    Sweeping them is what `@triton.autotune` does and is out of scope.",
        "",
        "Real surface (this is what the project exists to optimize):",
        "  - Triton MLIR passes: `tritongpu-coalesce` (layout selection),",
        "    `tritongpu-accelerate-matmul` (MMA tile / dot-operand layout),",
        "    `tritongpu-pipeline` (loop pipelining strategy),",
        "    `tritongpu-optimize-thread-locality`, `-remove-layout-conversions`,",
        "    `-reorder-instructions`, `-optimize-dot-operands`,",
        "    `-reduce-data-duplication`, etc. Skip / parameterize / replace.",
        "  - Inductor pass-like surface: scheduler `_pre_/_post_fusion_custom_pass`,",
        "    `register_lowering` swaps for ops, FX-graph rewrites, and",
        "    InductorChoices autotune-config heuristics. These are compiler",
        "    decisions, not user-provided constants.",
        "",
        "Your job is to form **specific, testable hypotheses** about which",
        "compiler decisions matter for THIS workload, then propose multi-knob",
        "candidates that test them as a unit.",
        "",
        "Budget rules:",
        "  - Only **successful** candidates (compile OK, timing measured,",
        "    correctness within tolerance when checkable) count toward the",
        "    trial budget. Failed compiles / out-of-bounds correctness do",
        "    NOT consume a slot.",
        "  - Each `run_candidate(...)` result includes `slots_remaining`,",
        "    `successful_count`, `failed_attempts`, `compile_diagnostics`,",
        "    `compile_warnings`, and `hint`. When a candidate fails, READ",
        "    those fields and propose a next candidate that addresses the",
        "    root cause (e.g. shared-memory overflow ⇒ pair different knobs;",
        "    correctness drift ⇒ avoid numerics-altering interventions).",
        "  - Stop when `slots_remaining` reaches 0 OR you have a clear winner.",
        "",
        f"Workload: `{spec.id}` ({spec.kind.value}, backend={spec.backend_id}).",
        f"Baseline median_ms: {baseline_median_ms!r}",
        "",
        "Each tool's full signature + docstring is already exposed to you by",
        "the runtime — read those, don't rely on a hand-edited list here.",
        "High-level guidance only:",
        "  - Inspect the workload + derived search space first.",
        "  - Prefer **batch** propose/run tools so you can synthesize across",
        "    multiple feedback sources in one round; one-knob probes are",
        "    rarely worth a slot.",
        "",
        "Decision policy:",
        "1. Read inspect_workload() AND inspect_search_space() before proposing.",
        "2. Form a NAMED hypothesis (e.g. 'memory-bound LayerNorm + residual-add",
        "   epilogue is the bottleneck; padding + epilogue_fusion + autotune",
        "   should expose more Tensor-Core matmul') and cite evidence from the",
        "   search-space `evidence` field.",
        "3. Propose a MULTI-INTERVENTION candidate via propose_candidate that",
        "   tests that hypothesis as a unit. Mixing 2-4 levers per candidate is",
        "   normal; one-lever candidates are usually wasted budget.",
        "4. After each run_candidate, summarise what the timing told you, then",
        "   propose the next candidate that contradicts or refines the prior.",
        "5. Accept ONLY a candidate with speedup_vs_baseline > 1.0 AND correctness",
        "   inside the workload's tolerance. Stop after at most 4-5 candidates.",
        "",
        "Be concise. Your FINAL reply must be a short markdown report naming",
        "the best candidate id (or stating none beat baseline), the speedup, and",
        "the multi-knob combination + the evidence that motivated it. Avoid",
        "narrating tool calls — the trace stream already shows them.",
    ))


def _starter_prompt(
    *, spec: WorkloadSpec, run_id: str, baseline_median_ms: float | None,
    search_space_size: int, max_candidates: int = 4,
) -> str:
    """Adapt the starter prompt to the chosen experiment budget."""

    if max_candidates <= 2:
        style = (
            "You have a TIGHT budget of {n} candidates. Pick ONE high-confidence "
            "hypothesis and design a single decisive multi-intervention candidate; "
            "use the second slot only if the first surprises you. Avoid scattered "
            "single-knob probes."
        )
    elif max_candidates <= 6:
        style = (
            "You have a budget of {n} candidates. Form 2-3 distinct hypotheses, "
            "test each as a multi-intervention candidate, and use the remaining "
            "slot(s) to refine the most promising direction. Each candidate "
            "should test something the previous ones didn't."
        )
    else:
        style = (
            "You have a wider budget of {n} candidates. Plan a small structured "
            "sweep: 3-4 candidates that each test a distinct hypothesis class "
            "(autotune family / fusion family / layout family / dispatch family), "
            "then spend the remaining budget refining the best direction (varying "
            "one knob at a time around the winner's settings)."
        )
    style = style.format(n=max_candidates)

    return (
        f"Workload `{spec.id}` is loaded (kind={spec.kind.value}, "
        f"backend={spec.backend_id}). Run id: `{run_id}`. The baseline has "
        f"already compiled + timed (median_ms={baseline_median_ms!r}); the "
        f"derived search space contains {search_space_size} levers, each with "
        "an `evidence` field linking it to a specific signal in the IR / "
        "op-count / device capability.\n\n"
        f"{style}\n\n"
        "Workflow:\n"
        "  1. Inspect the workload and the derived search space first.\n"
        "  2. Announce 2-3 NAMED hypotheses grounded in specific lever evidence.\n"
        "  3. Register the batch with the multi-candidate propose tool.\n"
        "  4. Run the batch with the multi-candidate run tool and read every\n"
        "     result together. Failed compiles do NOT consume budget slots.\n"
        "  5. Synthesize across the batch (per-target stats, co-occurring\n"
        "     winners, failure modes); design the next batch from that signal\n"
        "     rather than guesswork.\n"
        f"  6. Stop when `slots_remaining` hits 0 or you have a clear winner "
        f"(budget = {max_candidates}). End with a short markdown report citing "
        "the best candidate id, the speedup, and the multi-knob combination + "
        "evidence that motivated it."
    )
