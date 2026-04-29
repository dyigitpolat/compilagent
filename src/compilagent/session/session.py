"""`OptimizationSession` — the canonical optimization loop.

Construction is synchronous and runs the deterministic baseline phase
eagerly: compile baseline, analyze, derive the search space, time the
baseline, consult the cross-run policy. After construction the session is
read-only from the agent's POV; the only mutating tool calls are
`propose_*` and `run_*`.

The session holds **no** harness-specific state and imports neither
`pydantic_ai` nor `claude_agent_sdk`. The agent loop is owned by
`Harness.run`; this module just provides the tool handlers the harness
dispatches into and the bootstrap that prepares them.

Use `run_session(session, harness, request)` to drive an end-to-end run:
it forwards `StreamEvent`s from the harness into `ObservationEvent`s on the
session's sink and returns a `HarnessResult`-like summary.
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from compilagent.bootstrap import load_entry_point_integrations
from compilagent.core.analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    PassEvent,
    TimingResult,
)
from compilagent.core.backend import Backend, backend_registry
from compilagent.core.candidate_policy import CandidatePolicy, NullPolicy, PolicyHint
from compilagent.core.plan import Intervention, Plan, Target
from compilagent.core.workload import (
    ToleranceConfig,
    WorkloadInstance,
    WorkloadSpec,
)
from compilagent.core.workload_registry import workload_registry
from compilagent.harness.base import (
    Harness,
    HarnessResult,
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
)
from compilagent.observation.artifacts import (
    ArtifactRendererRegistry,
    artifact_renderer_registry,
)
from compilagent.observation.events import EventKind, ObservationEvent
from compilagent.observation.sink import NullSink, ObservationSink
from compilagent.storage.episode_store import EpisodeStore
from compilagent.storage.experiment_log import ExperimentLog
from compilagent.storage.workspace import OptimizationWorkspace
from compilagent.toolset import Toolset

from .leaderboard import best_validated_candidate, build_leaderboard
from .tools import build_session_toolset


def _loads_lenient(text: str) -> Any:
    """Permissive JSON parser used by the propose_* tools.

    LLMs routinely emit JSON with minor mistakes the strict parser rejects:
    smart quotes, trailing commas, lone control characters. We try strict
    first; on failure we sanitise and retry; final fallback is
    `ast.literal_eval`.
    """

    text = (text or "").strip()
    if not text:
        raise ValueError("empty JSON input")
    try:
        return json.loads(text)
    except json.JSONDecodeError as strict_err:
        cleaned = (
            text.replace("‘", "'")
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
        )
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            pass
        raise ValueError(
            f"could not parse JSON (strict and lenient parsers failed): {strict_err}"
        ) from strict_err


class OptimizationSession:
    """Per-run state + tool surface for one workload optimization."""

    def __init__(
        self,
        *,
        workload_id: str,
        run_id: str | None = None,
        workspace: OptimizationWorkspace,
        sink: ObservationSink | None = None,
        backend: Backend | None = None,
        max_candidates: int = 8,
        policy: CandidatePolicy | None = None,
        artifact_renderers: ArtifactRendererRegistry | None = None,
        experiment_log: ExperimentLog | None = None,
    ) -> None:
        # Out-of-tree integrations registered via setuptools entry points
        # are imported once per process — the call is a no-op on subsequent
        # sessions. Explicit `import compilagent.integrations.<x>` calls
        # before construction take precedence and remain authoritative.
        load_entry_point_integrations()

        self.workload_id = workload_id
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.workspace = workspace.ensure()
        self.sink: ObservationSink = sink or NullSink()
        self.max_candidates = max_candidates
        self.policy: CandidatePolicy = policy or NullPolicy()
        self.artifact_renderers = artifact_renderers or artifact_renderer_registry
        self.experiment_log = experiment_log or ExperimentLog(self.workspace.root)

        self.spec: WorkloadSpec = workload_registry.get_spec(workload_id)
        self.backend: Backend = backend or backend_registry.get(self.spec.backend_id)

        # Register backend-provided artifact renderers (idempotent best-effort).
        for renderer in self.backend.list_artifact_renderers() or ():
            with contextlib.suppress(TypeError, ValueError):
                self.artifact_renderers.register(renderer)

        self.workload_instance: WorkloadInstance = workload_registry.build(workload_id)

        baseline_dir = self.workspace.baseline_dir(workload_id, self.run_id)
        baseline_dir.mkdir(parents=True, exist_ok=True)

        self.budget_state: dict[str, int] = {
            "successful_count": 0,
            "failed_attempts": 0,
            "max_failed_attempts": max(8, max_candidates * 3),
        }
        self.candidates: dict[str, dict[str, Any]] = {}

        cap = self.backend.device_capability()
        self.arch = cap.arch

        self._emit(
            EventKind.SESSION_STARTED,
            payload={
                "run_id": self.run_id,
                "workload_id": workload_id,
                "backend_id": self.backend.id,
                "max_candidates": max_candidates,
                "device": {
                    "arch": cap.arch,
                    "name": cap.name,
                    "capability_int": cap.capability_int,
                },
            },
        )

        # ---- baseline compile + analysis ----
        self._emit(EventKind.COMPILE_STARTED, payload={"candidate_id": "baseline"})
        self.baseline_compile: CompileResult = self.backend.compile(
            self.spec,
            Plan(),
            artifact_dir=baseline_dir,
            pass_callback=self._make_pass_callback("baseline"),
        )
        self._emit(
            EventKind.COMPILE_COMPLETED,
            payload={
                "candidate_id": "baseline",
                "ok": self.baseline_compile.ok,
                "elapsed_ms": self.baseline_compile.elapsed_ms,
                "diagnostics": self.baseline_compile.diagnostics,
                "warnings": list(self.baseline_compile.warnings),
            },
        )
        self._emit_artifacts(self.baseline_compile, candidate_id="baseline")

        self.analysis: Analysis = self.backend.analyze(
            self.spec, baseline_artifacts=self.baseline_compile.artifacts
        )
        self.baseline_time: TimingResult = self.backend.time_workload(
            self.spec,
            Plan(),
            warmup=self.spec.budget.warmup,
            repetitions=self.spec.budget.repetitions,
            max_seconds=self.spec.budget.max_seconds,
        )
        self._emit(
            EventKind.BENCHMARK_COMPLETED,
            payload={
                "candidate_id": "baseline",
                "median_ms": self.baseline_time.median_ms,
                "p20_ms": self.baseline_time.p20_ms,
                "p80_ms": self.baseline_time.p80_ms,
            },
        )

        self.search_space = self.backend.derive_search_space(self.spec, self.analysis)
        self._emit(
            EventKind.SEARCH_SPACE_DERIVED,
            payload={
                "lever_count": len(self.search_space.levers),
                "backend_id": self.backend.id,
            },
        )

        # ---- cross-run hints ----
        family = self.backend.infer_workload_family(self.spec)
        self.family = family
        self.prior_hints: tuple[PolicyHint, ...] = tuple(
            self.policy.consult(
                workload=self.spec,
                analysis=self.analysis,
                family=family,
                arch=self.arch,
            )
        )

        # ---- agent toolset ----
        self.toolset: Toolset = build_session_toolset(self).with_extra(
            self.backend.list_introspection_tools() or ()
        )

    # ------------------------------------------------------------------ helpers

    def _emit(
        self,
        kind: EventKind | str,
        *,
        payload: Mapping[str, Any] | None = None,
        artifact_paths: Sequence[Path] | Sequence[str] | None = None,
        candidate_id: str | None = None,
    ) -> None:
        paths = tuple(str(p) for p in (artifact_paths or ()))
        self.sink.emit(
            ObservationEvent.make(
                kind,
                session_id=None,
                run_id=self.run_id,
                candidate_id=candidate_id,
                payload=payload,
                artifact_paths=paths,
            )
        )

    def _emit_artifacts(self, compile_result: CompileResult, *, candidate_id: str) -> None:
        for path in compile_result.artifacts:
            self._emit(
                EventKind.ARTIFACT_CREATED,
                payload={
                    "candidate_id": candidate_id,
                    "stage": Path(path).suffix.lstrip(".").lower() or "artifact",
                    "path": str(path),
                },
                artifact_paths=(path,),
                candidate_id=candidate_id,
            )

    def _make_pass_callback(self, candidate_id: str):
        def _cb(event: PassEvent) -> None:
            self._emit(
                EventKind.COMPILER_PASS,
                payload={
                    "candidate_id": candidate_id,
                    "stage": event.stage,
                    "name": event.name,
                    "duration_ms": event.duration_ms,
                    "action": event.action,
                    "ir_after_size": event.ir_after_size,
                    "error": event.error,
                },
                candidate_id=candidate_id,
            )

        return _cb

    def _new_candidate_id(self) -> str:
        return f"cand-{uuid.uuid4().hex[:10]}"

    def _intervention_from_dict(self, entry: Mapping[str, Any], *, where: str) -> Intervention:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} must be an object")
        kind = entry.get("target_kind")
        if not kind:
            raise ValueError(f"{where} is missing `target_kind`")
        return Intervention(
            target=Target(kind=str(kind), selector=str(entry.get("target_selector", ""))),
            payload=entry.get("payload"),
            rationale=str(entry.get("rationale") or ""),
        )

    def _validate_plan(self, ivs: Sequence[Intervention]) -> None:
        for i, iv in enumerate(ivs):
            result = self.backend.validate_intervention(iv)
            if not result.ok:
                raise ValueError(
                    f"intervention #{i} ({iv.target}) rejected by backend: "
                    + "; ".join(result.errors)
                )

    def _register_candidate(
        self,
        ivs: Sequence[Intervention],
        *,
        description: str,
        expected_effect: str,
    ) -> dict[str, Any]:
        self._validate_plan(ivs)
        cid = self._new_candidate_id()
        plan = Plan(interventions=tuple(ivs))
        readable = description or expected_effect or (
            ", ".join(f"{iv.target}={iv.payload!r}" for iv in ivs)
        )
        self.candidates[cid] = {
            "id": cid,
            "plan": plan,
            "description": description,
            "expected_effect": expected_effect,
            "rationale": expected_effect or description,
        }
        self._emit(
            EventKind.CANDIDATE_PROPOSED,
            payload={
                "candidate_id": cid,
                "description": readable,
                "expected_effect": expected_effect,
                "interventions": [iv.serialize() for iv in ivs],
            },
            candidate_id=cid,
        )
        return {
            "id": cid,
            "intervention_count": len(ivs),
            "description": description,
            "expected_effect": expected_effect,
        }

    # ------------------------------------------------------------------- tools

    def inspect_workload(self) -> str:
        """Return the workload spec, baseline timing, analysis summary, device,
        and any cross-run hints."""

        cap = self.backend.device_capability()
        return json.dumps(
            {
                "workload": self.spec.serialize(),
                "backend_id": self.backend.id,
                "device": {
                    "arch": cap.arch,
                    "name": cap.name,
                    "capability_int": cap.capability_int,
                    "memory_total_bytes": cap.memory_total_bytes,
                    "memory_peak_bandwidth_gbps": cap.memory_peak_bandwidth_gbps,
                },
                "baseline_timing": {
                    "median_ms": self.baseline_time.median_ms,
                    "p20_ms": self.baseline_time.p20_ms,
                    "p80_ms": self.baseline_time.p80_ms,
                },
                "analysis_summary": self.analysis.summary,
                "family": self.family,
                "prior_hints": [
                    {
                        "rationale": h.rationale,
                        "confidence": h.confidence,
                        "interventions": [iv.serialize() for iv in h.suggested_interventions],
                    }
                    for h in self.prior_hints
                ],
            },
            indent=2,
            default=str,
        )

    def inspect_search_space(self) -> str:
        """Return the derived lever catalog with evidence."""

        return json.dumps(
            {
                "workload_id": self.spec.id,
                "backend_id": self.backend.id,
                "lever_count": len(self.search_space.levers),
                "levers": [lever.serialize() for lever in self.search_space.levers],
            },
            indent=2,
            default=str,
        )

    def propose_candidate(
        self,
        *,
        interventions_json: str,
        description: str,
        expected_effect: str = "",
    ) -> str:
        """Register a multi-intervention candidate."""

        try:
            entries = _loads_lenient(interventions_json)
        except ValueError as exc:
            raise ValueError(f"interventions_json is not valid JSON: {exc}") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError("interventions_json must be a non-empty JSON list")
        ivs = [
            self._intervention_from_dict(entry, where=f"intervention #{i}")
            for i, entry in enumerate(entries)
        ]
        registered = self._register_candidate(
            ivs, description=description, expected_effect=expected_effect
        )
        return json.dumps(registered, indent=2)

    def propose_candidates(self, *, plans_json: str) -> str:
        """Register several candidates at once."""

        try:
            entries = _loads_lenient(plans_json)
        except ValueError as exc:
            raise ValueError(f"plans_json is not valid JSON: {exc}") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError("plans_json must be a non-empty JSON list of plan dicts")
        registered: list[dict[str, Any]] = []
        for i, plan_dict in enumerate(entries):
            if not isinstance(plan_dict, Mapping):
                raise ValueError(f"plan #{i} must be an object")
            ivs_raw = plan_dict.get("interventions") or []
            if not isinstance(ivs_raw, list) or not ivs_raw:
                raise ValueError(f"plan #{i} is missing a non-empty `interventions` list")
            ivs = [
                self._intervention_from_dict(iv, where=f"plan #{i} intervention #{j}")
                for j, iv in enumerate(ivs_raw)
            ]
            registered.append(
                self._register_candidate(
                    ivs,
                    description=str(plan_dict.get("description", "")),
                    expected_effect=str(plan_dict.get("expected_effect", "")),
                )
            )
        return json.dumps(
            {"registered": len(registered), "candidates": registered}, indent=2
        )

    def run_candidate(self, *, candidate_id: str) -> str:
        """Compile + time + correctness-check a previously proposed candidate."""

        if candidate_id not in self.candidates:
            known = list(self.candidates.keys())[-5:]
            raise ValueError(
                f"unknown candidate `{candidate_id}`; recent: {known}"
            )

        c = self.candidates[candidate_id]
        plan: Plan = c["plan"]

        try:
            self.backend.reset_between_compiles(self.spec)
        except Exception:
            # Backend-side cleanup must not break the run. Surface as warning.
            self.sink.emit_kv(
                EventKind.LOG_LINE,
                payload={"level": "warn", "message": "reset_between_compiles raised"},
                run_id=self.run_id,
                candidate_id=candidate_id,
            )

        plan = self.backend.interpret_plan(plan)
        cdir = self.workspace.candidate_dir(self.spec.id, self.run_id, candidate_id)
        cdir.mkdir(parents=True, exist_ok=True)

        self._emit(
            EventKind.COMPILE_STARTED,
            payload={"candidate_id": candidate_id},
            candidate_id=candidate_id,
        )
        compile_outcome: CompileResult = self.backend.compile(
            self.spec,
            plan,
            artifact_dir=cdir,
            pass_callback=self._make_pass_callback(candidate_id),
        )
        self._emit(
            EventKind.COMPILE_COMPLETED,
            payload={
                "candidate_id": candidate_id,
                "ok": compile_outcome.ok,
                "elapsed_ms": compile_outcome.elapsed_ms,
                "diagnostics": compile_outcome.diagnostics,
                "warnings": list(compile_outcome.warnings),
            },
            candidate_id=candidate_id,
        )
        self._emit_artifacts(compile_outcome, candidate_id=candidate_id)

        timing: TimingResult | None = None
        correctness: CorrectnessResult | None = None
        if compile_outcome.ok:
            timing = self.backend.time_workload(
                self.spec,
                plan,
                warmup=self.spec.budget.warmup,
                repetitions=self.spec.budget.repetitions,
                max_seconds=self.spec.budget.max_seconds,
            )
            if self.baseline_compile.ok:
                tol = ToleranceConfig(
                    atol=self.spec.tolerance.atol,
                    rtol=self.spec.tolerance.rtol,
                    notes=self.spec.tolerance.notes,
                )
                try:
                    correctness = self.backend.validate_correctness(
                        self.spec, self.baseline_compile, compile_outcome, tol
                    )
                except Exception as exc:  # noqa: BLE001
                    correctness = None
                    self.sink.emit_kv(
                        EventKind.LOG_LINE,
                        payload={
                            "level": "warn",
                            "message": f"correctness check failed: {exc!r}",
                        },
                        run_id=self.run_id,
                        candidate_id=candidate_id,
                    )

        speedup: float | None = (
            (self.baseline_time.median_ms / timing.median_ms)
            if timing
            and timing.median_ms
            and self.baseline_time.median_ms
            else None
        )

        c.update(
            {
                "compile": compile_outcome,
                "timing": timing,
                "correctness": correctness,
                "speedup": speedup,
            }
        )

        successful = bool(
            compile_outcome.ok
            and timing
            and timing.median_ms is not None
            and (correctness is None or correctness.ok)
        )
        if successful:
            self.budget_state["successful_count"] += 1
        else:
            self.budget_state["failed_attempts"] += 1

        slots_remaining = max(
            0, self.max_candidates - self.budget_state["successful_count"]
        )

        self._emit(
            EventKind.BENCHMARK_COMPLETED,
            payload={
                "candidate_id": candidate_id,
                "median_ms": timing.median_ms if timing else None,
                "p20_ms": timing.p20_ms if timing else None,
                "p80_ms": timing.p80_ms if timing else None,
                "speedup_vs_baseline": speedup,
                "correctness_ok": correctness.ok if correctness else None,
            },
            candidate_id=candidate_id,
        )
        self._emit(
            EventKind.RUN_PROGRESS,
            payload={
                "successful_count": self.budget_state["successful_count"],
                "failed_attempts": self.budget_state["failed_attempts"],
                "max_candidates": self.max_candidates,
                "slots_remaining": slots_remaining,
            },
        )
        self._emit(
            EventKind.LEADERBOARD_UPDATED,
            payload={
                "rows": [
                    row.serialize()
                    for row in build_leaderboard(
                        baseline_median_ms=self.baseline_time.median_ms,
                        candidates=[
                            {
                                "id": cid,
                                "median_ms": (
                                    cc.get("timing").median_ms
                                    if cc.get("timing")
                                    else None
                                ),
                                "speedup_vs_baseline": cc.get("speedup"),
                                "correctness_ok": (
                                    cc.get("correctness").ok
                                    if cc.get("correctness")
                                    else None
                                ),
                                "rationale": cc.get("rationale", ""),
                            }
                            for cid, cc in self.candidates.items()
                        ],
                    )
                ]
            },
        )

        if successful:
            self.experiment_log.append(
                {
                    "run_id": self.run_id,
                    "workload_id": self.spec.id,
                    "backend_id": self.backend.id,
                    "family": self.family,
                    "arch": self.arch,
                    "successful": True,
                    "median_ms": timing.median_ms if timing else None,
                    "speedup": speedup,
                    "correctness_ok": correctness.ok if correctness else None,
                    "interventions": [iv.serialize() for iv in plan.interventions],
                }
            )
        elif compile_outcome.ok is False:
            self.experiment_log.append(
                {
                    "run_id": self.run_id,
                    "workload_id": self.spec.id,
                    "backend_id": self.backend.id,
                    "family": self.family,
                    "arch": self.arch,
                    "successful": False,
                    "diagnostics": compile_outcome.diagnostics,
                    "interventions": [iv.serialize() for iv in plan.interventions],
                }
            )

        if not successful:
            self._emit(
                EventKind.CANDIDATE_REJECTED,
                payload={
                    "candidate_id": candidate_id,
                    "reason": (
                        "compile_failed"
                        if not compile_outcome.ok
                        else (
                            "correctness_drift"
                            if correctness and not correctness.ok
                            else "no_timing"
                        )
                    ),
                },
                candidate_id=candidate_id,
            )

        hint = self._build_run_hint(compile_outcome, correctness)
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "compile_ok": compile_outcome.ok,
                "median_ms": timing.median_ms if timing else None,
                "speedup_vs_baseline": speedup,
                "correctness_ok": correctness.ok if correctness else None,
                "max_abs_diff": correctness.max_abs_diff if correctness else None,
                "compile_diagnostics": compile_outcome.diagnostics,
                "compile_warnings": list(compile_outcome.warnings)[:6],
                "successful": successful,
                "successful_count": self.budget_state["successful_count"],
                "failed_attempts": self.budget_state["failed_attempts"],
                "slots_remaining": slots_remaining,
                "hint": hint,
            },
            indent=2,
            default=str,
        )

    def run_candidates(self, *, candidate_ids_json: str) -> str:
        """Run a batch of previously proposed candidates in sequence."""

        try:
            ids = _loads_lenient(candidate_ids_json)
        except ValueError as exc:
            raise ValueError(f"candidate_ids_json is not valid JSON: {exc}") from exc
        if not isinstance(ids, list) or not ids:
            raise ValueError("candidate_ids_json must be a non-empty list")
        results: list[dict[str, Any]] = []
        for cid in ids:
            try:
                results.append(json.loads(self.run_candidate(candidate_id=str(cid))))
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001
                results.append({"candidate_id": str(cid), "error": repr(exc)})
        successes = [r for r in results if r.get("successful")]
        best = best_validated_candidate(results)
        return json.dumps(
            {
                "ran": len(results),
                "successful": len(successes),
                "failed": len(results) - len(successes),
                "best": best,
                "results": results,
                "slots_remaining": max(
                    0, self.max_candidates - self.budget_state["successful_count"]
                ),
            },
            indent=2,
            default=str,
        )

    def synthesize_findings(self) -> str:
        """Aggregate the current run's results."""

        successes = [c for c in self.candidates.values() if c.get("speedup")]
        failures = [
            c
            for c in self.candidates.values()
            if c.get("compile") and not getattr(c["compile"], "ok", True)
        ]
        per_kind: dict[str, dict[str, Any]] = {}
        per_target: dict[str, dict[str, Any]] = {}
        pair_wins: dict[tuple[str, str], int] = {}
        for c in successes:
            sp = c.get("speedup") or 0.0
            ivs = list(c["plan"].interventions) if c.get("plan") else []
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
                for b in keys[i + 1 :]:
                    pair_key = tuple(sorted([f"{a[0]}:{a[1]}", f"{b[0]}:{b[1]}"]))
                    pair_wins[pair_key] = pair_wins.get(pair_key, 0) + 1
        for stats in per_kind.values():
            ss = stats.pop("speedups")
            stats["avg"] = (sum(ss) / len(ss)) if ss else 0.0
        failure_targets: dict[str, int] = {}
        for c in failures:
            ivs = list(c["plan"].interventions) if c.get("plan") else []
            for iv in ivs:
                key = f"{iv.target.kind}:{iv.target.selector}"
                failure_targets[key] = failure_targets.get(key, 0) + 1
        return json.dumps(
            {
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
            },
            indent=2,
            default=str,
        )

    def compare_runs(self) -> str:
        """Return a leaderboard of (baseline + judged candidates)."""

        rows = build_leaderboard(
            baseline_median_ms=self.baseline_time.median_ms,
            candidates=[
                {
                    "id": cid,
                    "median_ms": (
                        c.get("timing").median_ms if c.get("timing") else None
                    ),
                    "speedup_vs_baseline": c.get("speedup"),
                    "correctness_ok": (
                        c.get("correctness").ok if c.get("correctness") else None
                    ),
                    "rationale": c.get("rationale", ""),
                }
                for cid, c in self.candidates.items()
            ],
        )
        return json.dumps([row.serialize() for row in rows], indent=2, default=str)

    # --------------------------------------------------------------- finalisation

    def finalize(self, *, episode_path: Path | None = None) -> dict[str, Any]:
        """Persist the closed session as one episode JSON document."""

        path = episode_path or self.workspace.episode_path(self.spec.id, self.run_id)
        rows = build_leaderboard(
            baseline_median_ms=self.baseline_time.median_ms,
            candidates=[
                {
                    "id": cid,
                    "median_ms": (
                        c.get("timing").median_ms if c.get("timing") else None
                    ),
                    "speedup_vs_baseline": c.get("speedup"),
                    "correctness_ok": (
                        c.get("correctness").ok if c.get("correctness") else None
                    ),
                    "rationale": c.get("rationale", ""),
                }
                for cid, c in self.candidates.items()
            ],
        )
        episode = {
            "run_id": self.run_id,
            "workload": self.spec.serialize(),
            "backend_id": self.backend.id,
            "arch": self.arch,
            "family": self.family,
            "baseline": {
                "ok": self.baseline_compile.ok,
                "median_ms": self.baseline_time.median_ms,
            },
            "leaderboard": [row.serialize() for row in rows],
            "successful_count": self.budget_state["successful_count"],
            "failed_attempts": self.budget_state["failed_attempts"],
        }
        EpisodeStore(path).save(episode)
        self._emit(
            EventKind.SESSION_FINISHED,
            payload={"episode_path": str(path), **episode},
            artifact_paths=(path,),
        )
        return episode

    # ------------------------------------------------------------------ hint

    def _build_run_hint(
        self,
        compile_outcome: CompileResult,
        correctness: CorrectnessResult | None,
    ) -> str | None:
        if not compile_outcome.ok:
            warnings = "; ".join(list(compile_outcome.warnings)[:3])
            if warnings:
                return f"compile failed; warnings: {warnings} — adjust knobs accordingly."
            return (
                "compile failed; see compile_diagnostics. Adjust the plan and "
                "propose a new candidate. This attempt did NOT count against "
                "your trial budget."
            )
        if correctness is not None and not correctness.ok:
            return (
                "compile + run succeeded but the candidate output drifted "
                f"outside tolerance (max_abs_diff={correctness.max_abs_diff}). "
                "Pick interventions that don't change numerics. This attempt "
                "did NOT count against your trial budget."
            )
        return None


# ============================================================================
# async driver
# ============================================================================


async def run_session(
    *,
    session: OptimizationSession,
    harness: Harness,
    request: HarnessRunRequest,
) -> HarnessResult:
    """Drive `session` against `harness` and translate the stream into events.

    Tool calls are dispatched by the harness; this driver only mirrors the
    `StreamEvent` stream into the session's `ObservationSink`.
    """

    started = time.perf_counter()
    final_text: str | None = None
    metadata: dict[str, Any] = {}
    failed = False

    iterator: AsyncIterator[StreamEvent] = harness.run(request)
    async for event in iterator:
        _translate_stream_event(session, event)
        if event.kind is StreamEventKind.RUN_FINISHED:
            final_text = event.text
            metadata = dict(event.extra or {})
        elif event.kind is StreamEventKind.RUN_FAILED:
            failed = True
            metadata = dict(event.extra or {})

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if failed:
        session.sink.emit_kv(
            EventKind.SESSION_FAILED,
            payload={"elapsed_ms": elapsed_ms, **metadata},
            run_id=session.run_id,
        )
    return HarnessResult(final_text=final_text, elapsed_ms=elapsed_ms, metadata=metadata)


def _translate_stream_event(session: OptimizationSession, event: StreamEvent) -> None:
    """Translate one harness `StreamEvent` into one `ObservationEvent`."""

    if event.kind is StreamEventKind.THINKING_STARTED:
        session.sink.emit_kv(
            EventKind.AGENT_THINKING_STARTED,
            payload={"part_index": event.part_index},
            run_id=session.run_id,
        )
    elif event.kind is StreamEventKind.THINKING_DELTA:
        session.sink.emit_kv(
            EventKind.AGENT_THINKING_DELTA,
            payload={"part_index": event.part_index, "text": event.text or ""},
            run_id=session.run_id,
        )
    elif event.kind is StreamEventKind.TEXT_STARTED:
        session.sink.emit_kv(
            EventKind.AGENT_TEXT_STARTED,
            payload={"part_index": event.part_index},
            run_id=session.run_id,
        )
    elif event.kind is StreamEventKind.TEXT_DELTA:
        session.sink.emit_kv(
            EventKind.AGENT_TEXT_DELTA,
            payload={"part_index": event.part_index, "text": event.text or ""},
            run_id=session.run_id,
        )
    elif event.kind is StreamEventKind.TOOL_CALL:
        session.sink.emit_kv(
            EventKind.TOOL_CALL_STARTED,
            payload={
                "tool_name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "args": event.tool_args,
            },
            run_id=session.run_id,
        )
    elif event.kind is StreamEventKind.TOOL_RESULT:
        session.sink.emit_kv(
            EventKind.TOOL_CALL_COMPLETED,
            payload={
                "tool_name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "result": event.tool_result,
            },
            run_id=session.run_id,
        )
    elif event.kind is StreamEventKind.TOOL_ERROR:
        session.sink.emit_kv(
            EventKind.TOOL_CALL_FAILED,
            payload={
                "tool_name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "error_type": event.error_type,
                "error_message": event.error_message,
            },
            run_id=session.run_id,
        )
    # RUN_FINISHED / RUN_FAILED are handled by run_session()
