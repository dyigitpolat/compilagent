"""E9 — the session calls the optional `CandidatePolicy.observe` hook.

`observe` is duck-typed (not a Protocol member): consult-only policies keep
working, a recording policy sees every `run_candidate` outcome, and a
raising policy must never break the run (the failure is downgraded to a
LOG_LINE warning).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compilagent.core.analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    TimingResult,
)
from compilagent.core.backend import backend_registry
from compilagent.core.candidate_policy import NullPolicy
from compilagent.core.plan import Plan, ValidationResult
from compilagent.core.search_space import SearchSpace
from compilagent.core.workload import (
    BenchmarkBudget,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.core.workload_registry import register_workload
from compilagent.observation.events import EventKind
from compilagent.observation.sink import CapturingSink
from compilagent.session.session import OptimizationSession
from compilagent.storage.workspace import OptimizationWorkspace


@dataclass
class _ObserveBackend:
    """Fake backend: `knob:fail` compiles to failure, anything else wins."""

    id: str = "observe_fake"
    artifact_stages: tuple[str, ...] = ("ir",)

    def device_capability(self) -> DeviceCapability:
        return DeviceCapability(
            arch="cpu",
            capability_int=None,
            name="Fake",
            memory_total_bytes=None,
            memory_peak_bandwidth_gbps=None,
        )

    def analyze(self, workload, *, baseline_artifacts) -> Analysis:
        return Analysis(summary={"kind": workload.kind.value})

    def derive_search_space(self, workload, analysis) -> SearchSpace:
        return SearchSpace(workload_id=workload.id, backend_id=self.id)

    def validate_intervention(self, intervention) -> ValidationResult:
        return ValidationResult(ok=True)

    def interpret_plan(self, plan: Plan) -> Plan:
        return plan

    def apply_intervention(self, plan, intervention) -> Plan:
        return Plan(interventions=plan.interventions + (intervention,))

    def compile(self, workload, plan, *, artifact_dir, pass_callback=None):
        fails = any(iv.target.selector == "fail" for iv in plan.interventions)
        return CompileResult(
            ok=not fails,
            elapsed_ms=1.0,
            diagnostics="boom" if fails else None,
        )

    def time_workload(self, workload, plan, *, warmup, repetitions, max_seconds=None):
        ms = 10.0 if plan.is_empty else 5.0
        return TimingResult(timings_ms=(ms,) * 3, median_ms=ms, p20_ms=ms, p80_ms=ms)

    def validate_correctness(self, workload, baseline, candidate, tolerance):
        return CorrectnessResult(ok=True)

    def reset_between_compiles(self, workload) -> None:
        return None

    def list_introspection_tools(self) -> Sequence:
        return ()

    def list_artifact_renderers(self) -> Sequence:
        return ()

    def infer_workload_family(self, workload) -> str | None:
        return "fake_family"

    def objectives_for_candidate(self, workload, plan, compile_result, timing_result):
        return {}


class _RecordingPolicy(NullPolicy):
    name = "recording"

    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []

    def observe(self, **kwargs: Any) -> None:
        self.observations.append(kwargs)


class _RaisingPolicy(NullPolicy):
    name = "raising"

    def observe(self, **kwargs: Any) -> None:
        raise RuntimeError("policy exploded")


def _make_session(tmp_path: Path, policy) -> OptimizationSession:
    backend_registry.register("observe_fake", _ObserveBackend)
    spec = WorkloadSpec(
        id="observe_workload",
        title="observe",
        description="observe hook fixture",
        kind=WorkloadKind.KERNEL,
        backend_id="observe_fake",
        tolerance=ToleranceConfig(),
        budget=BenchmarkBudget(warmup=1, repetitions=3, max_seconds=5.0),
    )

    @register_workload(spec)
    def _build(s: WorkloadSpec) -> WorkloadInstance:
        return WorkloadInstance(spec=s, forward=lambda: None)

    return OptimizationSession(
        workload_id="observe_workload",
        workspace=OptimizationWorkspace(session_cwd=tmp_path),
        sink=CapturingSink(),
        max_candidates=2,
        policy=policy,
    )


def _propose_and_run(session: OptimizationSession, selector: str) -> dict[str, Any]:
    registered = json.loads(
        session.propose_candidate(
            interventions=[
                {
                    "target_kind": "knob",
                    "target_selector": selector,
                    "payload": {"v": 1},
                    "rationale": "test",
                }
            ],
            description=f"knob {selector}",
        )
    )
    return json.loads(session.run_candidate(candidate_id=registered["id"]))


def test_observe_fires_once_per_run_candidate_with_full_outcome(tmp_path):
    policy = _RecordingPolicy()
    session = _make_session(tmp_path, policy)

    ok = _propose_and_run(session, "win")
    bad = _propose_and_run(session, "fail")

    assert len(policy.observations) == 2
    first, second = policy.observations

    assert first["workload"].id == "observe_workload"
    assert first["candidate_id"] == ok["candidate_id"]
    assert isinstance(first["plan"], Plan)
    assert first["compile_result"].ok is True
    assert first["timing"].median_ms == 5.0
    assert first["correctness"].ok is True
    assert first["speedup"] == 2.0
    assert first["successful"] is True
    assert first["family"] == "fake_family"
    assert first["arch"] == "cpu"

    assert second["candidate_id"] == bad["candidate_id"]
    assert second["successful"] is False
    assert second["compile_result"].ok is False
    assert second["timing"] is None
    assert second["speedup"] is None


def test_raising_observe_downgrades_to_warning(tmp_path):
    session = _make_session(tmp_path, _RaisingPolicy())

    result = _propose_and_run(session, "win")

    assert result["successful"] is True  # the run itself is unaffected
    warnings = [
        e
        for e in session.sink.events
        if e.kind == EventKind.LOG_LINE.value
        and "policy.observe raised" in str(e.payload.get("message"))
    ]
    assert len(warnings) == 1


def test_consult_only_policy_still_works(tmp_path):
    class _ConsultOnly:
        name = "consult_only"

        def consult(self, *, workload, analysis, family, arch):
            return ()

    session = _make_session(tmp_path, _ConsultOnly())
    result = _propose_and_run(session, "win")
    assert result["successful"] is True


def test_null_policy_observe_is_a_noop():
    policy = NullPolicy()
    assert (
        policy.observe(
            workload=WorkloadSpec(
                id="x",
                title="x",
                description="x",
                kind=WorkloadKind.KERNEL,
                backend_id="fake",
            ),
            candidate_id="cand-1",
            plan=Plan(),
            compile_result=CompileResult(ok=True),
            timing=None,
            correctness=None,
            speedup=None,
            successful=False,
            family=None,
            arch="cpu",
        )
        is None
    )
