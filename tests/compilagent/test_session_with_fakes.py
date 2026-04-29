"""End-to-end session test driven by a synthetic Backend + Harness."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from compilagent.core.analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    PassCallback,
    TimingResult,
)
from compilagent.core.backend import backend_registry
from compilagent.core.plan import Intervention, Plan, ValidationResult
from compilagent.core.search_space import (
    DerivationEvidence,
    EnumChoice,
    Lever,
    SearchSpace,
)
from compilagent.core.tool_decl import ToolDecl
from compilagent.core.workload import (
    BenchmarkBudget,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.core.workload_registry import register_workload
from compilagent.harness.base import HarnessRunRequest, StreamEvent, StreamEventKind
from compilagent.observation.events import EventKind
from compilagent.observation.sink import CapturingSink
from compilagent.session.session import OptimizationSession, run_session
from compilagent.storage.workspace import OptimizationWorkspace

# ---------------------------------------------------------------- fake backend


@dataclass
class _FakeBackend:
    id: str = "fake"
    artifact_stages: tuple[str, ...] = ("ir", "asm")
    _baseline_calls: int = 0

    def device_capability(self) -> DeviceCapability:
        return DeviceCapability(
            arch="cpu",
            capability_int=None,
            name="Fake CPU",
            memory_total_bytes=None,
            memory_peak_bandwidth_gbps=None,
        )

    def analyze(self, workload: WorkloadSpec, *, baseline_artifacts: Sequence[Path]) -> Analysis:
        return Analysis(
            summary={
                "kind": workload.kind.value,
                "tensor_shapes": {"x": [1024]},
                "dtypes": ["fp32"],
                "op_counts": {"add": 1},
            }
        )

    def derive_search_space(self, workload: WorkloadSpec, analysis: Analysis) -> SearchSpace:
        lever = Lever(
            id="kernel.flag",
            target_kind="knob",
            target_selector="optimize",
            range=EnumChoice(candidates=("on", "off")),
            default="off",
            description="Toggle the fake optimisation flag.",
            evidence=DerivationEvidence(
                rule="fake.lever", signal="synthetic", citations=()
            ),
            backend_id=self.id,
        )
        return SearchSpace(
            workload_id=workload.id,
            backend_id=self.id,
            levers=(lever,),
        )

    def validate_intervention(self, intervention: Intervention) -> ValidationResult:
        if intervention.target.kind == "knob":
            return ValidationResult(ok=True)
        return ValidationResult(ok=False, errors=(f"unknown kind {intervention.target.kind}",))

    def interpret_plan(self, plan: Plan) -> Plan:
        return plan

    def apply_intervention(self, plan: Plan, intervention: Intervention) -> Plan:
        return Plan(interventions=plan.interventions + (intervention,))

    def compile(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        artifact_dir: Path,
        pass_callback: PassCallback | None = None,
    ) -> CompileResult:
        artifact = artifact_dir / "out.ir"
        artifact.write_text("fake ir", encoding="utf-8")
        return CompileResult(
            ok=True,
            elapsed_ms=1.0,
            artifacts=(artifact,),
            compiled_callable=lambda: None,
            metadata={"plan_size": len(plan.interventions)},
        )

    def time_workload(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        warmup: int,
        repetitions: int,
        max_seconds: float | None = None,
    ) -> TimingResult:
        # Baseline: 10ms median; any non-empty plan: 5ms median (2× speedup).
        timings = (10.0,) * repetitions if plan.is_empty else (5.0,) * repetitions
        median = timings[len(timings) // 2]
        return TimingResult(
            timings_ms=timings,
            median_ms=median,
            p20_ms=median,
            p80_ms=median,
        )

    def validate_correctness(
        self,
        workload: WorkloadSpec,
        baseline: CompileResult,
        candidate: CompileResult,
        tolerance: ToleranceConfig,
    ) -> CorrectnessResult:
        return CorrectnessResult(ok=True, max_abs_diff=0.0, max_rel_diff=0.0)

    def reset_between_compiles(self, workload: WorkloadSpec) -> None:
        return None

    def list_introspection_tools(self) -> Sequence[ToolDecl]:
        return ()

    def list_artifact_renderers(self) -> Sequence[object]:
        return ()

    def infer_workload_family(self, workload: WorkloadSpec) -> str | None:
        return "fake_family"


# ---------------------------------------------------------------- fake harness


@dataclass
class _ToolCall:
    name: str
    args: dict


@dataclass
class _FakeHarness:
    """Drives a deterministic sequence of tool calls against the session."""

    calls: list[_ToolCall] = field(default_factory=list)
    id: str = "fake_harness"
    supported_providers: tuple[str, ...] = ("fake",)

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        toolset = request.toolset

        async def _yield_call(name: str, args: dict, call_id: str):
            yield StreamEvent(
                kind=StreamEventKind.TOOL_CALL,
                tool_name=name,
                tool_call_id=call_id,
                tool_args=args,
            )
            try:
                result = toolset.by_name(name).handler(args)
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_RESULT,
                    tool_name=name,
                    tool_call_id=call_id,
                    tool_result=result,
                )
            except ValueError as exc:
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_ERROR,
                    tool_name=name,
                    tool_call_id=call_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        yield StreamEvent(kind=StreamEventKind.THINKING_STARTED, part_index=0)
        yield StreamEvent(
            kind=StreamEventKind.THINKING_DELTA, part_index=0, text="planning..."
        )

        async for ev in _yield_call("inspect_workload", {}, "c1"):
            yield ev
        async for ev in _yield_call("inspect_search_space", {}, "c2"):
            yield ev

        # propose a single-intervention candidate on the fake lever
        propose_args = {
            "interventions_json": json.dumps(
                [
                    {
                        "target_kind": "knob",
                        "target_selector": "optimize",
                        "payload": "on",
                        "rationale": "try the fake flag",
                    }
                ]
            ),
            "description": "enable optimize flag",
            "expected_effect": "should make it faster",
        }
        registered_id: str | None = None
        async for ev in _yield_call("propose_candidate", propose_args, "c3"):
            if ev.kind is StreamEventKind.TOOL_RESULT and ev.tool_result:
                registered_id = json.loads(ev.tool_result)["id"]
            yield ev

        assert registered_id is not None
        async for ev in _yield_call(
            "run_candidate", {"candidate_id": registered_id}, "c4"
        ):
            yield ev

        async for ev in _yield_call("compare_runs", {}, "c5"):
            yield ev
        async for ev in _yield_call("synthesize_findings", {}, "c6"):
            yield ev

        yield StreamEvent(kind=StreamEventKind.TEXT_STARTED, part_index=1)
        yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, part_index=1, text="done")
        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text="done",
            extra={"turns": 1},
        )


# ----------------------------------------------------------------------- test


@pytest.fixture
def _registered_workload():
    spec = WorkloadSpec(
        id="fake_workload",
        title="Fake workload",
        description="Synthetic test workload",
        kind=WorkloadKind.KERNEL,
        backend_id="fake",
        tolerance=ToleranceConfig(atol=1e-6, rtol=1e-6),
        budget=BenchmarkBudget(warmup=1, repetitions=3, max_seconds=1.0),
    )

    @register_workload(spec)
    def _build(s: WorkloadSpec) -> WorkloadInstance:
        return WorkloadInstance(spec=s, forward=lambda: None)

    return spec


def test_session_with_fakes_drives_one_run(tmp_path: Path, _registered_workload):
    backend_registry.register("fake", _FakeBackend)
    workspace = OptimizationWorkspace(session_cwd=tmp_path)
    sink = CapturingSink()

    session = OptimizationSession(
        workload_id="fake_workload",
        run_id="run-test",
        workspace=workspace,
        sink=sink,
        max_candidates=4,
    )

    # Bootstrap should have emitted these in order:
    bootstrap_kinds = sink.kinds()
    assert EventKind.SESSION_STARTED.value in bootstrap_kinds
    assert EventKind.COMPILE_COMPLETED.value in bootstrap_kinds
    assert EventKind.BENCHMARK_COMPLETED.value in bootstrap_kinds
    assert EventKind.SEARCH_SPACE_DERIVED.value in bootstrap_kinds

    # Drive the harness
    harness = _FakeHarness()
    request = HarnessRunRequest(
        toolset=session.toolset,
        system_instructions="be brief",
        user_prompt="optimise me",
        model_id="fake:fake-1",
    )
    result = asyncio.run(run_session(session=session, harness=harness, request=request))

    assert result.final_text == "done"
    assert result.metadata.get("turns") == 1

    # The candidate ran and was successful
    assert session.budget_state["successful_count"] == 1
    assert session.budget_state["failed_attempts"] == 0

    # The leaderboard should put the candidate ahead of baseline
    rows_json = session.compare_runs()
    rows = json.loads(rows_json)
    assert rows[0]["candidate_id"] != "baseline"
    assert rows[0]["median_ms"] == pytest.approx(5.0)
    assert rows[1]["candidate_id"] == "baseline"
    assert rows[1]["median_ms"] == pytest.approx(10.0)

    # The captured event stream should include the agent + tool events
    after_kinds = sink.kinds()
    assert EventKind.AGENT_THINKING_STARTED.value in after_kinds
    assert EventKind.TOOL_CALL_STARTED.value in after_kinds
    assert EventKind.TOOL_CALL_COMPLETED.value in after_kinds
    assert EventKind.CANDIDATE_PROPOSED.value in after_kinds
    assert EventKind.LEADERBOARD_UPDATED.value in after_kinds
    assert EventKind.RUN_PROGRESS.value in after_kinds

    # Finalise persists an episode
    episode = session.finalize()
    assert episode["successful_count"] == 1
    assert episode["leaderboard"][0]["candidate_id"] != "baseline"


def test_proposing_candidate_with_invalid_intervention_raises(
    tmp_path: Path, _registered_workload
):
    backend_registry.register("fake", _FakeBackend)
    workspace = OptimizationWorkspace(session_cwd=tmp_path)
    session = OptimizationSession(
        workload_id="fake_workload",
        run_id="run-bad",
        workspace=workspace,
        sink=CapturingSink(),
        max_candidates=2,
    )

    bad = json.dumps(
        [{"target_kind": "made_up_kind", "target_selector": "x", "payload": 1}]
    )
    with pytest.raises(ValueError):
        session.propose_candidate(
            interventions_json=bad,
            description="should be rejected",
        )


def test_run_candidates_aggregates_results(tmp_path: Path, _registered_workload):
    backend_registry.register("fake", _FakeBackend)
    workspace = OptimizationWorkspace(session_cwd=tmp_path)
    session = OptimizationSession(
        workload_id="fake_workload",
        run_id="run-batch",
        workspace=workspace,
        sink=CapturingSink(),
        max_candidates=4,
    )

    plans = json.dumps(
        [
            {
                "description": "p1",
                "interventions": [
                    {
                        "target_kind": "knob",
                        "target_selector": "optimize",
                        "payload": "on",
                    }
                ],
            },
            {
                "description": "p2",
                "interventions": [
                    {
                        "target_kind": "knob",
                        "target_selector": "optimize",
                        "payload": "off",
                    }
                ],
            },
        ]
    )
    registered = json.loads(session.propose_candidates(plans_json=plans))
    ids = [c["id"] for c in registered["candidates"]]
    summary = json.loads(session.run_candidates(candidate_ids_json=json.dumps(ids)))
    assert summary["ran"] == 2
    assert summary["successful"] == 2
    assert summary["best"] is not None
