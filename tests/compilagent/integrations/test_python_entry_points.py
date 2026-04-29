"""Smoke tests for `optimize_module` / `optimize_kernel`.

We don't actually compile through Triton or Inductor here — the harness
they reach is the registered fake one — but we exercise the workload-spec
construction, the registry validation, and the dispatch into
`OptimizationSession`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest

from compilagent.core.analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    TimingResult,
)
from compilagent.core.backend import backend_registry
from compilagent.core.plan import Plan, ValidationResult
from compilagent.core.search_space import (
    SearchSpace,
)
from compilagent.core.tool_decl import ToolDecl
from compilagent.harness.base import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
)
from compilagent.harness.registry import harness_registry


@dataclass
class _NopBackend:
    id: str = "fake"
    artifact_stages: tuple[str, ...] = ()

    def device_capability(self):
        return DeviceCapability(arch="cpu", capability_int=None, name="Fake",
                                memory_total_bytes=None, memory_peak_bandwidth_gbps=None)

    def analyze(self, workload, *, baseline_artifacts):
        return Analysis(summary={"kind": workload.kind.value})

    def derive_search_space(self, workload, analysis):
        return SearchSpace(workload_id=workload.id, backend_id=self.id, levers=())

    def validate_intervention(self, intervention):
        return ValidationResult(ok=True)

    def interpret_plan(self, plan):
        return plan

    def apply_intervention(self, plan, intervention):
        return Plan(interventions=plan.interventions + (intervention,))

    def compile(self, workload, plan, *, artifact_dir, pass_callback=None):
        return CompileResult(ok=True, elapsed_ms=1.0)

    def time_workload(self, workload, plan, *, warmup, repetitions, max_seconds=None):
        return TimingResult(timings_ms=(5.0,) * repetitions, median_ms=5.0,
                            p20_ms=5.0, p80_ms=5.0)

    def validate_correctness(self, workload, baseline, candidate, tolerance):
        return CorrectnessResult(ok=True)

    def reset_between_compiles(self, workload):
        return None

    def list_introspection_tools(self) -> Sequence[ToolDecl]:
        return ()

    def list_artifact_renderers(self) -> Sequence[object]:
        return ()

    def infer_workload_family(self, workload):
        return None


@dataclass
class _NoOpHarness:
    id: str = "nop"
    supported_providers: tuple[str, ...] = ("fake",)

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind=StreamEventKind.RUN_FINISHED, text="ok")


def test_optimize_module_dispatches_through_registries(tmp_path):
    from compilagent.integrations.python import optimize_module
    from compilagent.settings import CompilagentSettings
    from compilagent.storage.workspace import OptimizationWorkspace

    backend_registry.register("fake", _NopBackend)
    harness_registry.register("nop", _NoOpHarness)

    class _Model:
        def __call__(self, x):
            return x

    settings = CompilagentSettings(
        model_name="test",
        harness="nop",
        max_candidates=2,
        max_benchmark_seconds=1,
    )
    workspace = OptimizationWorkspace(session_cwd=tmp_path)
    result = optimize_module(
        _Model(),
        example_inputs=(0,),
        backend_id="fake",
        max_candidates=2,
        settings=settings,
        workspace=workspace,
    )
    assert result.workload_id.startswith("py_module_")
    assert result.backend_id == "fake"
    assert result.harness == "nop"
    assert result.baseline_median_ms == 5.0


def test_optimize_module_unknown_backend_raises(tmp_path):
    from compilagent.integrations.python import optimize_module
    from compilagent.settings import CompilagentSettings
    from compilagent.storage.workspace import OptimizationWorkspace

    settings = CompilagentSettings(model_name="test", harness="nop")
    with pytest.raises(KeyError, match="Unknown backend"):
        optimize_module(
            object(),
            example_inputs=(),
            backend_id="does_not_exist",
            settings=settings,
            workspace=OptimizationWorkspace(session_cwd=tmp_path),
        )
