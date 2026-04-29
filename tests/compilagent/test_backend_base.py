"""`BackendBase` provides safe defaults so integrations only implement what matters."""

from __future__ import annotations

from compilagent.core.analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    TimingResult,
)
from compilagent.core.backend import Backend, BackendBase
from compilagent.core.plan import Intervention, Plan, Target, ValidationResult
from compilagent.core.search_space import SearchSpace
from compilagent.core.workload import (
    WorkloadKind,
    WorkloadSpec,
)


class _MinimalBackend(BackendBase):
    id = "minimal"
    artifact_stages: tuple[str, ...] = ("ir",)

    def device_capability(self) -> DeviceCapability:
        return DeviceCapability(
            arch="cpu",
            capability_int=None,
            name="Minimal",
            memory_total_bytes=None,
            memory_peak_bandwidth_gbps=None,
        )

    def analyze(self, workload, *, baseline_artifacts):
        return Analysis(summary={"kind": workload.kind.value})

    def derive_search_space(self, workload, analysis):
        return SearchSpace(workload_id=workload.id, backend_id=self.id, levers=())

    def validate_intervention(self, intervention):
        return ValidationResult(ok=True)

    def compile(self, workload, plan, *, artifact_dir, pass_callback=None):
        return CompileResult(ok=True, elapsed_ms=0.0)

    def time_workload(self, workload, plan, *, warmup, repetitions, max_seconds=None):
        return TimingResult(timings_ms=(1.0,), median_ms=1.0, p20_ms=1.0, p80_ms=1.0)

    def validate_correctness(self, workload, baseline, candidate, tolerance):
        return CorrectnessResult(ok=True)


def test_minimal_backend_satisfies_protocol():
    backend = _MinimalBackend()
    assert isinstance(backend, Backend)


def test_default_interpret_plan_is_identity():
    backend = _MinimalBackend()
    plan = Plan(interventions=(Intervention(target=Target(kind="x"), payload=1),))
    assert backend.interpret_plan(plan) is plan


def test_default_apply_intervention_appends():
    backend = _MinimalBackend()
    plan = Plan(interventions=())
    iv = Intervention(target=Target(kind="x"), payload=1)
    extended = backend.apply_intervention(plan, iv)
    assert extended.interventions == (iv,)


def test_default_optional_methods_return_safe_zeros():
    backend = _MinimalBackend()
    spec = WorkloadSpec(
        id="w", title="W", description="d", kind=WorkloadKind.KERNEL, backend_id="minimal"
    )
    assert backend.reset_between_compiles(spec) is None
    assert tuple(backend.list_introspection_tools()) == ()
    assert tuple(backend.list_artifact_renderers()) == ()
    assert backend.infer_workload_family(spec) is None
