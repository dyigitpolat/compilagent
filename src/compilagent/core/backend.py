"""Abstract `Backend` protocol shared by every compiler integration.

The session, the toolset, the experiment log, and the observation UI all
program against this interface. Concrete implementations live under
`compilagent.integrations.<name>` (in-tree) or external packages (out-of-tree)
and self-register via `backend_registry.register(id, factory)`.

A backend is responsible for:

  - Describing its target device.
  - Declaring its artifact stages (e.g. ("ttir","ttgir","llir","ptx") for
    Triton, ("fx_graph","output_code","schedule_log","fusion_log") for
    Inductor) so the observation UI can label them generically.
  - Compiling a `WorkloadSpec` under a `Plan` of `Intervention`s.
  - Timing a workload's end-to-end execution.
  - Validating numerical correctness against a reference compile.
  - Deriving a `SearchSpace` of typed `Lever`s from analysis — no
    hand-coded values.
  - Validating and interpreting a `Plan` (so domain-specific rewriting like
    Triton's pass split lives here, not in the session).
  - Listing extra agent tools and artifact renderers it wants registered.

Backend-specific concepts (MLIR passes, inductor knobs, FX rewrites) are
encapsulated in the `Plan`/`Intervention` payloads; the session never
inspects them.

Out-of-tree integrations: implement the `Backend` protocol structurally
(no inheritance required) or inherit from `BackendBase` to skip the
no-op-defaultable methods. See `docs/integration_guide.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    PassCallback,
    TimingResult,
)
from .plan import Intervention, Plan, ValidationResult
from .search_space import SearchSpace
from .tool_decl import ToolDecl
from .workload import ToleranceConfig, WorkloadSpec


@runtime_checkable
class Backend(Protocol):
    """Abstract compiler / optimizer backend.

    Every method below is a contract the core session relies on. Empty / no-op
    returns are explicitly allowed where noted; `BackendBase` provides safe
    defaults for the optional methods.
    """

    id: str
    """Stable string id this backend registers under (e.g. `"triton"`)."""

    artifact_stages: tuple[str, ...]
    """Ordered names of compile stages this backend produces artifacts for.

    Used by the observation UI to label per-stage tabs. Order is
    conventional (low IR → high IR / source → asm); the UI does not rely on
    it. Empty tuple is allowed for backends that do not surface staged IR.
    """

    def device_capability(self) -> DeviceCapability:
        """Describe the hardware this backend will compile against.

        Called once during session bootstrap; used to label the run, shown
        in the UI, and persisted on the episode artifact.
        """
        ...

    def analyze(
        self,
        workload: WorkloadSpec,
        *,
        baseline_artifacts: Sequence[Path],
    ) -> Analysis:
        """Inspect the baseline compile and return structured `Analysis`.

        `summary` MUST contain a `kind` key whose value matches
        `workload.kind.value`. Recommended (advisory) keys: `tensor_shapes`,
        `dtypes`, `op_counts`. `extra` is opaque and may carry any
        backend-specific metadata for later inspection.
        """
        ...

    def derive_search_space(
        self,
        workload: WorkloadSpec,
        analysis: Analysis,
    ) -> SearchSpace:
        """Produce the catalog of `Lever`s the agent can pull.

        Lever ranges MUST be derived from `workload` / `analysis` /
        `device_capability()` — never hand-coded. An empty `SearchSpace` is
        valid for backends with nothing to expose to the agent.
        """
        ...

    def validate_intervention(self, intervention: Intervention) -> ValidationResult:
        """Reject malformed interventions before the session registers them.

        Errors should be human-readable strings; the session turns them into
        a `ValueError` so the harness surfaces them as a model retry. Return
        `ValidationResult(ok=True)` for any intervention this backend
        accepts; the agent learns the vocabulary of valid `target.kind`
        values via `inspect_search_space`.
        """
        ...

    def interpret_plan(self, plan: Plan) -> Plan:
        """Last-mile rewrite of a `Plan` immediately before `compile`.

        The session never inspects the result; backends that need to split
        compound interventions (e.g. Triton's pass-and-launch split) do that
        here. A no-op `return plan` is the common case.
        """
        ...

    def apply_intervention(self, plan: Plan, intervention: Intervention) -> Plan:
        """Append (or merge) an intervention into a plan.

        Default semantics: append `intervention` to `plan.interventions`.
        Backends that want to dedupe or fuse compatible interventions can
        override.
        """
        ...

    def compile(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        artifact_dir: Path,
        pass_callback: PassCallback | None = None,
    ) -> CompileResult:
        """Compile `workload` under `plan`, dumping artifacts under `artifact_dir`.

        `pass_callback` (when provided) is called once per executed compile
        step with a `PassEvent`; backends that don't have a notion of
        per-pass execution may ignore it. `CompileResult.artifacts` is the
        only path collection the session reads; per-stage paths must be
        included there. Backend-specific extras go in `metadata`.

        On compile failure return `CompileResult(ok=False, ...)` with
        `diagnostics` populated. Do NOT raise — the session handles the
        budget bookkeeping for failed compiles.
        """
        ...

    def time_workload(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        warmup: int,
        repetitions: int,
        max_seconds: float | None = None,
    ) -> TimingResult:
        """Time the compiled workload end-to-end.

        Called by the session after `compile` succeeds; use the same `plan`
        the session passed to `compile` to drive the cached compile. Return
        a `TimingResult` populated with per-rep timings and the median.
        """
        ...

    def validate_correctness(
        self,
        workload: WorkloadSpec,
        baseline: CompileResult,
        candidate: CompileResult,
        tolerance: ToleranceConfig,
    ) -> CorrectnessResult:
        """Verify candidate output matches baseline within `tolerance`.

        Called only after both compiles succeeded. Backends that cannot run
        a numerical check (e.g. compile-only smoke tests) may return
        `CorrectnessResult(ok=True, diagnostics="not checked")`.
        """
        ...

    def reset_between_compiles(self, workload: WorkloadSpec) -> None:
        """Clear backend-side state between candidate compiles.

        Optional. PyTorch backends typically call `torch._dynamo.reset()` /
        `torch.cuda.empty_cache()` here; pure-Python or stateless backends
        return without doing anything. Exceptions are swallowed by the
        session and surfaced as warnings.
        """
        ...

    def list_introspection_tools(self) -> Sequence[ToolDecl]:
        """Return any extra agent tools this backend wants exposed.

        Each tool is appended to the session's canonical 8-tool toolset via
        `Toolset.with_extra(...)`. Names must not collide with the canonical
        tools. An empty sequence is valid.
        """
        ...

    def list_artifact_renderers(self) -> Sequence[object]:
        """Return `ArtifactRenderer` instances for backend-specific suffixes.

        Typed loosely to avoid an import cycle with
        `compilagent.observation.artifacts`; the registry validates the
        protocol at registration time. An empty sequence is valid.
        """
        ...

    def infer_workload_family(self, workload: WorkloadSpec) -> str | None:
        """Best-effort family label used by `CandidatePolicy` to correlate runs.

        Return a short string (`"matmul"`, `"reduction"`, …) that lets the
        cross-run memory match this workload to prior outcomes on similar
        ones, or `None` if the backend has no useful grouping.
        """
        ...


class BackendBase:
    """Concrete base class with safe defaults for the optional `Backend` methods.

    Inheriting from `BackendBase` lets an integration only implement the
    methods that actually carry semantics for its compiler. The protocol
    itself is structurally typed, so inheriting from `BackendBase` is *not*
    required — it's a convenience.

    Methods left to the integration to implement:
      - `device_capability`
      - `analyze`
      - `derive_search_space`
      - `validate_intervention`
      - `compile`
      - `time_workload`
      - `validate_correctness`

    Defaults provided here:
      - `interpret_plan` — return plan unchanged.
      - `apply_intervention` — append intervention.
      - `reset_between_compiles` — no-op.
      - `list_introspection_tools` — empty tuple.
      - `list_artifact_renderers` — empty tuple.
      - `infer_workload_family` — `None`.
    """

    id: str = "base"
    artifact_stages: tuple[str, ...] = ()

    def interpret_plan(self, plan: Plan) -> Plan:
        return plan

    def apply_intervention(self, plan: Plan, intervention: Intervention) -> Plan:
        return Plan(interventions=plan.interventions + (intervention,))

    def reset_between_compiles(self, workload: WorkloadSpec) -> None:
        return None

    def list_introspection_tools(self) -> Sequence[ToolDecl]:
        return ()

    def list_artifact_renderers(self) -> Sequence[object]:
        return ()

    def infer_workload_family(self, workload: WorkloadSpec) -> str | None:
        return None


class BackendRegistry:
    """Process-wide map from backend id to a zero-arg factory.

    `register` accepts any callable returning a `Backend`; the most common
    pattern is to pass the class itself (acts as its own zero-arg factory
    when no constructor args are needed).
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Backend]] = {}

    def register(self, backend_id: str, factory: Callable[[], Backend]) -> None:
        if backend_id in self._factories:
            raise ValueError(f"Backend `{backend_id}` is already registered.")
        self._factories[backend_id] = factory

    def get(self, backend_id: str) -> Backend:
        if backend_id not in self._factories:
            known = sorted(self._factories.keys())
            raise KeyError(
                f"Unknown backend `{backend_id}`. Registered: {known or '(none)'}."
            )
        return self._factories[backend_id]()

    def ids(self) -> list[str]:
        return sorted(self._factories.keys())

    def clear(self) -> None:
        self._factories.clear()


backend_registry = BackendRegistry()
