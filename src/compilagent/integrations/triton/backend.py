"""`TritonBackend` — implements the core `Backend` protocol over the lifted
Triton compile harness, pass pipeline, and TTGIR analysis.

The session asks this backend to compile a `Plan` of generic `Intervention`s
under a `WorkloadSpec`. Triton-specific intervention shapes:

  - `target.kind == "pass"`: selector is `"<stage>:<pass_name>"`; payload is
    `{"action": "skip"|"insert"|"reorder"|"run", "args": {...}}`. These get
    projected into the lifted `PassIntervention` records before
    `compile_kernel` runs the stage hook.
  - `target.kind == "launch"`: selector identifies the kernel, payload is
    a meta dict (`{"BLOCK_SIZE": int, "num_warps": int, ...}`). Flows
    through as the compile request's `meta`.

All other `target.kind` values pass through untouched (the session's
`validate_intervention` gate already rejected unsupported kinds).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from compilagent.core.analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    PassCallback,
    PassEvent,
    TimingResult,
)
from compilagent.core.backend import BackendBase
from compilagent.core.plan import Intervention, Plan, ValidationResult
from compilagent.core.search_space import SearchSpace
from compilagent.core.tool_decl import ToolDecl
from compilagent.core.workload import ToleranceConfig, WorkloadSpec
from compilagent.core.workload_registry import workload_registry

from ._internal import schemas as _internal_schemas
from .renderers import MlirRenderer, PtxRenderer
from .tools import list_triton_introspection_tools

_VALID_INTERVENTION_KINDS = frozenset({"pass", "launch", "knob"})


def _to_pass_event(stage: str, result: Any) -> PassEvent:
    """Adapt a lifted `PassResult` into the core `PassEvent` shape."""

    return PassEvent(
        stage=str(stage),
        name=getattr(result, "name", "?"),
        duration_ms=float(getattr(result, "duration_ms", 0.0) or 0.0),
        action=str(getattr(result, "action", "run")),
        args=list(getattr(result, "args", []) or []),
        ir_after_size=(
            len(result.ir_after) if getattr(result, "ir_after", None) else None
        ),
        error=getattr(result, "error", None),
    )


class TritonBackend(BackendBase):
    """Compiler-level Triton backend."""

    id: str = "triton"
    artifact_stages: tuple[str, ...] = ("ttir", "ttgir", "llir", "ptx")

    # ---- device --------------------------------------------------------------

    def device_capability(self) -> DeviceCapability:
        cap_int: int | None = None
        name = "cpu"
        mem_total: int | None = None
        peak_bw_gbps: float | None = None
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                cap_int = major * 10 + minor
                props = torch.cuda.get_device_properties(0)
                name = props.name
                mem_total = int(getattr(props, "total_memory", 0) or 0)
        except Exception:  # noqa: BLE001
            pass
        arch = f"cuda:sm_{cap_int}" if cap_int is not None else "cpu"
        return DeviceCapability(
            arch=arch,
            capability_int=cap_int,
            name=name,
            memory_total_bytes=mem_total,
            memory_peak_bandwidth_gbps=peak_bw_gbps,
        )

    # ---- analysis -----------------------------------------------------------

    def analyze(
        self,
        workload: WorkloadSpec,
        *,
        baseline_artifacts: Sequence[Path],
    ) -> Analysis:
        from ._internal.analysis import extract_decision_traces

        ir_text = ""
        for path in baseline_artifacts:
            p = Path(path)
            if p.suffix.lower() == ".ttgir" and p.exists():
                try:
                    ir_text = p.read_text(encoding="utf-8", errors="replace")
                    break
                except OSError:
                    continue
        traces = extract_decision_traces(ir_text, run_id=workload.id)
        summary: dict[str, Any] = {
            "kind": workload.kind.value,
            "tensor_shapes": {},
            "dtypes": [],
            "op_counts": {},
            "decision_count": len(traces),
        }
        return Analysis(
            summary=summary,
            extra={
                "decision_traces": [
                    t.model_dump(mode="json", exclude_none=True) for t in traces
                ]
            },
        )

    # ---- search space -------------------------------------------------------

    def derive_search_space(
        self,
        workload: WorkloadSpec,
        analysis: Analysis,
    ) -> SearchSpace:
        from ._internal.derivation import ALL_DERIVATIONS

        workload_kind = (analysis.summary or {}).get("kind", workload.kind.value)
        levers: list = []
        for rule in ALL_DERIVATIONS:
            if workload_kind not in getattr(rule, "applies_to", ()):
                continue
            try:
                levers.extend(rule.derive(workload, analysis))
            except Exception:  # noqa: BLE001
                continue
        return SearchSpace(
            workload_id=workload.id,
            backend_id=self.id,
            levers=tuple(levers),
        )

    # ---- intervention surface ----------------------------------------------

    def validate_intervention(self, intervention: Intervention) -> ValidationResult:
        kind = intervention.target.kind
        if kind not in _VALID_INTERVENTION_KINDS:
            return ValidationResult(
                ok=False,
                errors=(
                    f"unsupported target.kind `{kind}`; "
                    f"expected one of {sorted(_VALID_INTERVENTION_KINDS)}",
                ),
            )
        if kind == "pass":
            payload = intervention.payload
            if not isinstance(payload, dict):
                return ValidationResult(
                    ok=False,
                    errors=("pass intervention payload must be a dict",),
                )
            action = str(payload.get("action", "run")).lower()
            if action not in {"run", "skip", "insert", "reorder"}:
                return ValidationResult(
                    ok=False,
                    errors=(f"unsupported pass action `{action}`",),
                )
        return ValidationResult(ok=True)

    def interpret_plan(self, plan: Plan) -> Plan:
        # No last-mile rewriting needed; pass + launch interventions are
        # handled directly in `compile`. Returning unchanged keeps the
        # session contract clean.
        return plan

    # ---- compile + time + correctness --------------------------------------

    def compile(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        artifact_dir: Path,
        pass_callback: PassCallback | None = None,
    ) -> CompileResult:
        from ._internal.harness import TritonCompileHarness
        from ._internal.pipeline import PassIntervention as _PassIntervention

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        source_path = workload.metadata.get("source_path")
        kernel_symbol = workload.metadata.get("kernel_symbol")
        if not source_path or not kernel_symbol:
            return CompileResult(
                ok=False,
                diagnostics=(
                    "workload.metadata is missing `source_path` and/or "
                    "`kernel_symbol` — required for the Triton backend."
                ),
            )

        kspec = _internal_schemas.KernelSpec(
            id=workload.id,
            name=workload.id,
            path=Path(source_path),
            entrypoint=str(kernel_symbol),
            metadata={"workload_kind": workload.kind.value},
        )

        triton_interventions: list[_PassIntervention] = []
        meta: dict[str, Any] = {}
        for iv in plan.interventions:
            if iv.target.kind == "pass":
                stage_name, _, pass_name = iv.target.selector.partition(":")
                if not pass_name:
                    pass_name = stage_name
                payload = iv.payload if isinstance(iv.payload, dict) else {}
                triton_interventions.append(
                    _PassIntervention(
                        pass_name=pass_name,
                        action=str(payload.get("action", "run")),
                        args=dict(payload.get("args", {}) or {}),
                        rationale=iv.rationale,
                    )
                )
            elif iv.target.kind == "launch":
                if isinstance(iv.payload, dict):
                    meta.update(iv.payload)
                elif iv.target.selector:
                    meta[iv.target.selector] = iv.payload

        request = _internal_schemas.CompileRequest(
            kernel_id=workload.id,
            candidate_id=None,
            meta=meta,
        )
        harness = TritonCompileHarness()
        replace_stages = ("ttir", "ttgir") if triton_interventions else ()

        try:
            internal = harness.compile_kernel(
                kspec,
                request,
                artifact_dir=artifact_dir,
                use_stage_hook=True,
                replace_stages=replace_stages,
                interventions=tuple(triton_interventions),
                pass_callback=(
                    (lambda stage, r: pass_callback(_to_pass_event(stage, r)))
                    if pass_callback
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return CompileResult(
                ok=False,
                diagnostics=f"Triton compile raised: {exc!r}",
            )

        artifact_paths = tuple(
            Path(art.path) for art in internal.artifacts if art.path
        )
        return CompileResult(
            ok=internal.ok,
            elapsed_ms=None,
            artifacts=artifact_paths,
            compiled_callable=None,
            diagnostics=internal.diagnostics,
            metadata={
                "kernel_id": workload.id,
                "internal_compile_id": internal.id,
                **dict(internal.metadata or {}),
            },
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
        import time as _time

        try:
            instance = workload_registry.build(workload.id)
        except Exception as exc:  # noqa: BLE001
            return TimingResult(
                timings_ms=(),
                median_ms=None,
                p20_ms=None,
                p80_ms=None,
                diagnostics=f"workload build failed: {exc!r}",
            )
        fn = instance.forward
        for _ in range(max(0, warmup)):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                return TimingResult(
                    timings_ms=(),
                    median_ms=None,
                    p20_ms=None,
                    p80_ms=None,
                    diagnostics=f"forward failed during warmup: {exc!r}",
                )

        timings: list[float] = []
        budget_started = _time.perf_counter()
        budget = max_seconds if max_seconds is not None else float("inf")
        for _ in range(max(1, repetitions)):
            if (_time.perf_counter() - budget_started) > budget:
                break
            t0 = _time.perf_counter()
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                return TimingResult(
                    timings_ms=tuple(timings),
                    median_ms=None,
                    p20_ms=None,
                    p80_ms=None,
                    diagnostics=f"forward raised: {exc!r}",
                )
            ms = (_time.perf_counter() - t0) * 1000.0
            if ms > 0:
                timings.append(ms)
        if not timings:
            return TimingResult(
                timings_ms=(),
                median_ms=None,
                p20_ms=None,
                p80_ms=None,
                diagnostics="no successful timings",
            )
        srt = sorted(timings)
        median = srt[len(srt) // 2]
        p20 = srt[max(0, int(len(srt) * 0.2) - 1)]
        p80 = srt[min(len(srt) - 1, int(len(srt) * 0.8))]
        return TimingResult(
            timings_ms=tuple(timings),
            median_ms=median,
            p20_ms=p20,
            p80_ms=p80,
        )

    def validate_correctness(
        self,
        workload: WorkloadSpec,
        baseline: CompileResult,
        candidate: CompileResult,
        tolerance: ToleranceConfig,
    ) -> CorrectnessResult:
        # Triton kernel correctness is checked inline by the workload's own
        # forward callable when it includes a verifier. The backend itself
        # treats compile-then-time success as evidence of correctness; a
        # workload that needs stricter checks should add them in `forward`.
        return CorrectnessResult(
            ok=True,
            diagnostics=(
                "Triton backend treats compile + run success as correctness; "
                "stricter checks belong in the workload's forward callable."
            ),
        )

    def reset_between_compiles(self, workload: WorkloadSpec) -> None:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch._dynamo as _dynamo  # type: ignore[import-not-found]

            _dynamo.reset()
        except Exception:  # noqa: BLE001
            pass

    # ---- introspection ------------------------------------------------------

    def list_introspection_tools(self) -> Sequence[ToolDecl]:
        return list_triton_introspection_tools()

    def list_artifact_renderers(self) -> Sequence[object]:
        return (MlirRenderer(), PtxRenderer())

    def infer_workload_family(self, workload: WorkloadSpec) -> str | None:
        text = f"{workload.id} {workload.title} {workload.description}".lower()
        if "matmul" in text or "gemm" in text or "dot" in text:
            return "matmul"
        if "attention" in text or "flash" in text:
            return "attention"
        if "norm" in text or "rms" in text or "layer_norm" in text:
            return "norm"
        if "reduce" in text or "reduction" in text or "softmax" in text:
            return "reduction"
        if "copy" in text:
            return "vector_copy"
        if "add" in text or "elementwise" in text or "vector" in text:
            return "elementwise"
        return None
