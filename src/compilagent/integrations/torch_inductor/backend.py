"""`TorchInductorBackend` — implements the core `Backend` protocol on
torch.compile / Inductor.

Translates generic `Intervention`s into an internal `InductorPlan`:

  - `target.kind == "knob"` with selector `dynamo.<leaf>` → `dynamo_config`.
  - `target.kind == "knob"` (any other selector) → `inductor_config`.
  - `target.kind == "lowering"` → `lowering_overrides[selector] = payload`.
  - `target.kind == "fx_node"` → appended to `fx_rewriters`.
  - `target.kind == "scheduler"` with selector `pre_fusion`/`post_fusion`.
  - `target.kind == "choices"` → `choices_handler`.

Other kinds are rejected by `validate_intervention`.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

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

from .renderers import FxGraphRenderer, OutputCodeRenderer, SchedulerLogRenderer
from .tools import list_inductor_introspection_tools

_VALID_INTERVENTION_KINDS = frozenset(
    {"knob", "lowering", "fx_node", "scheduler", "choices"}
)


@contextmanager
def _scratch_dir():
    p = Path(tempfile.mkdtemp(prefix="compilagent-inductor-"))
    try:
        yield p
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(p, ignore_errors=True)


class TorchInductorBackend(BackendBase):
    """`torch.compile` driven by a custom Dynamo backend."""

    id: str = "torch_inductor"
    artifact_stages: tuple[str, ...] = (
        "fx_graph",
        "output_code",
        "schedule_log",
        "fusion_log",
    )

    # ---- device --------------------------------------------------------------

    def device_capability(self) -> DeviceCapability:
        cap_int: int | None = None
        name = "cpu"
        mem_total: int | None = None
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
            memory_peak_bandwidth_gbps=None,
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
        if kind == "scheduler" and intervention.target.selector not in {
            "pre_fusion",
            "post_fusion",
        }:
            return ValidationResult(
                ok=False,
                errors=(
                    "scheduler intervention selector must be "
                    "`pre_fusion` or `post_fusion`",
                ),
            )
        return ValidationResult(ok=True)

    def interpret_plan(self, plan: Plan) -> Plan:
        return plan

    def _project_plan(self, plan: Plan):
        from ._internal.harness import InductorPlan

        ip = InductorPlan()
        for iv in plan.interventions:
            kind = iv.target.kind
            if kind == "knob":
                if iv.target.selector.startswith("dynamo."):
                    ip.dynamo_config[iv.target.selector] = iv.payload
                else:
                    ip.inductor_config[iv.target.selector] = iv.payload
            elif kind == "lowering":
                ip.lowering_overrides[iv.target.selector] = iv.payload
            elif kind == "fx_node":
                ip.fx_rewriters.append(iv.payload)
            elif kind == "scheduler":
                if iv.target.selector == "pre_fusion":
                    ip.pre_fusion_pass = iv.payload
                elif iv.target.selector == "post_fusion":
                    ip.post_fusion_pass = iv.payload
            elif kind == "choices":
                ip.choices_handler = iv.payload
        return ip

    # ---- compile / time / correctness --------------------------------------

    def compile(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        artifact_dir: Path,
        pass_callback: PassCallback | None = None,
    ) -> CompileResult:
        from ._internal.harness import drive_compile

        instance = workload_registry.build(workload.id)
        try:
            outcome = drive_compile(
                instance, self._project_plan(plan), artifact_dir=artifact_dir
            )
        except Exception as exc:  # noqa: BLE001
            return CompileResult(
                ok=False, diagnostics=f"Inductor compile raised: {exc!r}"
            )

        if pass_callback is not None:
            with contextlib.suppress(Exception):
                pass_callback(
                    PassEvent(
                        stage="inductor",
                        name="compile_fx",
                        duration_ms=float(outcome.elapsed_ms or 0.0),
                        action="run",
                        args=[],
                        error=outcome.diagnostics if not outcome.ok else None,
                    )
                )

        # Collect every captured path into `artifacts`. The session never
        # branches on backend identity to extract them.
        artifacts: list[Path] = []
        for attr in ("fx_graph_path", "output_code_path", "schedule_log_path"):
            p = getattr(outcome, attr, None)
            if p:
                artifacts.append(Path(p))
        for p in (outcome.captured_logs or {}).values():
            if p:
                artifacts.append(Path(p))

        return CompileResult(
            ok=outcome.ok,
            elapsed_ms=outcome.elapsed_ms,
            artifacts=tuple(artifacts),
            compiled_callable=outcome.compiled_callable,
            diagnostics=outcome.diagnostics,
            warnings=tuple(getattr(outcome, "warnings", []) or []),
            metadata={"workload_id": workload.id},
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
        from ._internal.harness import drive_compile

        instance = workload_registry.build(workload.id)
        with _scratch_dir() as tmp:
            outcome = drive_compile(
                instance, self._project_plan(plan), artifact_dir=tmp
            )
        if not outcome.ok or outcome.compiled_callable is None:
            return TimingResult(
                timings_ms=(),
                median_ms=None,
                p20_ms=None,
                p80_ms=None,
                diagnostics=outcome.diagnostics or "compile failed",
            )
        compiled = outcome.compiled_callable
        ex = instance.example_inputs

        def _run():
            return compiled(*ex) if ex else compiled()

        for _ in range(max(0, warmup)):
            try:
                _run()
            except Exception:  # noqa: BLE001
                break

        timings: list[float] = []
        budget = max_seconds if max_seconds is not None else float("inf")
        started = time.perf_counter()
        for _ in range(max(1, repetitions)):
            if (time.perf_counter() - started) > budget:
                break
            t0 = time.perf_counter()
            try:
                _run()
            except Exception as exc:  # noqa: BLE001
                return TimingResult(
                    timings_ms=tuple(timings),
                    median_ms=None,
                    p20_ms=None,
                    p80_ms=None,
                    diagnostics=f"forward raised: {exc!r}",
                )
            ms = (time.perf_counter() - t0) * 1000.0
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
        from ._internal.correctness import compare_forward

        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            return CorrectnessResult(
                ok=False, diagnostics=f"torch import failed: {exc!r}"
            )
        if baseline.compiled_callable is None or candidate.compiled_callable is None:
            return CorrectnessResult(
                ok=False, diagnostics="missing compiled callable"
            )
        instance = workload_registry.build(workload.id)
        ex = instance.example_inputs
        seed = int(workload.metadata.get("seed", 0))

        def _run(fn):
            torch.manual_seed(seed)
            return fn(*ex) if ex else fn()

        try:
            b_out = _run(baseline.compiled_callable)
            c_out = _run(candidate.compiled_callable)
        except Exception as exc:  # noqa: BLE001
            return CorrectnessResult(
                ok=False, diagnostics=f"correctness run raised: {exc!r}"
            )
        return compare_forward(
            baseline_run=b_out, candidate_run=c_out, tolerance=tolerance
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

    # ---- analysis / search-space --------------------------------------------

    def analyze(
        self,
        workload: WorkloadSpec,
        *,
        baseline_artifacts: Sequence[Path],
    ) -> Analysis:
        from ._internal.analysis import parse_inductor_artifacts, to_generic_analysis

        def _find(suffix: str) -> Path | None:
            for p in baseline_artifacts:
                if str(p).endswith(suffix):
                    return Path(p)
            return None

        output_code = _find("output_code.py") or _find("output_code.log")
        schedule = _find("schedule.log")
        fx_graph = _find("fx_graph.py")
        data = parse_inductor_artifacts(
            output_code=output_code, schedule_log=schedule, fx_graph=fx_graph
        )
        analysis = to_generic_analysis(
            data, workload_kind=workload.kind.value
        )
        if fx_graph and fx_graph.exists():
            try:
                analysis.extra["fx_text"] = fx_graph.read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception:  # noqa: BLE001
                analysis.extra["fx_text"] = ""
        return analysis

    def derive_search_space(
        self,
        workload: WorkloadSpec,
        analysis: Analysis,
    ) -> SearchSpace:
        from ._internal.derivation import ALL_DERIVATIONS

        kind = (analysis.summary or {}).get("kind", workload.kind.value)
        levers: list = []
        for rule in ALL_DERIVATIONS:
            if kind not in getattr(rule, "applies_to", ()):
                continue
            try:
                levers.extend(rule.derive(workload, analysis))
            except Exception:  # noqa: BLE001
                continue
        return SearchSpace(
            workload_id=workload.id, backend_id=self.id, levers=tuple(levers)
        )

    # ---- introspection ------------------------------------------------------

    def list_introspection_tools(self) -> Sequence[ToolDecl]:
        return list_inductor_introspection_tools()

    def list_artifact_renderers(self) -> Sequence[object]:
        return (FxGraphRenderer(), OutputCodeRenderer(), SchedulerLogRenderer())

    def infer_workload_family(self, workload: WorkloadSpec) -> str | None:
        text = f"{workload.id} {workload.title} {workload.description}".lower()
        if "vit" in text or "transformer" in text:
            return "transformer"
        if "matmul" in text or "gemm" in text:
            return "matmul"
        if "attention" in text or "flash" in text:
            return "attention"
        if "norm" in text or "rms" in text:
            return "norm"
        if "conv" in text:
            return "conv"
        return None
