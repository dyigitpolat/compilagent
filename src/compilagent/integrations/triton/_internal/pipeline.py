"""Pass-by-pass execution of Triton's compilation pipeline.

The default Triton NVIDIA backend builds its TTGIR pipeline by registering
all passes on a single `pass_manager` and calling `pm.run(mod)`. That gives us
no visibility into what each pass did, no way to swap pass parameters per
candidate, and no way to skip / reorder passes.

This module rebuilds the pipeline as a sequence of named, single-pass
`pass_manager.run` invocations. Between passes we capture the IR text via
`mod.str()`. A `PipelinePlan` describes per-pass overrides (skip, replace,
parameterize, insert custom rewrite). The default plan mirrors the upstream
backend exactly so swapping in our pipeline is a no-op unless the agent
applies overrides.

Stage replacement is the integration point: when `replace=True`, the
`scoped_stage_hook` in `stages.py` substitutes the backend's `make_ttir` /
`make_ttgir` lambdas with `StagePipeline.run_*`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from triton._C.libtriton import ir  # type: ignore[import-not-found]

from .passes import PassDescriptor, callable_for, get_pass

PassAction = Literal["run", "skip", "replace"]


@dataclass(slots=True)
class PassIntervention:
    """Per-pass override applied when executing a pipeline plan."""

    pass_name: str
    action: PassAction = "run"
    args: dict[str, Any] = field(default_factory=dict)
    """Override args by parameter name (matched against PassDescriptor.params)."""
    rationale: str = ""
    ir_rewriter: Callable[[str], str] | None = None
    """Optional callable: takes the IR text **before** this pass runs, returns
    the rewritten IR text. The pipeline re-parses the result and feeds it to
    the pass. Lets the agent inject layout / encoding rewrites that no built-in
    pass produces. Skipped if `action == "skip"`."""

    def model_dump(self) -> dict[str, Any]:
        return {
            "pass_name": self.pass_name,
            "action": self.action,
            "args": dict(self.args),
            "rationale": self.rationale,
            "has_ir_rewriter": self.ir_rewriter is not None,
        }


@dataclass(slots=True)
class PipelinePlanStep:
    """Single step in the executable plan: a (pass, default_args) entry."""

    descriptor: PassDescriptor
    args: tuple[Any, ...]
    """Positional args appended after pass_manager (mirrors descriptor.params order)."""

    def model_dump(self) -> dict[str, Any]:
        return {
            "name": self.descriptor.name,
            "origin": self.descriptor.origin,
            "args": list(self.args),
        }


@dataclass(slots=True)
class PipelinePlan:
    """Executable plan = ordered steps + agent interventions keyed by pass name."""

    stage: Literal["ttir", "ttgir"]
    capability: int
    steps: list[PipelinePlanStep]
    interventions: dict[str, PassIntervention] = field(default_factory=dict)

    def with_intervention(self, intervention: PassIntervention) -> PipelinePlan:
        out = dict(self.interventions)
        out[intervention.pass_name] = intervention
        return PipelinePlan(
            stage=self.stage,
            capability=self.capability,
            steps=list(self.steps),
            interventions=out,
        )

    def reorder(self, names: list[str]) -> PipelinePlan:
        """Return a new plan whose steps are reordered to match `names`.

        Steps not in `names` are dropped (or kept at the tail if `names` is a
        prefix). Unknown names in `names` are ignored. The result is a fresh
        plan with the same interventions; pass-args are preserved.
        """

        by_name: dict[str, PipelinePlanStep] = {}
        for step in self.steps:
            by_name.setdefault(step.descriptor.name, step)
        new_steps: list[PipelinePlanStep] = []
        seen: set[str] = set()
        for name in names:
            if name in by_name and name not in seen:
                new_steps.append(by_name[name])
                seen.add(name)
        # If the caller supplied a prefix, append the rest in original order.
        if seen and len(seen) < len(by_name):
            for step in self.steps:
                if step.descriptor.name not in seen:
                    new_steps.append(step)
        return PipelinePlan(
            stage=self.stage,
            capability=self.capability,
            steps=new_steps or list(self.steps),
            interventions=dict(self.interventions),
        )

    def names(self) -> list[str]:
        return [step.descriptor.name for step in self.steps]

    def model_dump(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "capability": self.capability,
            "steps": [step.model_dump() for step in self.steps],
            "interventions": {k: v.model_dump() for k, v in self.interventions.items()},
        }


@dataclass(slots=True)
class PassResult:
    """Result of running one pass."""

    name: str
    action: PassAction
    duration_ms: float
    args: list[Any]
    ir_after: str | None = None
    error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action": self.action,
            "duration_ms": self.duration_ms,
            "args": list(self.args),
            "ir_after_size": (len(self.ir_after) if self.ir_after is not None else None),
            "error": self.error,
        }


@dataclass(slots=True)
class StageResult:
    """Result of running a full stage pipeline."""

    stage: Literal["ttir", "ttgir"]
    capability: int
    ir_before: str
    ir_after: str
    passes: list[PassResult]

    @property
    def total_ms(self) -> float:
        return sum(p.duration_ms for p in self.passes)

    def model_dump(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "capability": self.capability,
            "ir_before_size": len(self.ir_before),
            "ir_after_size": len(self.ir_after),
            "total_ms": self.total_ms,
            "passes": [p.model_dump() for p in self.passes],
        }


# --- default plan builders ----------------------------------------------------


def build_ttir_plan(
    *,
    capability: int,
    num_warps: int,
    num_stages: int,
    num_ctas: int,
) -> PipelinePlan:
    del num_warps, num_stages, num_ctas  # unused for ttir
    steps: list[PipelinePlanStep] = []

    def step(name: str, *args: Any) -> None:
        steps.append(PipelinePlanStep(get_pass(name), tuple(args)))

    step("inliner")
    step("triton-rewrite-tensor-pointer")
    if capability // 10 < 9:
        step("triton-rewrite-tensor-descriptor-to-pointer")
    step("canonicalizer")
    step("triton-combine")
    step("triton-reorder-broadcast")
    step("cse")
    step("symbol-dce")
    step("triton-loop-unroll")

    return PipelinePlan(stage="ttir", capability=capability, steps=steps)


def build_ttgir_plan(
    *,
    capability: int,
    num_warps: int,
    num_stages: int,
    num_ctas: int,
    threads_per_warp: int = 32,
    dump_enabled: bool = False,
) -> PipelinePlan:
    """Mirror CUDABackend.make_ttgir as a sequence of single-pass steps."""

    steps: list[PipelinePlanStep] = []

    def step(name: str, *args: Any) -> None:
        steps.append(PipelinePlanStep(get_pass(name), tuple(args)))

    emu_tf32 = capability // 10 >= 8
    target = f"cuda:{capability}"

    step("convert-triton-to-tritongpu", target, num_warps, threads_per_warp, num_ctas)
    step("tritongpu-coalesce")
    step("tritongpu-f32-dot-tc", emu_tf32)
    step("ttng-plan-cta")
    step("tritongpu-remove-layout-conversions")
    step("tritongpu-optimize-thread-locality")
    step("tritongpu-accelerate-matmul")
    step("tritongpu-remove-layout-conversions")
    step("tritongpu-optimize-dot-operands", capability >= 80)
    step("ttng-optimize-descriptor-encoding")
    step("triton-loop-aware-cse")

    if capability // 10 in {8, 9}:
        step("tritongpu-fuse-nested-loops")
        step("canonicalizer")
        step("triton-licm")
        step("canonicalizer")
        step("tritongpu-combine-tensor-select-and-if")
        # add_hopper_warpspec is not in the catalog (Hopper-only); call inline
        # via raw passes module if needed.
        step("tritongpu-assign-latencies", num_stages)
        step("tritongpu-schedule-loops")
        step("tritongpu-pipeline", num_stages, dump_enabled)
    elif capability // 10 >= 10:
        step("tritongpu-fuse-nested-loops")
        step("canonicalizer")
        step("triton-licm")
        step("tritongpu-optimize-accumulator-init")
        step("tritongpu-hoist-tmem-alloc", False)
        step("ttng-promote-lhs-to-tmem")
        step("tritongpu-assign-latencies", num_stages)
        step("tritongpu-schedule-loops")
        step("tritongpu-warp-specialize", num_stages)
        step("tritongpu-pipeline", num_stages, dump_enabled)
        step("tritongpu-optimize-partition-warps")
        step("tritongpu-combine-tensor-select-and-if")
        step("tritongpu-hoist-tmem-alloc", True)
        step("ttng-remove-tmem-tokens")
    else:
        step("triton-licm")

    step("canonicalizer")
    step("triton-loop-aware-cse")
    step("tritongpu-prefetch")
    step("tritongpu-optimize-dot-operands", capability >= 80)
    step("tritongpu-coalesce-async-copy")
    if capability // 10 >= 10:
        step("ttng-optimize-tmem-layouts")
    if capability // 10 >= 9:
        step("ttng-tma-lowering")
    step("tritongpu-remove-layout-conversions")
    if capability // 10 >= 10:
        step("ttng-interleave-tmem")
    step("tritongpu-reduce-data-duplication")
    step("tritongpu-reorder-instructions")
    step("triton-loop-aware-cse")
    step("symbol-dce")
    step("ttng-fence-insertion", capability)
    step("ttng-lower-mma")
    step("sccp")
    step("cse")
    step("canonicalizer")

    return PipelinePlan(stage="ttgir", capability=capability, steps=steps)


# --- executor -----------------------------------------------------------------


@dataclass(slots=True)
class ExecutorOptions:
    """Tunables for stage execution."""

    capture_ir: bool = True
    capture_only_changed: bool = True
    """If True, only capture IR when it differs from the previous snapshot."""
    dump_dir: Path | None = None
    """If set, also write each captured IR to a file."""


def execute_plan(
    mod: Any,
    plan: PipelinePlan,
    *,
    options: ExecutorOptions | None = None,
    on_pass: Callable[[PassResult], None] | None = None,
) -> StageResult:
    """Run a pipeline plan pass-by-pass.

    `mod` is a Triton `ir.module` (mutated in place). The function returns a
    `StageResult` containing the IR before, the IR after, and a `PassResult`
    per step (with optional captured IR-after snapshots).
    """

    options = options or ExecutorOptions()
    ir_before = mod.str()
    last_ir = ir_before
    results: list[PassResult] = []

    for step in plan.steps:
        intervention = plan.interventions.get(step.descriptor.name)
        action: PassAction = intervention.action if intervention else "run"

        # parameter overrides
        args = list(step.args)
        if intervention and intervention.args:
            args = _merge_args(step.descriptor, args, intervention.args)

        if action == "skip":
            results.append(
                PassResult(
                    name=step.descriptor.name,
                    action="skip",
                    duration_ms=0.0,
                    args=args,
                    ir_after=None,
                )
            )
            if on_pass is not None:
                on_pass(results[-1])
            continue

        callable_obj = callable_for(step.descriptor)
        pm = ir.pass_manager(mod.context)
        if options.capture_ir:
            pm.enable_debug()

        # Inter-pass IR rewriter: if the intervention supplies an `ir_rewriter`,
        # apply it to the IR text *before* the pass runs. The rewritten text is
        # parsed back into a fresh module that replaces `mod` for this pass and
        # all subsequent ones. Errors here are reported but don't abort the run.
        rewriter_error: str | None = None
        if intervention is not None and intervention.ir_rewriter is not None and action != "skip":
            try:
                pre_text = mod.str()
                new_text = intervention.ir_rewriter(pre_text)
                if new_text and new_text != pre_text:
                    new_mod = ir.parse_mlir_module(new_text, mod.context)
                    if new_mod is not None:
                        # Triton's pass manager mutates whatever we pass in.
                        # Replace the in-place reference for the remainder of
                        # the pipeline by pointing `mod` at the new module.
                        mod = new_mod
                        last_ir = new_text
            except Exception as exc:  # noqa: BLE001
                rewriter_error = f"ir_rewriter: {type(exc).__name__}: {exc}"

        try:
            callable_obj(pm, *args)
        except TypeError as exc:
            raise TypeError(
                f"pass `{step.descriptor.name}` rejected args {args}: {exc}"
            ) from exc

        started = time.perf_counter()
        error: str | None = rewriter_error
        try:
            pm.run(mod, f"compilagent::{step.descriptor.name}")
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}" if not error else f"{error}; pass: {exc}"
        duration_ms = (time.perf_counter() - started) * 1000.0

        ir_after: str | None = None
        if options.capture_ir and error is None:
            current = mod.str()
            if not options.capture_only_changed or current != last_ir:
                ir_after = current
                last_ir = current
            if options.dump_dir is not None and ir_after is not None:
                options.dump_dir.mkdir(parents=True, exist_ok=True)
                idx = len(results)
                fname = f"{idx:03d}_{step.descriptor.name}.{plan.stage}"
                (options.dump_dir / fname).write_text(ir_after, encoding="utf-8")

        result = PassResult(
            name=step.descriptor.name,
            action=action,
            duration_ms=duration_ms,
            args=args,
            ir_after=ir_after,
            error=error,
        )
        results.append(result)
        if on_pass is not None:
            on_pass(result)
        if error is not None:
            break

    ir_after_total = mod.str()
    return StageResult(
        stage=plan.stage,
        capability=plan.capability,
        ir_before=ir_before,
        ir_after=ir_after_total,
        passes=results,
    )


def _merge_args(
    descriptor: PassDescriptor,
    base_args: list[Any],
    overrides: dict[str, Any],
) -> list[Any]:
    if not descriptor.params:
        return base_args
    merged = list(base_args)
    # Pad with None if base_args is shorter than the param list.
    while len(merged) < len(descriptor.params):
        merged.append(None)
    for i, name in enumerate(descriptor.params):
        if name in overrides:
            merged[i] = overrides[name]
    return merged


# --- factory bound to a backend instance --------------------------------------


@dataclass(slots=True)
class StagePipeline:
    """Per-stage executable bound to a backend-options snapshot.

    `make_ttir(mod, metadata, opt, capability)` and
    `make_ttgir(mod, metadata, opt, capability)` are the upstream signatures.
    Instances of this class can be substituted for those lambdas in the
    backend's `stages` dict.
    """

    capability: int
    num_warps: int
    num_stages: int
    num_ctas: int
    interventions: dict[str, PassIntervention] = field(default_factory=dict)
    on_pass: Callable[[str, PassResult], None] | None = None
    """Callback invoked as (stage_name, pass_result) for every pass run."""
    capture_ir: bool = True
    dump_dir: Path | None = None

    def make_ttir(self, mod: Any, metadata: dict[str, Any]) -> Any:
        plan = build_ttir_plan(
            capability=self.capability,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            num_ctas=self.num_ctas,
        )
        for intervention in self.interventions.values():
            if any(s.descriptor.name == intervention.pass_name for s in plan.steps):
                plan = plan.with_intervention(intervention)
        result = execute_plan(
            mod,
            plan,
            options=ExecutorOptions(capture_ir=self.capture_ir, dump_dir=self.dump_dir),
            on_pass=(lambda r: self.on_pass("ttir", r)) if self.on_pass else None,
        )
        metadata["compilagent_stage_ttir"] = result.model_dump()
        return mod

    def make_ttgir(self, mod: Any, metadata: dict[str, Any]) -> Any:
        plan = build_ttgir_plan(
            capability=self.capability,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            num_ctas=self.num_ctas,
        )
        for intervention in self.interventions.values():
            if any(s.descriptor.name == intervention.pass_name for s in plan.steps):
                plan = plan.with_intervention(intervention)
        result = execute_plan(
            mod,
            plan,
            options=ExecutorOptions(capture_ir=self.capture_ir, dump_dir=self.dump_dir),
            on_pass=(lambda r: self.on_pass("ttgir", r)) if self.on_pass else None,
        )
        metadata["compilagent_stage_ttgir"] = result.model_dump()
        # Triton populates this metadata field after make_ttgir; replicate it
        # so downstream stages (make_llir) see the same module shape.
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod
