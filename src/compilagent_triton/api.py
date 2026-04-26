"""User-facing API for optimizing existing PyTorch / Triton code.

Two entry points cover the common cases:

  - `optimize_module(model, example_inputs, ...)` for any `nn.Module`.
  - `optimize_kernel(kernel, args, grid, ...)` for any `@triton.jit` function.

Both build an in-memory `WorkloadSpec`, register a builder closure, and run the
backend-agnostic `WorkloadSession` end-to-end. The return value carries
correctness + speedup numbers and the compiled callable so the caller can drop
it back into their existing code with no further changes.

Example — switch from `torch.compile` to `compilagent.optimize_module`:

    import compilagent_triton as cgt

    result = cgt.optimize_module(
        model, example_inputs=(x,),
        max_candidates=8,                # number of agent trials
        harness="pydantic_ai",           # or "claude_agent_sdk"
        model_name="mistral:mistral-large-latest",
    )
    print(f"speedup: {result.best_speedup:.3f}× over torch.compile baseline")
    optimized = result.compiled_callable  # ready to call: optimized(x)
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .core.workload import (
    BenchmarkBudget,
    DtypePolicy,
    ShapePolicy,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from .settings import CompilagentSettings
from .trace_store import TraceStore
from .workloads.registry import workload_registry


@dataclass(slots=True)
class OptimizationResult:
    """What `optimize_*` returns to the caller.

    `optimized_callable` is the drop-in replacement: call it with the same
    inputs you passed to `optimize_module(...)` / `optimize_kernel(...)` and
    you get the optimized forward pass / kernel launch. It is set to `None`
    iff no validated candidate beat the baseline, in which case `improved` is
    False and the caller should fall back to their original code path.
    """

    workload_id: str
    backend_id: str
    harness: str
    baseline_median_ms: float | None
    best_speedup: float | None
    best_candidate_id: str | None
    best_median_ms: float | None
    correctness_ok: bool | None
    max_abs_diff: float | None
    final_text: str | None
    elapsed_ms: float
    candidates: list[dict[str, Any]] = field(default_factory=list)
    workspace_root: Path | None = None
    # The optimized drop-in. `None` if no validated candidate beat baseline.
    optimized_callable: Any | None = None
    # The plan that produced `optimized_callable` — list of `Intervention`
    # objects describing the compiler decisions the agent settled on.
    best_plan: Any | None = None

    @property
    def improved(self) -> bool:
        """True iff a validated candidate beat baseline."""

        return self.optimized_callable is not None and bool(
            isinstance(self.best_speedup, (int, float)) and self.best_speedup > 1.0
        )

    def __call__(self, *args, **kwargs):
        """Call the optimized output directly: `result(x)` instead of
        `result.optimized_callable(x)`. Falls back to raising if no
        validated candidate exists.
        """

        if self.optimized_callable is None:
            raise RuntimeError(
                "No validated candidate beat baseline within tolerance. "
                "Use your original code path; `result.improved` is False."
            )
        return self.optimized_callable(*args, **kwargs)


# ---------------------------------------------------------------------------
# PyTorch entry point
# ---------------------------------------------------------------------------


def optimize_module(
    model: Any,
    example_inputs: tuple[Any, ...],
    *,
    max_candidates: int = 4,
    harness: str = "pydantic_ai",
    model_name: str | None = None,
    workload_id: str | None = None,
    workspace_root: Path | None = None,
    activation_dtype: str | None = None,
    param_dtype: str | None = None,
    atol: float = 5e-4,
    rtol: float = 5e-3,
    warmup: int = 3,
    repetitions: int = 10,
    max_seconds: float = 120.0,
    description: str | None = None,
) -> OptimizationResult:
    """Optimize an existing PyTorch `nn.Module` forward pass.

    Args:
        model: Any callable / `nn.Module`. Must run on CUDA — call `.cuda()`
            on the user side before invoking.
        example_inputs: Positional args that will be passed to `model(*args)`.
            They drive the shape/dtype the agent compiles against; using
            production-shape inputs gives the agent the most signal.
        max_candidates: Trial budget. Only successful candidates (compile OK,
            within tolerance, timed) consume a slot.
        harness: "pydantic_ai" (default, fast) or "claude_agent_sdk".
        model_name: LLM provider:model string (e.g. "anthropic:claude-opus-4-7",
            "mistral:mistral-large-latest"). Defaults to the value in
            `CompilagentSettings.from_env()` (env vars / .env).
        workload_id: Optional human-readable id for traces; auto-generated
            from `model.__class__.__name__` if omitted.
        workspace_root: Where to write artifacts + traces. Defaults to a temp
            dir under `./.compilagent-triton/`.
        activation_dtype, param_dtype: Override the inferred dtype. By default
            the API reads `model.parameters().dtype` and the first input's
            dtype.
        atol, rtol: Numerical tolerance for correctness checks.
        warmup, repetitions, max_seconds: Benchmark budget per candidate.
        description: Free-form note shown in the agent's UI / traces.

    Returns:
        `OptimizationResult` with `best_speedup`, `compiled_callable`, etc.
    """

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("optimize_module requires CUDA.")

    # Infer dtype from the model + inputs unless overridden.
    inferred_param_dtype = _torch_dtype_to_str(_first_param_dtype(model)) or "fp32"
    inferred_act_dtype = (
        _torch_dtype_to_str(example_inputs[0].dtype)
        if example_inputs and hasattr(example_inputs[0], "dtype")
        else inferred_param_dtype
    )
    activation_dtype = activation_dtype or inferred_act_dtype
    param_dtype = param_dtype or inferred_param_dtype

    cls_name = type(model).__name__.lower()
    workload_id = workload_id or f"user_{cls_name}_{uuid.uuid4().hex[:6]}"
    description = description or f"User-supplied {type(model).__name__}"

    # First input shape tells us a useful shape policy.
    first = example_inputs[0] if example_inputs else None
    batch = int(first.shape[0]) if (first is not None and hasattr(first, "shape") and first.ndim > 0) else 1
    image_size = None
    if first is not None and hasattr(first, "shape") and first.ndim == 4:
        image_size = (int(first.shape[2]), int(first.shape[3]))
    sequence_length = None
    if first is not None and hasattr(first, "shape") and first.ndim == 3:
        sequence_length = int(first.shape[1])

    spec = WorkloadSpec(
        id=workload_id,
        title=description,
        description=description,
        kind=WorkloadKind.FULL_MODEL,
        backend_id="torch_inductor",
        entrypoint=f"compilagent_triton.api:_inline_pytorch_builder#{workload_id}",
        dtype_policy=DtypePolicy(
            activation_dtype=activation_dtype, param_dtype=param_dtype,
        ),
        shape_policy=ShapePolicy(
            batch_size=batch, image_size=image_size, sequence_length=sequence_length,
        ),
        tolerance=ToleranceConfig(atol=atol, rtol=rtol),
        budget=BenchmarkBudget(
            warmup=warmup, repetitions=repetitions, max_seconds=max_seconds,
        ),
        metadata={"inline": True, "kind": "pytorch_module"},
    )

    def _builder(_spec: WorkloadSpec) -> WorkloadInstance:
        def forward():
            with torch.no_grad():
                return model(*example_inputs)

        return WorkloadInstance(
            spec=_spec,
            forward=forward,
            example_inputs=example_inputs,
            metadata={
                "module": model,
                "param_count": sum(p.numel() for p in model.parameters()),
            },
        )

    workload_registry.register(spec, _builder, replace=True)
    return _run_session(
        workload_id=workload_id,
        max_candidates=max_candidates,
        harness=harness,
        model_name=model_name,
        workspace_root=workspace_root,
    )


# ---------------------------------------------------------------------------
# Triton entry point
# ---------------------------------------------------------------------------


def optimize_kernel(
    kernel: Any,
    args: tuple[Any, ...],
    *,
    grid: Any,
    max_candidates: int = 4,
    harness: str = "pydantic_ai",
    model_name: str | None = None,
    workload_id: str | None = None,
    workspace_root: Path | None = None,
    constexpr: dict[str, Any] | None = None,
    output_index: int = -1,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    warmup: int = 5,
    repetitions: int = 20,
    max_seconds: float = 120.0,
    description: str | None = None,
) -> OptimizationResult:
    """Optimize an existing `@triton.jit` kernel.

    Args:
        kernel: A `@triton.jit`-decorated function.
        args: Positional args passed to `kernel[grid](*args, **constexpr)`.
            Tensors must already be CUDA-resident.
        grid: Either a tuple `(N,)` or a callable `(meta) -> tuple` (Triton's
            standard grid form).
        constexpr: Triton constexprs (e.g. `{"BLOCK_SIZE": 1024}`). Note:
            constexpr autotuning is NOT what this optimizer does — it
            optimizes the compiler's MLIR pass pipeline. Constexprs must be
            fixed by the caller.
        output_index: Which positional arg is the output buffer (used for
            correctness comparison). Default -1 (last arg).
        atol, rtol: Numerical tolerance.
        warmup, repetitions, max_seconds: Benchmark budget per candidate.
    """

    import inspect as _inspect
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("optimize_kernel requires CUDA.")

    constexpr = dict(constexpr or {})
    cls_name = getattr(kernel, "__name__", "kernel")
    workload_id = workload_id or f"user_{cls_name}_{uuid.uuid4().hex[:6]}"
    description = description or f"User-supplied @triton.jit `{cls_name}`"

    # Locate the kernel source so the triton backend can re-JIT it.
    src_fn = getattr(kernel, "fn", None) or kernel
    try:
        source_path = _inspect.getsourcefile(src_fn) or _inspect.getfile(src_fn)
    except TypeError:
        source_path = None

    spec = WorkloadSpec(
        id=workload_id,
        title=description,
        description=description,
        kind=WorkloadKind.KERNEL,
        backend_id="triton",
        entrypoint=f"compilagent_triton.api:_inline_triton_builder#{workload_id}",
        dtype_policy=DtypePolicy(
            activation_dtype=_torch_dtype_to_str(
                args[0].dtype if args and hasattr(args[0], "dtype") else None
            ) or "fp32",
            param_dtype="fp32",
        ),
        shape_policy=ShapePolicy(extra={
            "n_elements": int(args[0].numel())
            if args and hasattr(args[0], "numel") else 0,
        }),
        tolerance=ToleranceConfig(atol=atol, rtol=rtol),
        budget=BenchmarkBudget(
            warmup=warmup, repetitions=repetitions, max_seconds=max_seconds,
        ),
        metadata={
            "kernel_symbol": getattr(src_fn, "__name__", "kernel"),
            "source_path": str(source_path) if source_path else "",
            "inline": True,
            "kind": "triton_kernel",
        },
    )

    def _builder(_spec: WorkloadSpec) -> WorkloadInstance:
        def forward():
            kernel[grid](*args, **constexpr)
            torch.cuda.synchronize()
            return args[output_index]

        return WorkloadInstance(
            spec=_spec,
            forward=forward,
            example_inputs=args,
            metadata={
                "kernel": kernel,
                "constexpr": constexpr,
                "grid": grid,
                "output_index": output_index,
            },
        )

    workload_registry.register(spec, _builder, replace=True)
    return _run_session(
        workload_id=workload_id,
        max_candidates=max_candidates,
        harness=harness,
        model_name=model_name,
        workspace_root=workspace_root,
    )


# ---------------------------------------------------------------------------
# Shared session driver
# ---------------------------------------------------------------------------


def _run_session(
    *,
    workload_id: str,
    max_candidates: int,
    harness: str,
    model_name: str | None,
    workspace_root: Path | None,
) -> OptimizationResult:
    """Run a non-streaming WorkloadSession and project to OptimizationResult."""

    from .workload_runner import _run_pydantic_ai, _run_claude_sdk

    # Workspace lives under `<cwd>/.compilagent-triton/api/<run-id>/` by
    # default — every artifact (output_code.py, ttgir, traces) is written
    # here so the user can inspect them after the call returns.
    workspace_root = Path(workspace_root or Path.cwd() / ".compilagent-triton")
    workspace_root.mkdir(parents=True, exist_ok=True)
    run_id = f"api-{uuid.uuid4().hex[:10]}"
    trace_store = TraceStore(workspace_root).ensure()
    settings = CompilagentSettings.from_env(project_root=workspace_root.parent)
    if model_name:
        # CompilagentSettings is a pydantic model — copy with overrides.
        settings = settings.model_copy(update={"model_name": model_name})

    runner = _run_claude_sdk if harness == "claude_agent_sdk" else _run_pydantic_ai
    summary = asyncio.run(runner(
        workload_id=workload_id, run_id=run_id,
        workspace_root=workspace_root, trace_store=trace_store,
        settings=settings, max_candidates=max_candidates,
    ))

    candidates_list = _materialise_candidate_list(
        run_id=run_id, workspace_root=workspace_root,
    )

    return OptimizationResult(
        workload_id=workload_id,
        backend_id=workload_registry.get_spec(workload_id).backend_id,
        harness=harness,
        baseline_median_ms=summary.get("baseline_median_ms"),
        best_speedup=summary.get("best_speedup"),
        best_candidate_id=summary.get("best_candidate_id"),
        best_median_ms=summary.get("best_median_ms"),
        correctness_ok=summary.get("best_correctness_ok"),
        max_abs_diff=summary.get("best_max_abs_diff"),
        final_text=summary.get("final_text"),
        elapsed_ms=summary.get("elapsed_ms", 0.0),
        candidates=candidates_list,
        workspace_root=workspace_root,
        optimized_callable=summary.get("_compiled_callable"),
        best_plan=summary.get("_best_plan"),
    )


def _materialise_candidate_list(
    *,
    run_id: str,
    workspace_root: Path,
) -> list[dict[str, Any]]:
    """Read the trace events to reconstruct a per-candidate summary list."""

    trace_store = TraceStore(workspace_root)
    events = trace_store.read_events()
    by_cand: dict[str, dict[str, Any]] = {}
    for ev in events:
        kind = ev.kind
        payload = ev.payload or {}
        if kind == "candidate.proposed":
            for c in payload.get("candidates", []) or []:
                if c.get("kind") == "search_space_summary":
                    continue
                by_cand.setdefault(c["id"], {"id": c["id"]}).update({
                    "description": c.get("description", ""),
                    "changes": c.get("changes", {}),
                })
        elif kind == "benchmark.completed":
            cid = payload.get("candidate_id")
            if cid:
                by_cand.setdefault(cid, {"id": cid}).update({
                    "median_ms": payload.get("median_ms"),
                    "speedup_vs_baseline": payload.get("speedup_vs_baseline"),
                })
    return list(by_cand.values())


def _torch_dtype_to_str(dtype: Any) -> str | None:
    if dtype is None:
        return None
    name = getattr(dtype, "__name__", None) or str(dtype)
    return {
        "torch.float32": "fp32", "float32": "fp32",
        "torch.bfloat16": "bf16", "bfloat16": "bf16",
        "torch.float16": "fp16", "float16": "fp16",
    }.get(name, None)


def _first_param_dtype(model: Any) -> Any:
    try:
        return next(model.parameters()).dtype
    except Exception:  # noqa: BLE001
        return None


# Sentinel attribute lookups so the registry's entrypoint string can route
# back here without dotted-import gymnastics. The session never actually
# resolves `entrypoint` on inline workloads; the builder closure already lives
# in the registry. The dotted name is just kept human-readable for traces.
_inline_pytorch_builder = object()
_inline_triton_builder = object()
