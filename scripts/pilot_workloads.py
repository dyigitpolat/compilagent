"""Stable-id workload registration for the cross-space pilot grid (D9).

`run_pilot` drives `OptimizationSession(workload_id=...)`, which requires a
REGISTERED workload whose spec names its backend — the session resolves the
backend from `WorkloadSpec.backend_id`, so backend selection is generic
once registration exists. Three workload families plug in here:

  - the six `triton_source` kernel-source workloads (softmax_4096, ...):
    registered by importing `compilagent.integrations.triton_source`;
  - the six Inductor module workloads (`scripts/modules.py`: rmsnorm,
    swiglu_mlp, mha, gqa, rotary_attn, moe_ffn) → backend `torch_inductor`
    (config-knob decision space);
  - the six Triton kernel workloads (`scripts/kernel_specs.py`:
    fused_softmax, rmsnorm_kernel, layernorm_kernel, gelu_kernel,
    dropout_kernel, matmul_kernel) → backend `triton` (MLIR pass-pipeline
    decision space).

The module/kernel builders were written for `optimize_module` /
`optimize_kernel`, which wrap them in throwaway-id specs; this module
reuses the exact same spec/forward/hook construction (the
`integrations.python.api` helpers) but registers them under their stable
builder names so any driver cell can address them.

Device discipline: `backend_for` is a pure table lookup (no torch import),
so callers can decide on GPU leasing/pinning BEFORE
`ensure_workload_registered` imports torch and materializes CUDA tensors.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

#: Builder-name tables mirrored statically so backend routing never needs a
#: torch import. Keep in sync with scripts/modules.py MODULE_BUILDERS and
#: scripts/kernel_specs.py KERNEL_BUILDERS (asserted inside the register
#: functions, which do import them).
MODULE_WORKLOAD_IDS: tuple[str, ...] = (
    "rmsnorm",
    "swiglu_mlp",
    "mha",
    "gqa",
    "rotary_attn",
    "moe_ffn",
)
KERNEL_WORKLOAD_IDS: tuple[str, ...] = (
    "fused_softmax",
    "rmsnorm_kernel",
    "layernorm_kernel",
    "gelu_kernel",
    "dropout_kernel",
    "matmul_kernel",
)

#: Benchmark cap reused from `CompilagentSettings.max_benchmark_seconds`'s
#: default (the same number `optimize_module`/`optimize_kernel` would use).
_MAX_BENCHMARK_SECONDS = 120.0


def backend_for(workload_id: str) -> str:
    """Backend id a pilot cell's workload will resolve to (pure lookup).

    Unknown ids default to `triton_source` — they are assumed to be (or
    become) registered by the triton_source integration import, preserving
    the driver's historical behavior for source-space ids.
    """

    if workload_id in MODULE_WORKLOAD_IDS:
        return "torch_inductor"
    if workload_id in KERNEL_WORKLOAD_IDS:
        return "triton"
    return "triton_source"


def ensure_workload_registered(workload_id: str) -> str:
    """Register `workload_id` (idempotent) and return its backend id.

    Imports the owning backend integration so the backend registry is
    populated, then registers the workload spec + builder when it is one of
    the 12 knob/pass workloads. Returns `WorkloadSpec.backend_id`.
    """

    import importlib

    backend_id = backend_for(workload_id)
    importlib.import_module(f"compilagent.integrations.{backend_id}")

    from compilagent.core.workload_registry import workload_registry

    if workload_id in workload_registry.ids():
        return backend_id
    if workload_id in MODULE_WORKLOAD_IDS:
        _register_module_workload(workload_id)
    elif workload_id in KERNEL_WORKLOAD_IDS:
        _register_kernel_workload(workload_id)
    # Anything else: assume the integration import registered it; the
    # session raises the canonical unknown-workload error otherwise.
    return backend_id


def _register_module_workload(workload_id: str) -> None:
    """Register one scripts/modules.py builder as a torch_inductor workload."""

    from compilagent.core.workload import WorkloadInstance
    from compilagent.core.workload_registry import register_workload_safely
    from compilagent.integrations.python import api as papi
    from scripts.modules import MODULE_BUILDERS

    module, example_inputs = MODULE_BUILDERS[workload_id]()
    spec = papi._build_module_spec(
        workload_id=workload_id,
        model=module,
        example_inputs=example_inputs,
        backend_id="torch_inductor",
        max_seconds=_MAX_BENCHMARK_SECONDS,
    )
    doc = " ".join((type(module).__doc__ or "").split())
    shapes = [tuple(getattr(t, "shape", ())) for t in example_inputs]
    spec = dc_replace(
        spec,
        title=type(module).__name__,
        description=(
            f"Optimize Inductor compile decisions for `{type(module).__name__}`"
            + (f" — {doc}" if doc else "")
            + f" Input shapes: {shapes}. The module source is fixed."
        ),
    )

    @register_workload_safely(spec)
    def _build(s):
        return WorkloadInstance(
            spec=s,
            forward=papi._make_forward(module, example_inputs),
            example_inputs=example_inputs,
            metadata={
                "kind": "module",
                "module_class": type(module).__name__,
                "module": module,
            },
        )


def _register_kernel_workload(workload_id: str) -> None:
    """Register one scripts/kernel_specs.py builder as a triton workload."""

    from compilagent.core.workload import WorkloadInstance
    from compilagent.core.workload_registry import register_workload_safely
    from compilagent.integrations.python import api as papi
    from scripts.kernel_specs import KERNEL_BUILDERS

    built = KERNEL_BUILDERS[workload_id]()
    kernel = built["kernel"]
    args = tuple(built["args"])
    grid = built["grid"]
    constexpr = dict(built.get("constexpr") or {})

    spec = papi._build_kernel_spec(
        workload_id=workload_id,
        kernel=kernel,
        constexpr=constexpr,
        backend_id="triton",
        max_seconds=_MAX_BENCHMARK_SECONDS,
    )
    spec = dc_replace(
        spec,
        description=(
            f"Optimize the Triton MLIR pass pipeline for kernel "
            f"`{papi._kernel_symbol(kernel)}` (constexpr {constexpr}). The "
            "kernel source and launch grid are fixed; only compiler "
            "decisions change."
        ),
    )
    papi._attach_compile_hook(kernel, args=args, grid=grid, constexpr=constexpr)

    @register_workload_safely(spec)
    def _build(s):
        return WorkloadInstance(
            spec=s,
            forward=papi._make_kernel_forward(kernel, args, grid, constexpr),
            example_inputs=args,
            metadata={
                "kind": "kernel",
                "kernel_symbol": papi._kernel_symbol(kernel),
                "source_path": papi._kernel_source_path(kernel),
            },
        )
