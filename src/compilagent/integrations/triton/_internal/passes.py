"""Catalog of Triton MLIR passes exposed as introspectable, parameterizable units.

The Triton NVIDIA backend builds its TTIR/TTGIR pipelines by calling functions
from `triton._C.libtriton.passes.*` and `triton._C.libtriton.nvidia.passes.*`.
Each function takes a `pass_manager`, plus optional pass-specific arguments,
and appends a single MLIR pass to the manager. The catalog below names every
pass we know about, records its origin module, parameter signature, and a
short human-readable purpose so the agent can reason about which lever to
pull.

The catalog is descriptive: turning it into an executable plan is the job of
`pipeline.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from triton._C.libtriton import nvidia, passes  # type: ignore[import-not-found]


@dataclass(frozen=True, slots=True)
class PassDescriptor:
    """Describes a single MLIR pass surface."""

    name: str  # canonical short name, e.g. "tritongpu-coalesce"
    origin: str  # which submodule it lives in: "common" | "ttir" | "ttgpuir" | "ttnvgpuir"
    pyname: str  # the python wrapper function name, e.g. "add_coalesce"
    purpose: str  # one-line human description
    params: tuple[str, ...] = field(default_factory=tuple)
    """Names of extra parameters past the pass_manager. Order matches the wrapper."""
    capability_min: int | None = None
    """If set, only valid for compute capability >= this (e.g. 80, 90, 100)."""
    capability_max: int | None = None


def _pass_callable(origin: str, pyname: str) -> Callable[..., None]:
    if origin == "common":
        return getattr(passes.common, pyname)
    if origin == "ttir":
        return getattr(passes.ttir, pyname)
    if origin == "ttgpuir":
        return getattr(passes.ttgpuir, pyname)
    if origin == "ttnvgpuir":
        return getattr(nvidia.passes.ttnvgpuir, pyname)
    if origin == "hopper":
        return getattr(nvidia.passes.hopper, pyname)
    raise KeyError(f"unknown pass origin `{origin}`")


# --- the catalog --------------------------------------------------------------
# Names mirror the canonical MLIR pass names (not the python `add_*` wrapper).
# Where a wrapper takes extra args (num_warps, num_stages, capability bool, etc.)
# the params tuple records the parameter ordering expected by the wrapper after
# the pass_manager argument.

PASS_CATALOG: tuple[PassDescriptor, ...] = (
    # Common housekeeping passes
    PassDescriptor("inliner", "common", "add_inliner",
                   "Inline calls to internal functions before lowering."),
    PassDescriptor("canonicalizer", "common", "add_canonicalizer",
                   "Standard MLIR canonicalization (folding, simplification)."),
    PassDescriptor("cse", "common", "add_cse",
                   "Common-subexpression elimination."),
    PassDescriptor("symbol-dce", "common", "add_symbol_dce",
                   "Dead-symbol elimination."),
    PassDescriptor("sccp", "common", "add_sccp",
                   "Sparse conditional constant propagation."),
    PassDescriptor("licm", "common", "add_licm",
                   "Loop-invariant code motion."),

    # TTIR-stage passes
    PassDescriptor("triton-rewrite-tensor-pointer", "ttir", "add_rewrite_tensor_pointer",
                   "Rewrite block-pointer tensors to scalar pointers."),
    PassDescriptor("triton-rewrite-tensor-descriptor-to-pointer", "ttir",
                   "add_rewrite_tensor_descriptor_to_pointer",
                   "Lower tensor-descriptor accesses to pointer arithmetic (pre-Hopper)."),
    PassDescriptor("triton-combine", "ttir", "add_combine",
                   "Algebraic / pattern-based combination of TTIR ops."),
    PassDescriptor("triton-reorder-broadcast", "ttir", "add_reorder_broadcast",
                   "Hoist broadcasts to expose more fusion opportunities."),
    PassDescriptor("triton-loop-unroll", "ttir", "add_loop_unroll",
                   "Unroll tagged loops to reduce branch overhead."),
    PassDescriptor("triton-loop-aware-cse", "ttir", "add_loop_aware_cse",
                   "CSE that respects loop iteration boundaries."),
    PassDescriptor("triton-licm", "ttir", "add_triton_licm",
                   "Triton-aware loop-invariant code motion."),
    PassDescriptor("convert-triton-to-tritongpu", "ttir", "add_convert_to_ttgpuir",
                   "Lower TTIR to TTGIR with target-specific layouts.",
                   params=("target", "num_warps", "threads_per_warp", "num_ctas")),

    # TTGIR-stage passes
    PassDescriptor("tritongpu-coalesce", "ttgpuir", "add_coalesce",
                   "Pick layouts that maximize global-load coalescing."),
    PassDescriptor("tritongpu-f32-dot-tc", "ttgpuir", "add_f32_dot_tc",
                   "Decompose f32 dot products into TF32-friendly tensor-core ops.",
                   params=("emu_tf32",)),
    PassDescriptor("tritongpu-remove-layout-conversions", "ttgpuir",
                   "add_remove_layout_conversions",
                   "Eliminate redundant convert-layout ops between equivalent encodings."),
    PassDescriptor("tritongpu-optimize-thread-locality", "ttgpuir",
                   "add_optimize_thread_locality",
                   "Reorder ops to keep tensor slices in registers per thread."),
    PassDescriptor("tritongpu-accelerate-matmul", "ttgpuir", "add_accelerate_matmul",
                   "Pick MMA tile shapes and tensor-core dispatch."),
    PassDescriptor("tritongpu-optimize-dot-operands", "ttgpuir",
                   "add_optimize_dot_operands",
                   "Reshape dot operands for tensor-core layout fit.",
                   params=("capability_ge_80",)),
    PassDescriptor("tritongpu-fuse-nested-loops", "ttgpuir", "add_fuse_nested_loops",
                   "Flatten nested loops to expose more pipelining."),
    PassDescriptor("tritongpu-combine-tensor-select-and-if", "ttgpuir",
                   "add_combine_tensor_select_and_if",
                   "Combine select/if pairs that materialize the same tensor."),
    PassDescriptor("tritongpu-assign-latencies", "ttgpuir", "add_assign_latencies",
                   "Tag loop dependencies with latency for the scheduler.",
                   params=("num_stages",)),
    PassDescriptor("tritongpu-schedule-loops", "ttgpuir", "add_schedule_loops",
                   "Order loop body ops to overlap memory and compute."),
    PassDescriptor("tritongpu-pipeline", "ttgpuir", "add_pipeline",
                   "Software-pipeline matmul / load loops with `num_stages` buffers.",
                   params=("num_stages", "dump_enabled"),
                   capability_min=80),
    PassDescriptor("tritongpu-prefetch", "ttgpuir", "add_prefetch",
                   "Insert prefetch ops to hide global-load latency."),
    PassDescriptor("tritongpu-coalesce-async-copy", "ttgpuir",
                   "add_coalesce_async_copy",
                   "Coalesce async copy_global_to_shared ops."),
    PassDescriptor("tritongpu-reduce-data-duplication", "ttgpuir",
                   "add_reduce_data_duplication",
                   "Avoid redundant per-warp duplication of shared tensors."),
    PassDescriptor("tritongpu-reorder-instructions", "ttgpuir",
                   "add_reorder_instructions",
                   "Reorder instructions inside basic blocks for ILP."),
    PassDescriptor("tritongpu-optimize-accumulator-init", "ttgpuir",
                   "add_optimize_accumulator_init",
                   "Hoist matmul accumulator init out of inner loop.",
                   capability_min=100),
    PassDescriptor("tritongpu-hoist-tmem-alloc", "ttgpuir", "add_hoist_tmem_alloc",
                   "Hoist tensor-memory allocations.",
                   params=("hoist_out_of_if",), capability_min=100),
    PassDescriptor("tritongpu-warp-specialize", "ttgpuir", "add_warp_specialize",
                   "Split warps into producer/consumer roles for matmul.",
                   params=("num_stages",), capability_min=100),
    PassDescriptor("tritongpu-optimize-partition-warps", "ttgpuir",
                   "add_optimize_partition_warps",
                   "Optimize warp-partition layout after warp-specialize.",
                   capability_min=100),

    # NVIDIA-specific TTGIR passes
    PassDescriptor("ttng-plan-cta", "ttnvgpuir", "add_plan_cta",
                   "Plan CTA / cluster topology."),
    PassDescriptor("ttng-optimize-descriptor-encoding", "ttnvgpuir",
                   "add_optimize_descriptor_encoding",
                   "Pick best encoding for tensor-descriptor accesses."),
    PassDescriptor("ttng-tma-lowering", "ttnvgpuir", "add_tma_lowering",
                   "Lower TMA ops for sm_90+.", capability_min=90),
    PassDescriptor("ttng-optimize-tmem-layouts", "ttnvgpuir",
                   "add_optimize_tmem_layouts",
                   "Optimize tensor-memory layouts.", capability_min=100),
    PassDescriptor("ttng-promote-lhs-to-tmem", "ttnvgpuir", "add_promote_lhs_to_tmem",
                   "Promote matmul LHS to tensor memory.", capability_min=100),
    PassDescriptor("ttng-interleave-tmem", "ttnvgpuir", "add_interleave_tmem",
                   "Interleave tensor-memory accesses.", capability_min=100),
    PassDescriptor("ttng-fence-insertion", "ttnvgpuir", "add_fence_insertion",
                   "Insert memory-ordering fences.",
                   params=("capability",)),
    PassDescriptor("ttng-lower-mma", "ttnvgpuir", "add_lower_mma",
                   "Lower abstract MMA to NVIDIA tensor-core ops."),
    PassDescriptor("ttng-remove-tmem-tokens", "ttnvgpuir", "add_remove_tmem_tokens",
                   "Strip tmem token bookkeeping.", capability_min=100),
)


def get_pass(name: str) -> PassDescriptor:
    for descriptor in PASS_CATALOG:
        if descriptor.name == name:
            return descriptor
    raise KeyError(f"unknown pass `{name}`")


def list_passes_for_capability(capability: int) -> list[PassDescriptor]:
    out: list[PassDescriptor] = []
    for desc in PASS_CATALOG:
        if desc.capability_min is not None and capability < desc.capability_min:
            continue
        if desc.capability_max is not None and capability > desc.capability_max:
            continue
        out.append(desc)
    return out


def callable_for(descriptor: PassDescriptor) -> Callable[..., None]:
    return _pass_callable(descriptor.origin, descriptor.pyname)


def all_pass_names() -> list[str]:
    return [d.name for d in PASS_CATALOG]


# --- exported helpers used by pipeline.py -------------------------------------


def describe_pass(name: str) -> dict[str, Any]:
    desc = get_pass(name)
    return {
        "name": desc.name,
        "origin": desc.origin,
        "pyname": desc.pyname,
        "purpose": desc.purpose,
        "params": list(desc.params),
        "capability_min": desc.capability_min,
        "capability_max": desc.capability_max,
    }
