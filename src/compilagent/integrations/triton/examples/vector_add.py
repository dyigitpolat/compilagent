"""Example workload: masked elementwise add Triton kernel.

Registered automatically when `compilagent.integrations.triton` is imported.
The build function lazy-imports `torch` and `triton` so the spec literal
parses on a CPU-only box (the run will fail clearly on CUDA-less hardware
when the user clicks Start in the UI).
"""

from __future__ import annotations

from compilagent.core.workload import (
    BenchmarkBudget,
    DtypePolicy,
    ShapePolicy,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.core.workload_registry import register_workload_safely

_SPEC = WorkloadSpec(
    id="vector_add",
    title="Vector Add",
    description=(
        "Masked elementwise add Triton kernel. Smallest interesting workload "
        "to exercise the Triton MLIR pass pipeline end-to-end."
    ),
    kind=WorkloadKind.KERNEL,
    backend_id="triton",
    dtype_policy=DtypePolicy(activation_dtype="fp32", param_dtype="fp32"),
    shape_policy=ShapePolicy(extra={"n_elements": 8_388_608}),
    tolerance=ToleranceConfig(atol=1e-5, rtol=1e-4),
    budget=BenchmarkBudget(warmup=5, repetitions=20, max_seconds=120.0),
    metadata={
        "source_path": __file__,
        "kernel_symbol": "vector_add_kernel",
        "block_size": 1024,
        "num_warps": 4,
    },
)


@register_workload_safely(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch
    import triton
    import triton.language as tl

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialise vector_add.")

    @triton.jit
    def vector_add_kernel(  # noqa: F841 — closed over by `forward`
        x_ptr,
        y_ptr,
        out_ptr,
        n: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x_vals = tl.load(x_ptr + offsets, mask=mask)
        y_vals = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x_vals + y_vals, mask=mask)

    n = int(spec.shape_policy.extra.get("n_elements", 1024 * 1024))
    block_size = int(spec.metadata.get("block_size", 1024))
    num_warps = int(spec.metadata.get("num_warps", 4))
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[spec.dtype_policy.activation_dtype]

    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    out = torch.empty_like(x)

    def grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    def forward() -> torch.Tensor:
        vector_add_kernel[grid](
            x, y, out, n,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()
        return out

    return WorkloadInstance(
        spec=spec,
        forward=forward,
        example_inputs=(x, y),
        metadata={"output_buffer": out, "n_elements": n},
    )
