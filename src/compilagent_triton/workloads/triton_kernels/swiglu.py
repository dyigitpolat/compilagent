"""SwiGLU activation kernel — Llama-style FFN gating.

Input is split on the last axis into `gate` and `up`; output is
`silu(gate) * up`. This is the elementwise hot-loop in every Llama / Mistral
FFN; it's cheap per-element but called many times so memory bandwidth and
load coalescing dominate. Compiler-pass surface: `-tritongpu-coalesce`,
`-tritongpu-remove-layout-conversions`, eviction-policy choice.
"""

from __future__ import annotations

import triton
import triton.language as tl

from ...core.workload import (
    BenchmarkBudget,
    DtypePolicy,
    ShapePolicy,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from ..registry import register_workload


@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sigm = 1.0 / (1.0 + tl.exp(-g))
    y = (g * sigm) * u
    tl.store(out_ptr + offs, y, mask=mask)


def _compilagent_compile_swiglu(meta: dict) -> object:
    import torch
    n = int(meta.get("n_elements", 1 << 22))
    block_size = int(meta.get("BLOCK_SIZE", 1024))
    num_warps = int(meta.get("num_warps", 4))
    gate = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(gate)
    handle = swiglu_kernel[(triton.cdiv(n, block_size),)](
        gate, up, out, n,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return handle


swiglu_kernel.compilagent_compile = _compilagent_compile_swiglu


_SPEC = WorkloadSpec(
    id="swiglu",
    title="SwiGLU activation",
    description=(
        "Elementwise SwiGLU: `silu(gate) * up`. Inputs are two `[B*L, "
        "ffn_dim]` tensors — Llama-3 8B intermediate shape: `[2*2048, 14336]`."
    ),
    kind=WorkloadKind.KERNEL,
    backend_id="triton",
    entrypoint="compilagent_triton.workloads.triton_kernels.swiglu:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=2, sequence_length=2048,
        extra={"ffn_dim": 14336},
    ),
    tolerance=ToleranceConfig(atol=5e-3, rtol=5e-3),
    budget=BenchmarkBudget(warmup=5, repetitions=20, max_seconds=120.0),
    metadata={
        "kernel_symbol": "swiglu_kernel",
        "source_path": __file__,
        "block_size": 1024,
        "num_warps": 4,
    },
)


@register_workload(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(int(spec.metadata.get("seed", 0)))
    B = spec.shape_policy.batch_size or 2
    L = spec.shape_policy.sequence_length or 2048
    ffn = int(spec.shape_policy.extra.get("ffn_dim", 14336))
    n = B * L * ffn
    block_size = int(spec.metadata.get("block_size", 1024))
    num_warps = int(spec.metadata.get("num_warps", 4))
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        spec.dtype_policy.activation_dtype
    ]
    gate = torch.randn(n, device="cuda", dtype=dtype)
    up = torch.randn(n, device="cuda", dtype=dtype)
    out = torch.empty_like(gate)

    def forward():
        swiglu_kernel[(triton.cdiv(n, block_size),)](
            gate, up, out, n,
            BLOCK_SIZE=block_size, num_warps=num_warps,
        )
        torch.cuda.synchronize()
        return out

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=(gate, up),
        metadata={"output_buffer": out, "n_elements": n},
    )
