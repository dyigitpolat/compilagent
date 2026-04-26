"""Causal-masked row-wise softmax — the inner loop of any masked attention.

For each row, compute `softmax(x[i, j] + mask[i, j])` where `mask[i, j] = -inf
if j > i else 0`. Numerically stable: subtract row max before exp.

  m_i = max_j (x[i, j] + mask[i, j])
  e_i = exp(x[i, j] + mask[i, j] - m_i)
  out[i, j] = e_i / sum(e_i)

This is what `torch.nn.functional.scaled_dot_product_attention` decomposes
to for the "math" backend; benchmarking it standalone gives the agent room
to push pass-pipeline interventions like `-tritongpu-pipeline` (the row
reduction is a candidate for software pipelining when n_cols is large).
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
def causal_softmax_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols
    causal = col_idx <= row
    full_mask = mask & causal
    x = tl.load(x_ptr + row * n_cols + col_idx,
                mask=full_mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(full_mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(out_ptr + row * n_cols + col_idx, out, mask=mask)


def _compilagent_compile_softmax_mask(meta: dict) -> object:
    import torch
    n_rows = int(meta.get("n_rows", 2048))
    n_cols = int(meta.get("n_cols", 2048))
    block_size = int(meta.get("BLOCK_SIZE", n_cols))
    num_warps = int(meta.get("num_warps", 8))
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    handle = causal_softmax_kernel[(n_rows,)](
        x, out, n_rows, n_cols,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return handle


causal_softmax_kernel.compilagent_compile = _compilagent_compile_softmax_mask


_SPEC = WorkloadSpec(
    id="fused_softmax_mask",
    title="Causal-masked softmax",
    description=(
        "Row-wise causally-masked softmax with online-stable formulation. "
        "Inputs `[N=2048 rows × D=2048 cols]` fp32 — typical Llama-3 attn-score "
        "shape per (batch, head)."
    ),
    kind=WorkloadKind.KERNEL,
    backend_id="triton",
    entrypoint="compilagent_triton.workloads.triton_kernels.fused_softmax_mask:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="fp32", param_dtype="fp32"),
    shape_policy=ShapePolicy(extra={"n_rows": 2048, "n_cols": 2048}),
    tolerance=ToleranceConfig(atol=1e-5, rtol=1e-4),
    budget=BenchmarkBudget(warmup=5, repetitions=20, max_seconds=120.0),
    metadata={
        "kernel_symbol": "causal_softmax_kernel",
        "source_path": __file__,
        "block_size": 2048,
        "num_warps": 8,
    },
)


@register_workload(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(int(spec.metadata.get("seed", 0)))
    n_rows = int(spec.shape_policy.extra.get("n_rows", 2048))
    n_cols = int(spec.shape_policy.extra.get("n_cols", 2048))
    block_size = int(spec.metadata.get("block_size", n_cols))
    num_warps = int(spec.metadata.get("num_warps", 8))
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    def forward():
        causal_softmax_kernel[(n_rows,)](
            x, out, n_rows, n_cols,
            BLOCK_SIZE=block_size, num_warps=num_warps,
        )
        torch.cuda.synchronize()
        return out

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=(x,),
        metadata={"output_buffer": out, "n_rows": n_rows, "n_cols": n_cols},
    )
