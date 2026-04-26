"""RMSNorm forward kernel — the LayerNorm variant used in Llama / Mistral.

  out = (x * rsqrt(mean(x^2) + eps)) * g

One block per row, vectorised reduction across the hidden dimension. Exposes
the agent's compiler-pass surface around layout selection (reduction axis vs.
broadcast axis) and the inter-pass IR rewriter (RMSNorm benefits from
`-tritongpu-coalesce` and `-tritongpu-optimize-thread-locality`).
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
def rms_norm_kernel(
    x_ptr, g_ptr, out_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    y = (x * inv) * g
    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)


def _compilagent_compile_rms_norm(meta: dict) -> object:
    """Launch once for the harness so it can scrape the `.asm` dict."""

    import torch
    n_rows = int(meta.get("n_rows", 8192))
    n_cols = int(meta.get("n_cols", 4096))
    block_size = int(meta.get("BLOCK_SIZE", n_cols))
    num_warps = int(meta.get("num_warps", 4))
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
    g = torch.ones(n_cols, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    handle = rms_norm_kernel[(n_rows,)](
        x, g, out, n_rows, n_cols, 1e-6,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return handle


rms_norm_kernel.compilagent_compile = _compilagent_compile_rms_norm


_SPEC = WorkloadSpec(
    id="rms_norm",
    title="RMSNorm forward (Llama-style)",
    description=(
        "Row-wise RMSNorm: `y = (x * rsqrt(mean(x^2) + eps)) * g`. Inputs "
        "`[N=8192 rows × D=4096 cols]` fp32. One row per program; reduction "
        "across the hidden dim."
    ),
    kind=WorkloadKind.KERNEL,
    backend_id="triton",
    entrypoint="compilagent_triton.workloads.triton_kernels.rms_norm:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="fp32", param_dtype="fp32"),
    shape_policy=ShapePolicy(extra={"n_rows": 8192, "n_cols": 4096}),
    tolerance=ToleranceConfig(atol=1e-4, rtol=1e-3),
    budget=BenchmarkBudget(warmup=5, repetitions=20, max_seconds=120.0),
    metadata={
        "kernel_symbol": "rms_norm_kernel",
        "source_path": __file__,
        "block_size": 4096,
        "num_warps": 4,
    },
)


@register_workload(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(int(spec.metadata.get("seed", 0)))
    n_rows = int(spec.shape_policy.extra.get("n_rows", 8192))
    n_cols = int(spec.shape_policy.extra.get("n_cols", 4096))
    block_size = int(spec.metadata.get("block_size", n_cols))
    num_warps = int(spec.metadata.get("num_warps", 4))
    dtype = torch.float32
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype)
    g = torch.ones(n_cols, device="cuda", dtype=dtype)
    out = torch.empty_like(x)

    def forward():
        rms_norm_kernel[(n_rows,)](
            x, g, out, n_rows, n_cols, 1e-6,
            BLOCK_SIZE=block_size, num_warps=num_warps,
        )
        torch.cuda.synchronize()
        return out

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=(x, g),
        metadata={"output_buffer": out, "n_rows": n_rows, "n_cols": n_cols},
    )
