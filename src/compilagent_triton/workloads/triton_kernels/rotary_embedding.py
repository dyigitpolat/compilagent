"""RoPE (rotary position embedding) kernel — Llama / GPT-NeoX form.

Applies a position-dependent 2D rotation to pairs of channels in each Q / K
head. The rotation angles come from precomputed `cos[L, head_dim/2]` and
`sin[L, head_dim/2]` tables.

  out[..., 2i]   = x[..., 2i]   * cos[t,i] - x[..., 2i+1] * sin[t,i]
  out[..., 2i+1] = x[..., 2i]   * sin[t,i] + x[..., 2i+1] * cos[t,i]

One program per (batch, head, time) triple; vectorised across head_dim/2.
The interleaved access pattern stresses Triton's coalescer and is sensitive
to `-tritongpu-reorder-instructions` and `-tritongpu-optimize-thread-locality`.
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
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_b, stride_h, stride_t,
    seq_len, head_dim,
    HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    if pid >= seq_len:
        return
    base = bh * stride_h + pid * stride_t
    idx = tl.arange(0, HALF)
    mask = idx < HALF
    x_even = tl.load(x_ptr + base + 2 * idx, mask=mask).to(tl.float32)
    x_odd = tl.load(x_ptr + base + 2 * idx + 1, mask=mask).to(tl.float32)
    c = tl.load(cos_ptr + pid * HALF + idx, mask=mask).to(tl.float32)
    s = tl.load(sin_ptr + pid * HALF + idx, mask=mask).to(tl.float32)
    y_even = x_even * c - x_odd * s
    y_odd = x_even * s + x_odd * c
    tl.store(out_ptr + base + 2 * idx, y_even, mask=mask)
    tl.store(out_ptr + base + 2 * idx + 1, y_odd, mask=mask)


def _compilagent_compile_rope(meta: dict) -> object:
    import torch
    B = int(meta.get("batch", 4))
    H = int(meta.get("heads", 32))
    L = int(meta.get("seq_len", 2048))
    D = int(meta.get("head_dim", 128))
    num_warps = int(meta.get("num_warps", 4))
    half = D // 2
    x = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16).contiguous()
    out = torch.empty_like(x)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device="cuda").float() / half))
    pos = torch.arange(L, device="cuda").float()
    freqs = pos[:, None] * inv_freq[None, :]
    cos = freqs.cos().contiguous()
    sin = freqs.sin().contiguous()
    sb, sh, st, _ = x.stride()
    handle = rope_kernel[(L, B * H)](
        x, cos, sin, out,
        sb, sh, st, L, D,
        HALF=half, num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return handle


rope_kernel.compilagent_compile = _compilagent_compile_rope


_SPEC = WorkloadSpec(
    id="rotary_embedding",
    title="RoPE (rotary position embedding)",
    description=(
        "Position-dependent 2D rotation of head-dim pairs. Inputs `[B=4, "
        "H=32, L=2048, D=128]` bf16 — Llama-3 8B Q/K shape. cos/sin tables "
        "cached on device. Hot kernel in every transformer with rotary embeddings."
    ),
    kind=WorkloadKind.KERNEL,
    backend_id="triton",
    entrypoint="compilagent_triton.workloads.triton_kernels.rotary_embedding:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=4, sequence_length=2048,
        extra={"heads": 32, "head_dim": 128},
    ),
    tolerance=ToleranceConfig(atol=5e-3, rtol=5e-3),
    budget=BenchmarkBudget(warmup=5, repetitions=20, max_seconds=120.0),
    metadata={
        "kernel_symbol": "rope_kernel",
        "source_path": __file__,
        "num_warps": 4,
    },
)


@register_workload(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import math
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(int(spec.metadata.get("seed", 0)))
    B = spec.shape_policy.batch_size or 4
    L = spec.shape_policy.sequence_length or 2048
    H = int(spec.shape_policy.extra.get("heads", 32))
    D = int(spec.shape_policy.extra.get("head_dim", 128))
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        spec.dtype_policy.activation_dtype
    ]
    x = torch.randn(B, H, L, D, device="cuda", dtype=dtype).contiguous()
    out = torch.empty_like(x)
    half = D // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device="cuda").float() / half))
    pos = torch.arange(L, device="cuda").float()
    freqs = pos[:, None] * inv_freq[None, :]               # [L, half]
    cos = freqs.cos().contiguous()
    sin = freqs.sin().contiguous()
    num_warps = int(spec.metadata.get("num_warps", 4))

    stride_b, stride_h, stride_t, _ = x.stride()

    def forward():
        rope_kernel[(L, B * H)](
            x, cos, sin, out,
            stride_b, stride_h, stride_t, L, D,
            HALF=half, num_warps=num_warps,
        )
        torch.cuda.synchronize()
        return out

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=(x, cos, sin),
        metadata={"output_buffer": out, "head_dim": D, "seq_len": L},
    )
