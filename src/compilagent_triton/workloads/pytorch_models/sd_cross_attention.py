"""Stable-Diffusion U-Net cross-attention block (text-conditioning step).

Cross-attention here is the dominant cost in latent-diffusion image gen: a
spatial query of shape `[B, H*W, D]` attends over a text-embedding key/value
sequence `[B, L_text, D]`. Real SD-1.5 inner block; widely deployed.

Inductor exposes interesting levers around `epilogue_fusion`,
`max_autotune_gemm`, `shape_padding` (the asymmetric Q vs KV sequence lengths
benefit a lot from padding), and the per-kernel autotune for the two GEMMs
(QK^T and softmax · V).
"""

from __future__ import annotations

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


_SPEC = WorkloadSpec(
    id="sd_cross_attention",
    title="Stable Diffusion cross-attention block",
    description=(
        "U-Net cross-attention: spatial query [B=2, 64*64=4096, 320] attends "
        "over text K/V [B=2, 77, 320]. 8 heads, GroupNorm + linear projections "
        "+ feed-forward (GEGLU). Real SD-1.5 mid-block shape, bf16."
    ),
    kind=WorkloadKind.FULL_MODEL,
    backend_id="torch_inductor",
    entrypoint="compilagent_triton.workloads.pytorch_models.sd_cross_attention:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=2,
        sequence_length=4096,  # 64*64
        extra={"text_seq": 77, "dim": 320, "heads": 8, "ff_mult": 4},
    ),
    tolerance=ToleranceConfig(atol=5e-4, rtol=5e-3),
    budget=BenchmarkBudget(warmup=3, repetitions=15, max_seconds=90.0),
    metadata={"seed": 0},
)


@register_workload(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import math
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(int(spec.metadata.get("seed", 0)))

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        spec.dtype_policy.activation_dtype
    ]
    B = spec.shape_policy.batch_size or 2
    img_seq = spec.shape_policy.sequence_length or 4096
    text_seq = int(spec.shape_policy.extra.get("text_seq", 77))
    dim = int(spec.shape_policy.extra.get("dim", 320))
    heads = int(spec.shape_policy.extra.get("heads", 8))
    head_dim = dim // heads
    ff_mult = int(spec.shape_policy.extra.get("ff_mult", 4))

    class GEGLU(nn.Module):
        def __init__(self, d, mult):
            super().__init__()
            self.proj_in = nn.Linear(d, d * mult * 2)
            self.proj_out = nn.Linear(d * mult, d)

        def forward(self, x):
            a, b = self.proj_in(x).chunk(2, dim=-1)
            return self.proj_out(a * torch.nn.functional.gelu(b))

    class CrossAttnBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.q = nn.Linear(dim, dim, bias=False)
            self.k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=False)
            self.out = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = GEGLU(dim, ff_mult)

        def forward(self, x, ctx):
            h = self.norm1(x)
            q = self.q(h).view(B, img_seq, heads, head_dim).transpose(1, 2)
            k = self.k(ctx).view(B, text_seq, heads, head_dim).transpose(1, 2)
            v = self.v(ctx).view(B, text_seq, heads, head_dim).transpose(1, 2)
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(B, img_seq, dim)
            x = x + self.out(attn)
            x = x + self.ff(self.norm2(x))
            return x

    block = CrossAttnBlock().to(device="cuda", dtype=dtype).eval()
    x = torch.randn(B, img_seq, dim, device="cuda", dtype=dtype)
    ctx = torch.randn(B, text_seq, dim, device="cuda", dtype=dtype)

    def forward():
        with torch.no_grad():
            return block(x, ctx)

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=(x, ctx),
        metadata={"module": block,
                  "param_count": sum(p.numel() for p in block.parameters())},
    )
