"""Llama-style decoder block: RMSNorm → GroupedQueryAttention → SwiGLU FFN.

This is the dominant inference op for modern open-weights LLMs (Llama 2/3,
Mistral, Qwen, …). The block exercises:

  - **RMSNorm** (LayerNorm without mean centering).
  - **GroupedQueryAttention** (KV heads compressed; the standard Llama-3 shape
    has 32 Q heads / 8 KV heads).
  - **SwiGLU** FFN (two parallel up-projections, SiLU on one, then gated mul,
    then down-projection).
  - **Causal SDPA** (uses `is_causal=True`).

Inductor levers that often matter here: `shape_padding` (RoPE sin/cos cache
shape is awkward), `epilogue_fusion`, `max_autotune_gemm` for the SwiGLU
matmuls, and FX-graph rewrites that fuse RMSNorm + residual.
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
    id="llama_decoder_block",
    title="Llama-3 8B decoder block (one layer)",
    description=(
        "RMSNorm + GroupedQueryAttention (32 Q heads, 8 KV heads, 128 head dim) "
        "+ SwiGLU FFN (4096→14336→4096). Causal SDPA. Inputs `[B=2, 2048, 4096]` "
        "bf16 — typical Llama-3 8B inference shape."
    ),
    kind=WorkloadKind.FULL_MODEL,
    backend_id="torch_inductor",
    entrypoint="compilagent_triton.workloads.pytorch_models.llama_decoder_block:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=2,
        sequence_length=2048,
        extra={"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128, "ffn_dim": 14336},
    ),
    tolerance=ToleranceConfig(atol=5e-4, rtol=5e-3),
    budget=BenchmarkBudget(warmup=3, repetitions=10, max_seconds=120.0),
    metadata={"seed": 0},
)


@register_workload(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(int(spec.metadata.get("seed", 0)))

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        spec.dtype_policy.activation_dtype
    ]
    B = spec.shape_policy.batch_size or 2
    L = spec.shape_policy.sequence_length or 2048
    dim = int(spec.shape_policy.extra.get("dim", 4096))
    n_heads = int(spec.shape_policy.extra.get("n_heads", 32))
    n_kv_heads = int(spec.shape_policy.extra.get("n_kv_heads", 8))
    head_dim = int(spec.shape_policy.extra.get("head_dim", 128))
    ffn_dim = int(spec.shape_policy.extra.get("ffn_dim", 14336))

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = eps

        def forward(self, x):
            inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return (x.float() * inv).to(x.dtype) * self.weight

    class GQA(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
            self.rep = n_heads // n_kv_heads

        def forward(self, x):
            q = self.q_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, n_kv_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, n_kv_heads, head_dim).transpose(1, 2)
            # Repeat-interleave KV heads to align with Q heads (the canonical
            # Llama-3 GQA expansion before SDPA).
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o = o.transpose(1, 2).reshape(B, L, n_heads * head_dim)
            return self.o_proj(o)

    class SwiGLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
            self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
            self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = RMSNorm(dim)
            self.attn = GQA()
            self.norm2 = RMSNorm(dim)
            self.ffn = SwiGLU()

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

    block = Block().to(device="cuda", dtype=dtype).eval()
    inputs = (torch.randn(B, L, dim, device="cuda", dtype=dtype),)

    def forward():
        with torch.no_grad():
            return block(inputs[0])

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=inputs,
        metadata={"module": block,
                  "param_count": sum(p.numel() for p in block.parameters())},
    )
