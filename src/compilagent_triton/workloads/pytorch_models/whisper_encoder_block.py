"""Whisper audio-encoder block (one transformer layer over a 1500-token mel sequence).

Whisper-large encodes 30s of 16kHz audio to a 1500-token sequence at d=1280;
this is a single transformer encoder layer at that shape. Self-attention only
(no cross-attention), gelu MLP, pre-norm. Fewer ops than Llama but a longer
sequence — Inductor's matmul-template choice matters more here because the
QK^T matmul is `[1500 × 1500]` per head per batch, well above the threshold
where flash-attn-style fused kernels can win over the unfused
`scaled_dot_product_attention` decomposition.
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
    id="whisper_encoder_block",
    title="Whisper-large audio encoder block",
    description=(
        "One Whisper-large transformer encoder layer (d=1280, 20 heads, ffn=5120, "
        "seq=1500). LayerNorm + MHSA + LayerNorm + GELU MLP. Inputs `[B=4, 1500, 1280]` "
        "bf16 — the standard 30 s mel-token shape."
    ),
    kind=WorkloadKind.FULL_MODEL,
    backend_id="torch_inductor",
    entrypoint="compilagent_triton.workloads.pytorch_models.whisper_encoder_block:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=4,
        sequence_length=1500,
        extra={"dim": 1280, "heads": 20, "ffn_dim": 5120},
    ),
    tolerance=ToleranceConfig(atol=5e-4, rtol=5e-3),
    budget=BenchmarkBudget(warmup=3, repetitions=15, max_seconds=90.0),
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
    B = spec.shape_policy.batch_size or 4
    L = spec.shape_policy.sequence_length or 1500
    dim = int(spec.shape_policy.extra.get("dim", 1280))
    heads = int(spec.shape_policy.extra.get("heads", 20))
    ffn_dim = int(spec.shape_policy.extra.get("ffn_dim", 5120))
    head_dim = dim // heads

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.q = nn.Linear(dim, dim, bias=True)
            self.k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=True)
            self.o = nn.Linear(dim, dim, bias=True)
            self.norm2 = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, ffn_dim)
            self.fc2 = nn.Linear(ffn_dim, dim)

        def forward(self, x):
            h = self.norm1(x)
            q = self.q(h).view(B, L, heads, head_dim).transpose(1, 2)
            k = self.k(h).view(B, L, heads, head_dim).transpose(1, 2)
            v = self.v(h).view(B, L, heads, head_dim).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(1, 2).reshape(B, L, dim)
            x = x + self.o(a)
            x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
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
