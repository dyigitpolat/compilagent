"""Six modern PyTorch/Inductor primitives for the suite.

Each factory returns `(module, example_inputs)` where `module` is a fully
materialised `nn.Module` on CUDA in `bf16` and `example_inputs` is the
positional-arg tuple to pass into `optimize_module`.

Shapes lean on Llama-3 / GPT-style production sizes — small enough that
compile + benchmark cycles stay short, large enough that Inductor
heuristics can move the needle.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- helpers

_DEVICE = "cuda"
_DTYPE = torch.bfloat16


def _seed_eval(module: nn.Module) -> nn.Module:
    return module.to(device=_DEVICE, dtype=_DTYPE).eval()


# ----------------------------------------------------------- 1. RMSNorm

class RMSNorm(nn.Module):
    """Llama-style root-mean-square layer norm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(x.dtype) * self.weight


def build_rmsnorm() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    dim = 4096
    module = _seed_eval(RMSNorm(dim))
    x = torch.randn(8, 1024, dim, device=_DEVICE, dtype=_DTYPE)
    return module, (x,)


# -------------------------------------------------------------- 2. SwiGLU

class SwiGLU(nn.Module):
    """Llama FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def build_swiglu() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    module = _seed_eval(SwiGLU(2048, 5632))
    x = torch.randn(16, 512, 2048, device=_DEVICE, dtype=_DTYPE)
    return module, (x,)


# --------------------------------------------------------- 3. MultiHeadSelfAttention

class MultiHeadSelfAttention(nn.Module):
    """Standard MHA with QKV fused projection + scaled-dot-product attention."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out(out.transpose(1, 2).contiguous().view(b, s, -1))


def build_mha() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    module = _seed_eval(MultiHeadSelfAttention(dim=1024, num_heads=16))
    x = torch.randn(16, 512, 1024, device=_DEVICE, dtype=_DTYPE)
    return module, (x,)


# ---------------------------------------------------- 4. GroupedQueryAttention

class GroupedQueryAttention(nn.Module):
    """Llama-3-style GQA: fewer KV heads than Q heads."""

    def __init__(self, dim: int, q_heads: int, kv_heads: int) -> None:
        super().__init__()
        assert dim % q_heads == 0 and q_heads % kv_heads == 0
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = dim // q_heads
        self.q_proj = nn.Linear(dim, q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        rep = self.q_heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).contiguous().view(b, s, -1))


def build_gqa() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    module = _seed_eval(GroupedQueryAttention(dim=2048, q_heads=16, kv_heads=4))
    x = torch.randn(8, 512, 2048, device=_DEVICE, dtype=_DTYPE)
    return module, (x,)


# ----------------------------------------------------- 5. Rotary attention

def _build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.stack((freqs.cos(), freqs.sin()), dim=-1)


def _apply_rope(x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    cos = cache[..., 0]
    sin = cache[..., 1]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rot.flatten(-2)


class RotaryAttention(nn.Module):
    """Multi-head attention with rotary position embeddings on Q and K."""

    def __init__(self, dim: int, num_heads: int, seq_len: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer(
            "rope_cache",
            _build_rope_cache(seq_len, self.head_dim).to(_DTYPE),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        cache = self.rope_cache[:s].unsqueeze(0).unsqueeze(0)
        q = _apply_rope(q.transpose(1, 2), cache)
        k = _apply_rope(k.transpose(1, 2), cache)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out(out.transpose(1, 2).contiguous().view(b, s, -1))


def build_rotary_attention() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    seq_len = 512
    module = _seed_eval(RotaryAttention(dim=1024, num_heads=16, seq_len=seq_len))
    x = torch.randn(16, seq_len, 1024, device=_DEVICE, dtype=_DTYPE)
    return module, (x,)


# ------------------------------------------------------------- 6. MoE FFN

class MoEFFN(nn.Module):
    """Top-k mixture of SwiGLU experts (Mixtral-style FFN)."""

    def __init__(self, dim: int, hidden: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(num_experts, dim, hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden, dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, dim, hidden))
        for p in (self.w1, self.w2, self.w3):
            nn.init.normal_(p, std=1.0 / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        flat = x.reshape(-1, d)
        scores = self.gate(flat)
        top_w, top_i = scores.topk(self.top_k, dim=-1)
        top_w = top_w.softmax(dim=-1).to(x.dtype)
        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            mask = (top_i == e)
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)
            xe = flat[tok_idx]
            ye = (F.silu(xe @ self.w1[e]) * (xe @ self.w3[e])) @ self.w2[e]
            we = top_w[tok_idx, slot].unsqueeze(-1)
            out.index_add_(0, tok_idx, ye * we)
        return out.view(b, s, d)


def build_moe() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    module = _seed_eval(MoEFFN(dim=1024, hidden=2048, num_experts=8, top_k=2))
    x = torch.randn(8, 256, 1024, device=_DEVICE, dtype=_DTYPE)
    return module, (x,)


# --------------------------------------------------------------- registry

MODULE_BUILDERS: dict[str, Callable[[], tuple[nn.Module, tuple[torch.Tensor, ...]]]] = {
    "rmsnorm": build_rmsnorm,
    "swiglu_mlp": build_swiglu,
    "mha": build_mha,
    "gqa": build_gqa,
    "rotary_attn": build_rotary_attention,
    "moe_ffn": build_moe,
}
