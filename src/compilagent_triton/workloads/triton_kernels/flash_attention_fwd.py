"""FlashAttention-2 forward kernel — the hot path of every modern transformer.

Tiled, single-pass, online softmax. Each program computes one tile of `[BLOCK_M
queries × BLOCK_N keys]` per head, streaming over the K/V sequence and
accumulating the row-max + row-sum-of-exp on the fly. No causal mask in this
variant — bidirectional attention (encoder-style).

This is the most lever-rich kernel the project sees:
  - The two GEMMs (`Q @ K^T` and `P @ V`) interact through the online softmax,
    so `-tritongpu-accelerate-matmul` and `-tritongpu-pipeline` are tightly
    coupled.
  - `BLOCK_M`, `BLOCK_N` are user constexprs (out of scope), but the compiler
    decides MMA tile / dot-operand layout — exactly what the agent optimizes.
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
def flash_attn_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_q, seq_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = bh * stride_qh
    k_base = bh * stride_kh
    v_base = bh * stride_vh
    o_base = bh * stride_oh

    q_ptrs = Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_q, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    for start_n in range(0, seq_k, BLOCK_N):
        cur_n = start_n + offs_n
        k_ptrs = K + k_base + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=cur_n[:, None] < seq_k, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=cur_n[:, None] < seq_k, other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(cur_n[None, :] < seq_k, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    o = acc / l_i[:, None]
    o_ptrs = Out + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o, mask=offs_m[:, None] < seq_q)


def _compilagent_compile_flash_attn(meta: dict) -> object:
    import math
    import torch
    B = int(meta.get("batch", 2))
    H = int(meta.get("heads", 4))
    L = int(meta.get("seq_len", 1024))
    D = int(meta.get("head_dim", 128))
    BM = int(meta.get("BLOCK_M", 64))
    BN = int(meta.get("BLOCK_N", 64))
    num_warps = int(meta.get("num_warps", 4))
    num_stages = int(meta.get("num_stages", 2))
    dtype = torch.float16
    q = torch.randn(B, H, L, D, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(B, H, L, D, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(B, H, L, D, device="cuda", dtype=dtype).contiguous()
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    sqb, sqh, sqm, sqk = q.stride()
    skb, skh, skn, skk = k.stride()
    svb, svh, svn, svd = v.stride()
    sob, soh, som, sod = out.stride()
    handle = flash_attn_fwd_kernel[(triton.cdiv(L, BM), B * H)](
        q, k, v, out, sm_scale,
        sqb, sqh, sqm, sqk,
        skb, skh, skn, skk,
        svb, svh, svn, svd,
        sob, soh, som, sod,
        L, L,
        BLOCK_M=BM, BLOCK_N=BN, HEAD_DIM=D,
        num_warps=num_warps, num_stages=num_stages,
    )
    torch.cuda.synchronize()
    return handle


flash_attn_fwd_kernel.compilagent_compile = _compilagent_compile_flash_attn


_SPEC = WorkloadSpec(
    id="flash_attention_fwd",
    title="FlashAttention-2 forward (single head)",
    description=(
        "Tiled online-softmax fused attention forward. Inputs `[B*H=8, "
        "L=1024, D=128]` fp16 — single attention head per program, BLOCK_M=64 "
        "BLOCK_N=64. Bidirectional (no causal mask). Two coupled MMAs over the "
        "online-softmax accumulator."
    ),
    kind=WorkloadKind.KERNEL,
    backend_id="triton",
    entrypoint="compilagent_triton.workloads.triton_kernels.flash_attention_fwd:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="fp16", param_dtype="fp16"),
    shape_policy=ShapePolicy(
        batch_size=2, sequence_length=1024,
        extra={"heads": 4, "head_dim": 128},
    ),
    tolerance=ToleranceConfig(atol=5e-3, rtol=5e-2,
                              notes="online softmax accumulates fp error; tolerance loose."),
    budget=BenchmarkBudget(warmup=5, repetitions=15, max_seconds=120.0),
    metadata={
        "kernel_symbol": "flash_attn_fwd_kernel",
        "source_path": __file__,
        "block_m": 64,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
)


@register_workload(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import math
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(int(spec.metadata.get("seed", 0)))
    B = spec.shape_policy.batch_size or 2
    H = int(spec.shape_policy.extra.get("heads", 4))
    L = spec.shape_policy.sequence_length or 1024
    D = int(spec.shape_policy.extra.get("head_dim", 128))
    BM = int(spec.metadata.get("block_m", 64))
    BN = int(spec.metadata.get("block_n", 64))
    num_warps = int(spec.metadata.get("num_warps", 4))
    num_stages = int(spec.metadata.get("num_stages", 2))
    dtype = torch.float16

    q = torch.randn(B, H, L, D, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(B, H, L, D, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(B, H, L, D, device="cuda", dtype=dtype).contiguous()
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)

    sqb, sqh, sqm, sqk = q.stride()
    skb, skh, skn, skk = k.stride()
    svb, svh, svn, svd = v.stride()
    sob, soh, som, sod = out.stride()

    def forward():
        flash_attn_fwd_kernel[(triton.cdiv(L, BM), B * H)](
            q, k, v, out, sm_scale,
            sqb, sqh, sqm, sqk,
            skb, skh, skn, skk,
            svb, svh, svn, svd,
            sob, soh, som, sod,
            L, L,
            BLOCK_M=BM, BLOCK_N=BN, HEAD_DIM=D,
            num_warps=num_warps, num_stages=num_stages,
        )
        torch.cuda.synchronize()
        return out

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=(q, k, v),
        metadata={"output_buffer": out, "head_dim": D, "seq_len": L},
    )
