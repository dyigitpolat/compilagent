"""Forward RMSNorm — Llama-family normalization primitive."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_cols,
    row_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row * row_stride + cols, y, mask=mask)
