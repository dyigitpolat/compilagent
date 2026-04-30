"""Forward LayerNorm — classic transformer normalization primitive."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
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
    mean = tl.sum(x, axis=0) / n_cols
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(out_ptr + row * row_stride + cols, y, mask=mask)
