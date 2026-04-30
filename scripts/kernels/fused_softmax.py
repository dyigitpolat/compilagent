"""Row-wise fused softmax — production primitive for attention scores."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def fused_softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = x_ptr + row * row_stride
    x = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    z = tl.exp(x - x_max)
    denom = tl.sum(z, axis=0)
    tl.store(out_ptr + row * row_stride + col_offsets, z / denom, mask=mask)
