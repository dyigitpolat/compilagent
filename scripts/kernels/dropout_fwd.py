"""Forward dropout with seeded Philox RNG — training-time primitive."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def dropout_fwd_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    r = tl.rand(seed, offsets)
    keep = r > p
    scale = 1.0 / (1.0 - p)
    y = tl.where(keep, x * scale, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)
