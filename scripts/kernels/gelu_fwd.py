"""Forward GELU (tanh approximation) — production activation primitive."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def gelu_fwd_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    c0 = 0.7978845608028654   # sqrt(2/pi)
    c1 = 0.044715
    inner = c0 * (x + c1 * x * x * x)
    y = 0.5 * x * (1.0 + libdevice.tanh(inner))
    tl.store(out_ptr + offsets, y, mask=mask)
