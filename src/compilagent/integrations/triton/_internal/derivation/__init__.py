"""Triton-backend search-space derivation plugins.

The project's vision is to **replace compiler heuristics**, not to autotune
user-provided launch metaparameters. Triton's `BLOCK_SIZE`, `num_warps`,
`num_stages`, and `LOAD_CACHE_MODIFIER` are inputs the user (or
`@triton.autotune`) hands to the kernel — they are not compiler decisions —
so we deliberately do NOT expose them as agent levers. The agent's surface
is restricted to the actual MLIR pass pipeline (`pass_impact.py`):
skip / parameterize / replace passes such as `tritongpu-coalesce`,
`tritongpu-pipeline`, `tritongpu-accelerate-matmul`, `tritongpu-optimize-thread-locality`,
`tritongpu-remove-layout-conversions`, etc.

Re-enabling launch-meta levers should require an explicit opt-in flag at the
runtime layer (e.g. for autotune-comparison experiments) — never default.
"""

from __future__ import annotations

from .pass_impact import PASS_DERIVATIONS

ALL_DERIVATIONS = PASS_DERIVATIONS

__all__ = ["ALL_DERIVATIONS", "PASS_DERIVATIONS"]
