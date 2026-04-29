"""TorchInductor search-space derivation plugins.

Each module exports `DERIVATIONS: tuple[SearchSpaceDerivation, ...]`. The
backend runs every applicable rule and concatenates the resulting levers —
no static lever list anywhere.
"""

from __future__ import annotations

from .fx_rewrites import FX_REWRITE_DERIVATIONS
from .inductor_knobs import KNOB_DERIVATIONS
from .lowering_overrides import LOWERING_DERIVATIONS
from .per_kernel_tuning import PER_KERNEL_DERIVATIONS

ALL_DERIVATIONS = (
    KNOB_DERIVATIONS
    + PER_KERNEL_DERIVATIONS
    + FX_REWRITE_DERIVATIONS
    + LOWERING_DERIVATIONS
)

__all__ = [
    "ALL_DERIVATIONS",
    "KNOB_DERIVATIONS",
    "PER_KERNEL_DERIVATIONS",
    "FX_REWRITE_DERIVATIONS",
    "LOWERING_DERIVATIONS",
]
