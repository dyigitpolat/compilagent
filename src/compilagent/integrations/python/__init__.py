"""User-facing optimization entry points (`optimize_module` / `optimize_kernel`).

These are convenience APIs only. A sophisticated caller can always construct
a `WorkloadSpec` + `OptimizationSession` directly. The functions live in an
integration so the core stays free of Triton / PyTorch knowledge.
"""

from __future__ import annotations

from .api import optimize_kernel, optimize_module
from .result import OptimizationResult

__all__ = ["OptimizationResult", "optimize_kernel", "optimize_module"]
