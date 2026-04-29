"""`OptimizationResult` — what `optimize_module` / `optimize_kernel` return."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OptimizationResult:
    """Public result of `optimize_module` / `optimize_kernel`.

    `optimized_callable` is the drop-in replacement: call it with the same
    inputs you passed to the optimizer and you get the optimized forward
    pass / kernel launch. It is `None` iff no validated candidate beat the
    baseline; in that case `improved` is False and the caller should fall
    back to their original code path.
    """

    workload_id: str
    backend_id: str
    harness: str
    baseline_median_ms: float | None
    best_speedup: float | None
    best_candidate_id: str | None
    best_median_ms: float | None
    correctness_ok: bool | None
    max_abs_diff: float | None
    final_text: str | None
    elapsed_ms: float
    candidates: list[dict[str, Any]] = field(default_factory=list)
    workspace_root: Path | None = None
    optimized_callable: Any | None = None
    best_plan: Any | None = None

    @property
    def improved(self) -> bool:
        """True iff a validated candidate beat baseline."""

        return self.optimized_callable is not None and bool(
            isinstance(self.best_speedup, (int, float)) and self.best_speedup > 1.0
        )

    def __call__(self, *args, **kwargs):
        if self.optimized_callable is None:
            raise RuntimeError(
                "No validated candidate beat baseline within tolerance. "
                "Use your original code path; `result.improved` is False."
            )
        return self.optimized_callable(*args, **kwargs)
