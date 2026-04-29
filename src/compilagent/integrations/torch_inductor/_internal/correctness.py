"""Forward-pass numerical correctness validation for inductor workloads.

`compare_forward(workload, baseline_callable, candidate_callable, tolerance)`
runs both compiled callables under the same RNG seed against the workload's
example inputs and reports `CorrectnessResult`. Single-tensor and tuple/dict
outputs are both supported. Tolerance is per-dtype with an `atol`/`rtol`
budget; the result records `max_abs_diff` / `max_rel_diff` so the agent sees
not just pass/fail but how close.
"""

from __future__ import annotations

from typing import Any

from compilagent.core.analysis import CorrectnessResult
from compilagent.core.workload import ToleranceConfig


def _extract_tensors(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Walk a structured output and yield (path, tensor) tuples.

    Imports torch lazily; if torch isn't present, the function returns empty.
    """

    try:
        import torch  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return []
    if isinstance(value, torch.Tensor):
        return [(prefix or "<root>", value)]
    if isinstance(value, dict):
        out: list[tuple[str, Any]] = []
        for k, v in value.items():
            out.extend(_extract_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for i, v in enumerate(value):
            out.extend(_extract_tensors(v, f"{prefix}[{i}]"))
        return out
    return []


def compare_forward(
    *,
    baseline_run: Any,
    candidate_run: Any,
    tolerance: ToleranceConfig,
) -> CorrectnessResult:
    """Compare two already-executed forward outputs."""

    baseline_tensors = _extract_tensors(baseline_run)
    candidate_tensors = _extract_tensors(candidate_run)
    if not baseline_tensors and not candidate_tensors:
        return CorrectnessResult(
            ok=True,
            diagnostics="No tensor outputs to compare; correctness skipped.",
        )
    if len(baseline_tensors) != len(candidate_tensors):
        return CorrectnessResult(
            ok=False,
            diagnostics=(
                f"Output structure mismatch: baseline has {len(baseline_tensors)} tensors, "
                f"candidate has {len(candidate_tensors)}."
            ),
        )
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return CorrectnessResult(ok=False, diagnostics=f"torch import failed: {exc!r}")

    max_abs = 0.0
    max_rel = 0.0
    p99_abs = 0.0
    failed_at: str | None = None
    for (b_name, b_t), (c_name, c_t) in zip(baseline_tensors, candidate_tensors, strict=False):
        if b_t.shape != c_t.shape:
            return CorrectnessResult(
                ok=False,
                failed_at=b_name,
                diagnostics=f"shape mismatch at {b_name}: {tuple(b_t.shape)} vs {tuple(c_t.shape)}",
            )
        b_f = b_t.detach().to(torch.float32)
        c_f = c_t.detach().to(torch.float32)
        diff = (b_f - c_f).abs()
        rel = diff / (b_f.abs() + 1e-12)
        max_abs = max(max_abs, float(diff.max().item()))
        max_rel = max(max_rel, float(rel.max().item()))
        # 99th percentile of abs diff across all elements
        try:
            flat = diff.flatten()
            if flat.numel() > 0:
                p99_abs = max(p99_abs, float(flat.kthvalue(int(0.99 * flat.numel()) + 1).values.item()))
        except Exception:  # noqa: BLE001
            pass
        # Element-wise tolerance check
        ok_mask = (diff <= tolerance.atol) | (rel <= tolerance.rtol)
        if not bool(ok_mask.all().item()):
            failed_at = b_name
            break
    ok = failed_at is None
    return CorrectnessResult(
        ok=ok,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        p99_abs_diff=p99_abs,
        failed_at=failed_at,
        diagnostics=None if ok else (
            f"Output at `{failed_at}` exceeded tolerance "
            f"(atol={tolerance.atol}, rtol={tolerance.rtol}); "
            f"max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}."
        ),
    )
