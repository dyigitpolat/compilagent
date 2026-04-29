"""Parse captured Inductor artifacts (output_code, schedule, fusion logs).

Produces a structured `InductorAnalysis` (`Analysis` subclass) that downstream
derivation plugins consume. **No fixed list of inductor kernels is hard-coded
here** — every kernel/decision the analysis enumerates is read off the
captured artifacts produced for the actual workload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compilagent.core.analysis import Analysis

_TRITON_KERNEL_DECL_RE = re.compile(
    r"@triton\.jit\s*\ndef\s+(?P<name>\w+)\s*\(",
    re.MULTILINE,
)
_TRITON_HEURISTICS_BLOCK_RE = re.compile(
    r"triton_heuristics\.(?P<kind>\w+)\(\s*size_hints\s*=\s*(?P<hints>\{[^}]*\})",
)
_KERNEL_LAUNCH_RE = re.compile(
    r"\b(?P<name>triton_\w+)\.run\(",
)
_FUSION_DECISION_RE = re.compile(
    r"fuse[sd]?:?\s*(?P<lhs>\S+)\s*->\s*(?P<rhs>\S+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class InductorKernel:
    name: str
    size_hints: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InductorAnalysisData:
    """Structured view of one inductor compile."""

    kernels: tuple[InductorKernel, ...]
    fusion_decisions: tuple[tuple[str, str], ...]
    output_code_size_bytes: int
    schedule_log_size_bytes: int
    fx_graph_size_bytes: int


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def parse_inductor_artifacts(
    *,
    output_code: Path | None,
    schedule_log: Path | None,
    fx_graph: Path | None,
) -> InductorAnalysisData:
    output_text = _read_text(output_code)
    schedule_text = _read_text(schedule_log)
    fx_text = _read_text(fx_graph)

    kernels: list[InductorKernel] = []
    seen: set[str] = set()
    for match in _TRITON_KERNEL_DECL_RE.finditer(output_text):
        name = match.group("name")
        if name in seen:
            continue
        seen.add(name)
        # Look ahead a few lines for a triton_heuristics block.
        window = output_text[max(0, match.start() - 800) : match.start()]
        hint_match = _TRITON_HEURISTICS_BLOCK_RE.search(window)
        size_hints: dict[str, Any] = {}
        if hint_match:
            try:
                # Heuristics blocks are nominally Python literals — eval safely.
                import ast as _ast
                size_hints = _ast.literal_eval(hint_match.group("hints"))
            except Exception:  # noqa: BLE001
                size_hints = {"_raw": hint_match.group("hints")}
        kernels.append(InductorKernel(name=name, size_hints=size_hints))

    fusion_decisions: list[tuple[str, str]] = []
    for match in _FUSION_DECISION_RE.finditer(schedule_text):
        fusion_decisions.append((match.group("lhs"), match.group("rhs")))

    return InductorAnalysisData(
        kernels=tuple(kernels),
        fusion_decisions=tuple(fusion_decisions),
        output_code_size_bytes=len(output_text),
        schedule_log_size_bytes=len(schedule_text),
        fx_graph_size_bytes=len(fx_text),
    )


def to_generic_analysis(data: InductorAnalysisData, *, workload_kind: str) -> Analysis:
    """Project the inductor-specific view into the generic `Analysis` shape."""

    summary = {
        "kind": workload_kind,
        "tensor_shapes": {},
        "dtypes": [],
        "op_counts": {"inductor_kernels": len(data.kernels)},
        "fusion_decision_count": len(data.fusion_decisions),
        "output_code_size_bytes": data.output_code_size_bytes,
        "schedule_log_size_bytes": data.schedule_log_size_bytes,
    }
    extra = {
        "kernels": [
            {"name": k.name, "size_hints": k.size_hints, "extra": k.extra}
            for k in data.kernels
        ],
        "fusion_decisions": list(data.fusion_decisions),
    }
    return Analysis(summary=summary, extra=extra)
