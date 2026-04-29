from __future__ import annotations

import re

from .schemas import DecisionKind, DecisionTrace

_LOAD_STORE_RE = re.compile(r"\btt\.(load|store)\b")
_DOT_RE = re.compile(r"\btt\.dot(?:_scaled)?\b|\bttng\.tc_gen5_mma\b")
_ENCODING_RE = re.compile(r"#(?:ttg|triton_gpu)\.[A-Za-z0-9_]+<([^>]*)>")
_NUM_WARPS_RE = re.compile(r'"ttg\.num-warps"\s*=\s*(\d+)')
_THREADS_PER_WARP_RE = re.compile(r'"ttg\.threads-per-warp"\s*=\s*(\d+)')
_TENSOR_SHAPE_RE = re.compile(r"tensor<([0-9x]+)x")
_MASK_RE = re.compile(r"\bmask\s*=")
_VECTOR_WIDTH_RE = re.compile(r"vec(?:tor)?[_-]?width\s*=?\s*(\d+)|x(\d+)x")
_MEMDESC_RE = re.compile(r"!ttg\.memdesc<([^>]*)>|memdesc<([^>]*)>")
_SWIZZLE_RE = re.compile(r"swizzl\w+\s*=\s*([^,\s>]+)")
_LOCAL_ALLOC_RE = re.compile(r"\bttg\.local_alloc\b")


def extract_decision_traces(
    ir_text: str,
    *,
    run_id: str | None = None,
    max_evidence_chars: int = 800,
) -> list[DecisionTrace]:
    """Extract bootstrap decision traces from TTGIR-like text.

    This is intentionally conservative. A plugin-emitted JSON trace should
    replace it as the authoritative source once compiler-side instrumentation
    exists.
    """

    lines = ir_text.splitlines()
    num_warps = _first_int(_NUM_WARPS_RE, ir_text)
    threads_per_warp = _first_int(_THREADS_PER_WARP_RE, ir_text)
    traces: list[DecisionTrace] = []
    for index, line in enumerate(lines, start=1):
        if _LOAD_STORE_RE.search(line):
            traces.append(
                DecisionTrace(
                    run_id=run_id,
                    kind=DecisionKind.COALESCING,
                    op_name=_LOAD_STORE_RE.search(line).group(0),  # type: ignore[union-attr]
                    op_location=f"line:{index}",
                    tensor_shape=_shape_from_line(line),
                    num_warps=num_warps,
                    threads_per_warp=threads_per_warp,
                    evidence=_truncate(line.strip(), max_evidence_chars),
                    metadata=_memory_metadata(line),
                )
            )
        elif _DOT_RE.search(line):
            traces.append(
                DecisionTrace(
                    run_id=run_id,
                    kind=DecisionKind.MATMUL,
                    op_name="tt.dot",
                    op_location=f"line:{index}",
                    tensor_shape=_shape_from_line(line),
                    num_warps=num_warps,
                    threads_per_warp=threads_per_warp,
                    evidence=_truncate(line.strip(), max_evidence_chars),
                    metadata=_memory_metadata(line),
                )
            )
        elif _LOCAL_ALLOC_RE.search(line):
            traces.append(
                DecisionTrace(
                    run_id=run_id,
                    kind=DecisionKind.COALESCING,
                    op_name="ttg.local_alloc",
                    op_location=f"line:{index}",
                    tensor_shape=_shape_from_line(line),
                    num_warps=num_warps,
                    threads_per_warp=threads_per_warp,
                    evidence=_truncate(line.strip(), max_evidence_chars),
                    metadata=_memory_metadata(line) | {"local_alloc": True},
                )
            )
    return traces


def summarize_decision_traces(traces: list[DecisionTrace]) -> str:
    if not traces:
        return "No coalescing or matmul decisions were detected in the provided IR."
    coalescing = sum(trace.kind == DecisionKind.COALESCING for trace in traces)
    matmul = sum(trace.kind == DecisionKind.MATMUL for trace in traces)
    lines = [
        "Decision trace summary:",
        f"- coalescing-related memory ops: {coalescing}",
        f"- matmul-related ops: {matmul}",
    ]
    for trace in traces[:10]:
        lines.append(f"- `{trace.kind.value}` at `{trace.op_location}`: {trace.evidence[:160]}")
    if len(traces) > 10:
        lines.append(f"- ... {len(traces) - 10} more traces omitted")
    return "\n".join(lines)


def _first_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    if match is None:
        return None
    return int(match.group(1))


def _shape_from_line(line: str) -> tuple[int, ...] | None:
    match = _TENSOR_SHAPE_RE.search(line)
    if match is None:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("x") if part)
    except ValueError:
        return None


def _memory_metadata(line: str) -> dict[str, object]:
    vector_match = _VECTOR_WIDTH_RE.search(line)
    vector_width = None
    if vector_match is not None:
        raw = next((group for group in vector_match.groups() if group), None)
        vector_width = int(raw) if raw is not None else None
    memdesc = [item for match in _MEMDESC_RE.findall(line) for item in match if item]
    return {
        "encodings": _ENCODING_RE.findall(line),
        "has_mask": bool(_MASK_RE.search(line)),
        "memdesc": memdesc,
        "swizzles": _SWIZZLE_RE.findall(line),
        "vector_width": vector_width,
    }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"
