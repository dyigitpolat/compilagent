"""Outcome dataclasses every backend produces.

`DeviceCapability`, `Analysis`, `CompileResult`, `TimingResult`,
`CorrectnessResult`, `PassEvent` are the typed surface a backend returns;
the session reads these dataclasses uniformly without ever branching on
backend identity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DeviceCapability:
    """Hardware envelope a backend will compile against."""

    arch: str
    capability_int: int | None
    name: str
    memory_total_bytes: int | None
    memory_peak_bandwidth_gbps: float | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    """Outcome of comparing baseline vs. candidate output tensors."""

    ok: bool
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    p99_abs_diff: float | None = None
    failed_at: str | None = None
    diagnostics: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Analysis:
    """Backend-specific structured introspection of a workload + baseline.

    `summary` is the generic surface every backend MUST populate so the
    session can show a backend-agnostic snapshot. `extra` is opaque.

    Recommended `summary` keys (advisory, not enforced):
      - "kind": workload kind ("kernel"|"fused_subgraph"|"full_model")
      - "tensor_shapes": dict[str, list[int]]
      - "dtypes": list[str]
      - "op_counts": dict[str, int]
    """

    summary: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompileResult:
    """The single shape every backend returns from `Backend.compile`.

    `compiled_callable` is whatever the backend produces for `time_workload`
    to drive — a torch.compile-wrapped callable, a Triton kernel handle, a
    raw Python function. Backends round-trip metadata via `metadata`; the
    session never reaches in to extract a backend-specific path.
    """

    ok: bool
    elapsed_ms: float | None = None
    artifacts: tuple[Path, ...] = ()
    compiled_callable: Any | None = None
    diagnostics: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PassEvent:
    """A single executed compile step the backend wants to surface live.

    Streamed to the observation sink so the UI can render a per-pass timeline
    regardless of backend (Triton MLIR pass, Inductor scheduler pass,
    config-knob application, ...).
    """

    stage: str
    name: str
    duration_ms: float
    action: str = "run"
    args: list[Any] = field(default_factory=list)
    ir_after_size: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


PassCallback = Callable[[PassEvent], None]


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Raw end-to-end timing samples from `Backend.time_workload`."""

    timings_ms: tuple[float, ...]
    median_ms: float | None
    p20_ms: float | None
    p80_ms: float | None
    profile_metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: str | None = None
