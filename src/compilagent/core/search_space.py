"""Typed levers + a backend-pluggable derivation registry.

A `SearchSpace` is the catalog of candidate axes the agent can pull. Each
`Lever` declares a typed range, a default, and `DerivationEvidence` linking
it to the signal that produced it. **No lever's range is hand-coded** —
every range is derived from workload analysis or device capability.

A backend declares zero or more `SearchSpaceDerivation` plugins. The session
runs every applicable derivation against the workload's `Analysis` and
concatenates the resulting levers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class IntRange:
    """A bounded integer axis with explicit, derived candidate values."""

    candidates: tuple[int, ...]
    units: str = ""

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": "int_range",
            "candidates": list(self.candidates),
            "units": self.units,
        }


@dataclass(frozen=True, slots=True)
class FloatRange:
    candidates: tuple[float, ...]
    units: str = ""

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": "float_range",
            "candidates": list(self.candidates),
            "units": self.units,
        }


@dataclass(frozen=True, slots=True)
class EnumChoice:
    """A finite categorical axis with derived members."""

    candidates: tuple[str, ...]

    def serialize(self) -> dict[str, Any]:
        return {"kind": "enum", "candidates": list(self.candidates)}


@dataclass(frozen=True, slots=True)
class BooleanFlag:
    def serialize(self) -> dict[str, Any]:
        return {"kind": "bool", "candidates": [True, False]}


@dataclass(frozen=True, slots=True)
class StructuredJsonRange:
    """A complex-payload axis (e.g. a layout dict, an FX rewrite spec).

    `examples` is a derived sample of valid payload shapes the agent may
    copy or mutate; the actual validity is checked by the backend.
    """

    examples: tuple[dict[str, Any], ...]
    schema_hint: str = ""

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": "structured_json",
            "examples": list(self.examples),
            "schema_hint": self.schema_hint,
        }


LeverRange = IntRange | FloatRange | EnumChoice | BooleanFlag | StructuredJsonRange


@dataclass(frozen=True, slots=True)
class DerivationEvidence:
    """Tying a lever back to the signal that produced its range."""

    rule: str
    signal: str
    citations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Lever:
    """One typed candidate axis with derived bounds."""

    id: str
    target_kind: str
    target_selector: str
    range: LeverRange
    default: Any
    description: str
    evidence: DerivationEvidence
    backend_id: str

    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend_id": self.backend_id,
            "target": {"kind": self.target_kind, "selector": self.target_selector},
            "range": self.range.serialize(),
            "default": self.default,
            "description": self.description,
            "evidence": {
                "rule": self.evidence.rule,
                "signal": self.evidence.signal,
                "citations": list(self.evidence.citations),
            },
        }


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Concatenated lever list for one workload."""

    workload_id: str
    backend_id: str
    levers: tuple[Lever, ...] = ()

    def serialize(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "backend_id": self.backend_id,
            "levers": [lever.serialize() for lever in self.levers],
        }


class SearchSpaceDerivation(Protocol):
    """A pluggable rule that maps (workload, analysis) → some levers."""

    name: str
    applies_to: tuple[str, ...]

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]: ...


def pow2_around(value: int, *, lo: int = 32, hi: int = 4096, count: int = 5) -> tuple[int, ...]:
    """Return up to `count` power-of-two values centered on `value`, clamped to [lo, hi]."""

    if value <= 0:
        return tuple(sorted({lo, lo * 2, lo * 4, lo * 8})[:count])
    centre = 1 << max(0, int(round(math.log2(value))))
    half = count // 2
    seq: list[int] = []
    for shift in range(-half, count - half):
        v = max(lo, min(hi, centre << shift if shift >= 0 else centre >> -shift))
        if v not in seq:
            seq.append(v)
    return tuple(sorted(seq))


def pow2_range(lo: int, hi: int) -> tuple[int, ...]:
    """All powers of two in [lo, hi]."""

    out: list[int] = []
    v = 1
    while v < lo:
        v <<= 1
    while v <= hi:
        out.append(v)
        v <<= 1
    return tuple(out)
