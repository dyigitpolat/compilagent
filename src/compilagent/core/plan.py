"""Backend-agnostic candidate representation.

A `Plan` is an ordered list of `Intervention`s — one candidate compile. Each
intervention names a `Target` (a free-string `kind` plus a `selector`) and
carries an opaque `payload` that the backend interprets. The core never
inspects payloads; only `Backend.validate_intervention` and
`Backend.apply_intervention` do.

There is intentionally no enum of candidate kinds. Per-domain vocabularies
(MLIR pass actions, Inductor knob paths, FX rewrites, scheduler callbacks)
live in the backend's `Target.kind` strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Target:
    """Where in the compile pipeline an intervention applies.

    `kind` is a free string interpreted by the backend (e.g., "launch",
    "pass", "knob", "lowering", "fx_node", "kernel_src", "scheduler").
    `selector` further locates the target within the kind.
    """

    kind: str
    selector: str = ""

    def __str__(self) -> str:
        return f"{self.kind}({self.selector})" if self.selector else self.kind


@dataclass(frozen=True, slots=True)
class Intervention:
    """One concrete change an agent proposes for a candidate compile."""

    target: Target
    payload: Any
    rationale: str = ""

    def serialize(self) -> dict[str, Any]:
        return {
            "target": {"kind": self.target.kind, "selector": self.target.selector},
            "payload": self.payload,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class Plan:
    """An ordered tuple of interventions making up a single candidate compile."""

    interventions: tuple[Intervention, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.interventions

    def serialize(self) -> list[dict[str, Any]]:
        return [iv.serialize() for iv in self.interventions]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of `Backend.validate_intervention`.

    On `ok=False` the session turns `errors` into a `ValueError` so the
    harness surfaces it as a model retry.
    """

    ok: bool
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
