"""Optional cross-run memory consultor.

The session calls `policy.consult(...)` once during bootstrap to obtain
`PolicyHint`s — interventions a previous run found promising for a similar
workload. Hints are surfaced through `inspect_workload` so the agent can
factor them into its hypotheses.

`NullPolicy` (default) returns no hints. Phase-2 may ship an
`ExperimentLogPolicy` reading from `storage.experiment_log.ExperimentLog`.
Backends opt in by implementing `Backend.infer_workload_family` so the
policy can correlate across runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .analysis import Analysis
from .plan import Intervention
from .workload import WorkloadSpec


@dataclass(frozen=True, slots=True)
class PolicyHint:
    """One suggestion from a `CandidatePolicy`."""

    suggested_interventions: tuple[Intervention, ...]
    rationale: str
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class CandidatePolicy(Protocol):
    """Cross-run memory consultor.

    Called once during session bootstrap (after `analyze` and before the
    agent loop begins). Returned hints are surfaced through
    `inspect_workload`'s `prior_hints` field so the agent can factor them
    into its first proposals.

    Implementations are free to be stateful (read from disk, talk to a
    service, query a model). The session never mutates the policy; it only
    reads `consult`.
    """

    name: str
    """Stable string id for telemetry (`"null"`, `"experiment_log"`, ...)."""

    def consult(
        self,
        *,
        workload: WorkloadSpec,
        analysis: Analysis,
        family: str | None,
        arch: str,
    ) -> Sequence[PolicyHint]:
        """Return any prior hints relevant to this workload+arch+family.

        Empty sequence is the common case for cold-start runs and for
        workloads the policy has no record of. The session treats hints as
        advisory: nothing happens automatically — the agent decides whether
        to incorporate them.
        """
        ...


class NullPolicy:
    """Policy that returns no hints. Default for sessions."""

    name = "null"

    def consult(
        self,
        *,
        workload: WorkloadSpec,
        analysis: Analysis,
        family: str | None,
        arch: str,
    ) -> Sequence[PolicyHint]:
        return ()
