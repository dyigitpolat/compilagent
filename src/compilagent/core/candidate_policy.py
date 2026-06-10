"""Optional cross-run memory consultor.

The session calls `policy.consult(...)` once during bootstrap to obtain
`PolicyHint`s — interventions a previous run found promising for a similar
workload. Hints are surfaced through `inspect_workload` so the agent can
factor them into its hypotheses.

Policies MAY also implement the optional `observe(...)` hook (E9): the
session calls it after every `run_candidate` outcome so a stateful policy
can learn online (distill failure rules, update frequency counters, …).
The hook is duck-typed — `consult` stays the only required Protocol member,
so existing consult-only policies remain conformant; the session dispatches
`observe` via `getattr` and swallows policy exceptions (a learning bug must
never break the optimization run). `NullPolicy.observe` is the default
no-op, making `NullPolicy` a convenient base class.

`NullPolicy` (default) returns no hints. An `ExperimentLogPolicy` reading
from `storage.experiment_log.ExperimentLog` ships in
`integrations.archetype_harnesses.skill_memory`. Backends opt in by
implementing `Backend.infer_workload_family` so the policy can correlate
across runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .analysis import Analysis, CompileResult, CorrectnessResult, TimingResult
from .plan import Intervention, Plan
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

        OPTIONAL `observe` hook (not a Protocol member — see module
        docstring): policies that also define ::

            def observe(self, *, workload, candidate_id, plan,
                        compile_result, timing, correctness, speedup,
                        successful, family, arch) -> None

        receive every candidate outcome right after `run_candidate`
        evaluates it. This is the E9 update path that turns the
        consult-once policy into a closed loop (e.g. CASCADE's C6 skill
        memory distills failure rules from these observations).
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

    def observe(
        self,
        *,
        workload: WorkloadSpec,
        candidate_id: str,
        plan: Plan,
        compile_result: CompileResult,
        timing: TimingResult | None,
        correctness: CorrectnessResult | None,
        speedup: float | None,
        successful: bool,
        family: str | None,
        arch: str,
    ) -> None:
        """Default no-op observation hook (see Protocol docstring)."""
        return None
