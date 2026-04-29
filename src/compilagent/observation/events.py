"""Observation events emitted by the session.

Every observable beat in an optimization session — baseline timed, candidate
proposed, compile pass executed, agent thinking delta, leaderboard updated —
flows through `ObservationSink.emit(ObservationEvent(...))`. Both the
default `TraceStore` (JSONL) and any custom sink (e.g. an in-memory queue
that fans out to a WebSocket) consume the same shape.

`EventKind` enumerates the canonical event names; arbitrary string kinds are
also accepted for backend-specific events. Secrets in payloads are redacted
before persistence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import time
from typing import Any


class EventKind(StrEnum):
    SESSION_STARTED = "session.started"
    SESSION_FINISHED = "session.finished"
    SESSION_FAILED = "session.failed"

    COMPILE_STARTED = "compile.started"
    COMPILE_COMPLETED = "compile.completed"
    COMPILER_PASS = "compiler.pass"
    ARTIFACT_CREATED = "artifact.created"

    BENCHMARK_STARTED = "benchmark.started"
    BENCHMARK_COMPLETED = "benchmark.completed"

    SEARCH_SPACE_DERIVED = "search_space.derived"

    CANDIDATE_PROPOSED = "candidate.proposed"
    CANDIDATE_VALIDATED = "candidate.validated"
    CANDIDATE_REJECTED = "candidate.rejected"

    RUN_PROGRESS = "run.progress"
    LEADERBOARD_UPDATED = "leaderboard.updated"

    AGENT_THINKING_STARTED = "agent.thinking.started"
    AGENT_THINKING_DELTA = "agent.thinking.delta"
    AGENT_TEXT_STARTED = "agent.text.started"
    AGENT_TEXT_DELTA = "agent.text.delta"

    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"

    LOG_LINE = "log.line"


_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|authorization|bearer)",
    re.IGNORECASE,
)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<redacted>" if _SECRET_KEY_RE.search(k) else _redact_value(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v) for v in value)
    return value


def redact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort secret redaction for nested mappings/sequences."""

    return _redact_value(dict(payload))


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    """One event captured in the trace stream."""

    kind: str
    timestamp: float = field(default_factory=time)
    session_id: str | None = None
    run_id: str | None = None
    candidate_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    artifact_paths: tuple[str, ...] = ()

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "payload": redact_payload(self.payload),
            "artifact_paths": list(self.artifact_paths),
        }

    @classmethod
    def make(
        cls,
        kind: EventKind | str,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        candidate_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        artifact_paths: tuple[str, ...] | tuple[Path, ...] | None = None,
    ) -> ObservationEvent:
        kind_str = kind.value if isinstance(kind, EventKind) else str(kind)
        paths: tuple[str, ...] = tuple(str(p) for p in (artifact_paths or ()))
        return cls(
            kind=kind_str,
            session_id=session_id,
            run_id=run_id,
            candidate_id=candidate_id,
            payload=dict(payload or {}),
            artifact_paths=paths,
        )
