"""`ObservationSink` protocol and `NullSink` implementation.

Sessions emit through whatever sink is wired in. The default is
`storage.trace_store.TraceStore` (JSONL append). Tests use `NullSink` (or a
list-capturing impl); production deployments may stack a `TraceStore` and a
fan-out queue that pushes to a WebSocket.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .events import EventKind, ObservationEvent


@runtime_checkable
class ObservationSink(Protocol):
    """Where session events go.

    Production: `TraceStore` (JSONL append). Tests: `NullSink` or
    `CapturingSink`. Custom: a fan-out queue that writes to disk *and*
    pushes to a WebSocket — implement `emit` and `emit_kv` and pass the
    instance into `OptimizationSession(sink=...)`.

    Sinks MUST NOT raise. Persistence failure is the sink's problem to
    swallow / log; the session relies on `emit` returning normally.
    """

    def emit(self, event: ObservationEvent) -> None:
        """Persist or fan out one fully-formed event."""
        ...

    def emit_kv(
        self,
        kind: EventKind | str,
        *,
        payload: Mapping[str, object] | None = None,
        artifact_paths: Sequence[Path] | Sequence[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        """Convenience: build an `ObservationEvent` from kwargs and emit it."""
        ...


class NullSink:
    """Sink that drops every event. Useful in tests."""

    def emit(self, event: ObservationEvent) -> None:
        return None

    def emit_kv(
        self,
        kind: EventKind | str,
        *,
        payload: Mapping[str, object] | None = None,
        artifact_paths: Sequence[Path] | Sequence[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        return None


class CapturingSink:
    """In-memory sink that records every event. Convenient for tests."""

    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def emit(self, event: ObservationEvent) -> None:
        self.events.append(event)

    def emit_kv(
        self,
        kind: EventKind | str,
        *,
        payload: Mapping[str, object] | None = None,
        artifact_paths: Sequence[Path] | Sequence[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        paths = tuple(str(p) for p in (artifact_paths or ()))
        self.events.append(
            ObservationEvent.make(
                kind,
                session_id=session_id,
                run_id=run_id,
                candidate_id=candidate_id,
                payload=payload,
                artifact_paths=paths,
            )
        )

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]
