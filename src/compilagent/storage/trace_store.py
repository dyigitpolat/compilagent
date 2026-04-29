"""`TraceStore` — JSONL-backed default `ObservationSink`.

Append-only file at `<workspace.root>/traces/events.jsonl`. The store reads
robustly (decoded once on bytes) so concurrent writes don't trip the
incremental-newline decoder.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from compilagent.observation.events import EventKind, ObservationEvent


@dataclass(slots=True)
class TraceStore:
    """JSONL-backed `ObservationSink` implementation."""

    root: Path

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    @property
    def events_path(self) -> Path:
        return self.traces_dir / "events.jsonl"

    def ensure(self) -> TraceStore:
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        return self

    def emit(self, event: ObservationEvent) -> None:
        self.ensure()
        line = json.dumps(event.serialize(), default=str)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

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
        self.emit(
            ObservationEvent.make(
                kind,
                session_id=session_id,
                run_id=run_id,
                candidate_id=candidate_id,
                payload=payload,
                artifact_paths=paths,
            )
        )

    def read_events(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        kinds: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[ObservationEvent]:
        if not self.events_path.exists():
            return []
        events: list[ObservationEvent] = []
        for line in self.events_path.read_bytes().decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_id is not None and row.get("session_id") != session_id:
                continue
            if run_id is not None and row.get("run_id") != run_id:
                continue
            if kinds is not None and row.get("kind") not in kinds:
                continue
            events.append(
                ObservationEvent(
                    kind=row.get("kind", ""),
                    timestamp=float(row.get("timestamp", 0.0)),
                    session_id=row.get("session_id"),
                    run_id=row.get("run_id"),
                    candidate_id=row.get("candidate_id"),
                    payload=dict(row.get("payload", {}) or {}),
                    artifact_paths=tuple(row.get("artifact_paths", []) or ()),
                )
            )
        if limit is not None and limit >= 0:
            return events[-limit:]
        return events
