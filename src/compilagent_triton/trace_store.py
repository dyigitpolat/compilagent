from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .events import ObservationEvent


@dataclass(slots=True)
class TraceStore:
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

    def append(self, event: ObservationEvent) -> ObservationEvent:
        self.ensure()
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json(exclude_none=True) + "\n")
        return event

    def emit(
        self,
        kind: str,
        *,
        session_id: str | None = None,
        episode_id: str | None = None,
        payload: dict | None = None,
        artifact_paths: list[str] | None = None,
    ) -> ObservationEvent:
        return self.append(
            ObservationEvent.create(
                kind,
                session_id=session_id,
                episode_id=episode_id,
                payload=payload,
                artifact_paths=artifact_paths,
            )
        )

    def read_events(
        self,
        *,
        session_id: str | None = None,
        episode_id: str | None = None,
        after: str | None = None,
        limit: int | None = None,
    ) -> list[ObservationEvent]:
        if not self.events_path.exists():
            return []
        events: list[ObservationEvent] = []
        seen_after = after is None
        # Read as binary then decode — `read_text` uses IncrementalNewline-
        # Decoder which can hit "decoder not initialised" errors on Python
        # 3.12 when the file is concurrently appended to (the trace store
        # always is). Decoding once on bytes is robust and equally fast.
        for line in self.events_path.read_bytes().decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            event = ObservationEvent.model_validate_json(line)
            if not seen_after:
                seen_after = event.event_id == after
                continue
            if session_id is not None and event.session_id != session_id:
                continue
            if episode_id is not None and event.episode_id != episode_id:
                continue
            events.append(event)
        if limit is not None and limit >= 0:
            return events[-limit:]
        return events

    def tail_logs(self, *, limit: int = 200) -> list[str]:
        return [
            event.model_dump_json(exclude_none=True)
            for event in self.read_events(limit=limit)
            if event.kind in {"log.line", "tool.failed", "benchmark.completed"}
        ]
