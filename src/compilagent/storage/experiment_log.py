"""`ExperimentLog` — append-only JSONL of cross-run outcomes.

Every successful candidate (compile OK + valid timing + correctness within
tolerance) is appended as one line. Failures are also persisted so a
`CandidatePolicy` can avoid re-proposing combinations that previously OOM'd
or drifted out of tolerance. Reads are filtered by workload_id, backend_id,
and architecture so a hint targets the same context.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentLog:
    """Append-only JSONL log under `<workspace>/memory/experiments.jsonl`."""

    root: Path

    @property
    def path(self) -> Path:
        return self.root / "memory" / "experiments.jsonl"

    def append(self, row: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            entry = {**row}
            entry.setdefault("timestamp", time.time())
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            return None

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            return []
        return rows

    def recall(
        self,
        *,
        workload_id: str | None = None,
        backend_id: str | None = None,
        family: str | None = None,
        arch: str | None = None,
        successful_only: bool = True,
        top_n: int = 8,
    ) -> list[dict[str, Any]]:
        rows = self.read_all()
        out: list[dict[str, Any]] = []
        for r in rows:
            if successful_only and not r.get("successful", False):
                continue
            if workload_id is not None and r.get("workload_id") != workload_id:
                continue
            if backend_id is not None and r.get("backend_id") != backend_id:
                continue
            if family is not None and r.get("family") != family:
                continue
            if arch is not None and r.get("arch") != arch:
                continue
            out.append(r)
        out.sort(key=lambda r: r.get("speedup", 0.0) or 0.0, reverse=True)
        return out[:top_n]

    def recall_failures(
        self,
        *,
        workload_id: str | None = None,
        backend_id: str | None = None,
        top_n: int = 8,
    ) -> list[dict[str, Any]]:
        rows = self.read_all()
        out = [
            r
            for r in rows
            if not r.get("successful", True)
            and (workload_id is None or r.get("workload_id") == workload_id)
            and (backend_id is None or r.get("backend_id") == backend_id)
        ]
        out.sort(key=lambda r: r.get("timestamp", 0.0) or 0.0, reverse=True)
        return out[:top_n]
