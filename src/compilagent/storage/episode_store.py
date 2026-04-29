"""`EpisodeStore` — per-run snapshot of an optimization session.

Persists the closed session as one JSON document: workload spec, baseline
timing, leaderboard rows, and a list of run candidate summaries. The shape
is intentionally a plain dict, not a typed Pydantic schema — backends and
frontends can attach domain-specific fields without forcing core changes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EpisodeStore:
    """File-backed storage for one optimization episode."""

    path: Path

    def save(self, payload: Mapping[str, Any]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, default=str)
        return self.path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        return data
