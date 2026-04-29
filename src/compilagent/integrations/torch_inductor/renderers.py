"""Artifact renderers for TorchInductor's compile artifacts.

Inductor produces:
  - `*.fx_graph` — printed FX module text.
  - `*.output_code` — generated Python (Triton kernel + scheduling code).
  - `*.schedule_log`, `*.fusion_log` — scheduler traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from compilagent.observation.artifacts import ArtifactPreview


def _read_text(path: Path, *, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unreadable: {exc!r}>"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... <truncated {len(text) - max_chars} chars>"
    return text


@dataclass(frozen=True, slots=True)
class FxGraphRenderer:
    suffixes: tuple[str, ...] = (".fx_graph",)
    priority: int = 50

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        return ArtifactPreview(
            kind="code",
            language="python",
            text=_read_text(path, max_chars=max_chars),
        )


@dataclass(frozen=True, slots=True)
class OutputCodeRenderer:
    suffixes: tuple[str, ...] = (".output_code",)
    priority: int = 50

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        return ArtifactPreview(
            kind="code",
            language="python",
            text=_read_text(path, max_chars=max_chars),
        )


@dataclass(frozen=True, slots=True)
class SchedulerLogRenderer:
    suffixes: tuple[str, ...] = (".schedule_log", ".fusion_log")
    priority: int = 50

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        return ArtifactPreview(
            kind="text",
            language="text",
            text=_read_text(path, max_chars=max_chars),
        )
