"""Artifact renderers for Triton's IR stages.

The Triton backend's `list_artifact_renderers()` returns these so the
observation UI can render `.ttir`/`.ttgir`/`.llir`/`.ptx` files without any
suffix table in the UI itself.
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
class MlirRenderer:
    suffixes: tuple[str, ...] = (".ttir", ".ttgir", ".llir", ".mlir")
    priority: int = 50

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        return ArtifactPreview(
            kind="code",
            language="mlir",
            text=_read_text(path, max_chars=max_chars),
        )


@dataclass(frozen=True, slots=True)
class PtxRenderer:
    suffixes: tuple[str, ...] = (".ptx",)
    priority: int = 50

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        return ArtifactPreview(
            kind="code",
            language="ptx",
            text=_read_text(path, max_chars=max_chars),
        )
