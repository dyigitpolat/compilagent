"""Artifact previews — the seam that prevents the observation UI from
hard-coding backend-specific suffixes.

A backend declares one or more `ArtifactRenderer`s in its
`list_artifact_renderers()` method. The session's bootstrap registers them
into the `ArtifactRendererRegistry`. The UI calls `registry.for_suffix(s)`
and renders whatever `ArtifactPreview` it gets back, regardless of which
backend produced the file. The core ships fallback renderers for the
generic suffixes (`.json`, `.md`, `.txt`, `.log`, `.py`); MLIR / PTX / FX
renderers are owned by the integration packages.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

PreviewKind = Literal["json", "markdown", "code", "text", "binary"]


@dataclass(frozen=True, slots=True)
class ArtifactPreview:
    kind: PreviewKind
    language: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ArtifactRenderer(Protocol):
    """Renders one artifact file into an `ArtifactPreview` for the UI.

    Backends declare renderers in `Backend.list_artifact_renderers()`;
    out-of-tree code may also register directly via
    `artifact_renderer_registry.register(my_renderer)`.
    """

    suffixes: tuple[str, ...]
    """File suffixes this renderer handles, lowercase, including the dot
    (`(".ttgir", ".ttir")`). The registry compares case-insensitively."""

    priority: int
    """Tie-breaker when multiple renderers handle the same suffix; higher
    wins. Built-in fallbacks register at `priority=10`; integration-owned
    renderers should pick a higher value (e.g. 50) to override."""

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        """Read `path` and produce a preview no longer than `max_chars`.

        Implementations should:
          - Use `errors="replace"` decoding so partial bytes don't kill the
            UI.
          - Truncate gracefully and indicate truncation in the returned text.
          - Never raise on a missing or unreadable file — return a preview
            with `kind="text"` and a `<unreadable: ...>` body instead.
        """
        ...


def _read_text_safely(path: Path, *, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unreadable: {exc!r}>"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... <truncated {len(text) - max_chars} chars>"
    return text


@dataclass(frozen=True, slots=True)
class JsonRenderer:
    suffixes: tuple[str, ...] = (".json",)
    priority: int = 10

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        text = _read_text_safely(path, max_chars=max_chars)
        try:
            parsed = json.loads(text)
            pretty = json.dumps(parsed, indent=2, default=str)
            if len(pretty) > max_chars:
                pretty = pretty[:max_chars] + f"\n\n... <truncated {len(pretty) - max_chars} chars>"
            return ArtifactPreview(kind="json", language="json", text=pretty)
        except (json.JSONDecodeError, ValueError):
            return ArtifactPreview(kind="text", language="text", text=text)


@dataclass(frozen=True, slots=True)
class MarkdownRenderer:
    suffixes: tuple[str, ...] = (".md", ".markdown")
    priority: int = 10

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        return ArtifactPreview(
            kind="markdown",
            language="markdown",
            text=_read_text_safely(path, max_chars=max_chars),
        )


@dataclass(frozen=True, slots=True)
class PythonRenderer:
    suffixes: tuple[str, ...] = (".py",)
    priority: int = 10

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        return ArtifactPreview(
            kind="code",
            language="python",
            text=_read_text_safely(path, max_chars=max_chars),
        )


@dataclass(frozen=True, slots=True)
class TextRenderer:
    suffixes: tuple[str, ...] = (".txt", ".log")
    priority: int = 10

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        return ArtifactPreview(
            kind="text",
            language="text",
            text=_read_text_safely(path, max_chars=max_chars),
        )


class ArtifactRendererRegistry:
    """Suffix → renderer lookup with explicit priority tie-breaking."""

    def __init__(self) -> None:
        self._renderers: list[ArtifactRenderer] = []

    def register(self, renderer: ArtifactRenderer) -> None:
        if not isinstance(renderer, ArtifactRenderer):
            raise TypeError(
                f"`{type(renderer).__name__}` does not implement ArtifactRenderer"
            )
        self._renderers.append(renderer)

    def register_many(self, renderers: Sequence[ArtifactRenderer]) -> None:
        for r in renderers:
            self.register(r)

    def for_suffix(self, suffix: str) -> ArtifactRenderer | None:
        suffix = suffix.lower()
        matching = [r for r in self._renderers if suffix in r.suffixes]
        if not matching:
            return None
        return max(matching, key=lambda r: r.priority)

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        renderer = self.for_suffix(path.suffix)
        if renderer is not None:
            return renderer.render(path, max_chars=max_chars)
        return ArtifactPreview(
            kind="binary",
            language="",
            text=f"<no renderer registered for `{path.suffix or '<no suffix>'}`>",
            metadata={"path": str(path)},
        )

    def clear(self) -> None:
        self._renderers.clear()


def build_default_registry() -> ArtifactRendererRegistry:
    """Registry pre-populated with the generic fallback renderers."""

    registry = ArtifactRendererRegistry()
    registry.register_many(
        (JsonRenderer(), MarkdownRenderer(), PythonRenderer(), TextRenderer())
    )
    return registry


artifact_renderer_registry = build_default_registry()
