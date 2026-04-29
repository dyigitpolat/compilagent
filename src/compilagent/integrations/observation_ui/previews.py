"""Artifact previews for the observation UI.

Routes through `compilagent.observation.artifact_renderer_registry` —
backend-specific suffix handling lives in each backend's
`list_artifact_renderers()`, never in this module. The SPA never branches
on backend identity to decide how to render a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compilagent.observation.artifacts import (
    ArtifactRendererRegistry,
    artifact_renderer_registry,
)
from compilagent.storage.workspace import OptimizationWorkspace


def safe_resolve(workspace: OptimizationWorkspace, relative: str) -> Path:
    """Resolve `relative` inside `workspace.root`; raises on path-traversal."""

    return workspace.resolve(relative)


def render_preview(
    workspace: OptimizationWorkspace,
    relative: str,
    *,
    max_chars: int = 40_000,
    registry: ArtifactRendererRegistry | None = None,
) -> dict[str, Any]:
    """Read an artifact and produce the JSON preview the SPA renders.

    Returns `{kind, language, text, path, exists, error?}`.
    """

    reg = registry or artifact_renderer_registry
    try:
        path = safe_resolve(workspace, relative)
    except ValueError as exc:
        return {
            "kind": "text",
            "language": "text",
            "text": f"<bad path: {exc}>",
            "path": relative,
            "exists": False,
            "error": "path_traversal",
        }
    if not path.exists():
        return {
            "kind": "text",
            "language": "text",
            "text": f"<not found: {relative}>",
            "path": str(path),
            "exists": False,
        }
    preview = reg.render(path, max_chars=max_chars)
    return {
        "kind": preview.kind,
        "language": preview.language,
        "text": preview.text,
        "path": str(path),
        "exists": True,
    }
