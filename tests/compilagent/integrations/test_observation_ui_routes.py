"""Tests for the observation UI's HTTP routes.

Uses FastAPI's `TestClient` against an in-process app rooted at a tmp dir.
Verifies the registry-backed endpoints + the suffix-table-free preview path.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from compilagent.core.backend import backend_registry
from compilagent.harness.registry import harness_registry
from compilagent.observation.artifacts import (
    ArtifactPreview,
    artifact_renderer_registry,
)
from compilagent.settings import CompilagentSettings


def _make_app(tmp_path: Path):
    from compilagent.integrations.observation_ui import create_app

    settings = CompilagentSettings(model_name="test", harness="nop")
    return create_app(workspace_root=tmp_path, settings=settings)


def test_index_serves(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200


def test_backends_endpoint_returns_registry(tmp_path):
    class _B:
        id = "fake_b"
        artifact_stages: tuple[str, ...] = ("ir",)

    backend_registry.register("fake_b", _B)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/backends")
        assert r.status_code == 200
        body = r.json()
        ids = [b["id"] for b in body["backends"]]
        assert "fake_b" in ids


def test_harnesses_endpoint_returns_registry(tmp_path):
    class _H:
        id = "fake_h"
        supported_providers: tuple[str, ...] = ("provider_a",)

    harness_registry.register("fake_h", _H)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/harnesses")
        assert r.status_code == 200
        body = r.json()
        ids = [h["id"] for h in body["harnesses"]]
        assert "fake_h" in ids


def test_artifact_preview_uses_registered_renderer(tmp_path):
    """A backend-shaped suffix is rendered by whatever the registry holds —
    no suffix table in the UI code itself."""

    target = tmp_path / ".compilagent" / "demo.custom_ir"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("HELLO custom_ir BODY", encoding="utf-8")

    from dataclasses import dataclass

    @dataclass(frozen=True, slots=True)
    class CustomRenderer:
        suffixes: tuple[str, ...] = (".custom_ir",)
        priority: int = 99

        def render(self, path, *, max_chars: int = 40_000):
            return ArtifactPreview(
                kind="code",
                language="custom",
                text=path.read_text(encoding="utf-8"),
            )

    artifact_renderer_registry.register(CustomRenderer())
    try:
        app = _make_app(tmp_path)
        with TestClient(app) as client:
            r = client.get("/api/artifacts/preview", params={"path": "demo.custom_ir"})
            assert r.status_code == 200
            body = r.json()
            assert body["language"] == "custom"
            assert "HELLO custom_ir BODY" in body["text"]
    finally:
        # Best-effort cleanup so the registry singleton doesn't leak this
        # renderer into other tests' assertions.
        artifact_renderer_registry.clear()
        from compilagent.observation.artifacts import build_default_registry

        for r in build_default_registry()._renderers:  # type: ignore[attr-defined]
            artifact_renderer_registry.register(r)


def test_artifact_preview_path_traversal_blocked(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/artifacts/preview", params={"path": "../../../etc/passwd"})
        body = r.json()
        # Either error or empty preview; must not return host file contents.
        assert body.get("error") == "path_traversal" or not body.get("exists")


def test_runtime_config_returns_settings_metadata(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/runtime/config")
        assert r.status_code == 200
        body = r.json()
        assert body["harness"] == "nop"
        assert body["model_name"] == "test"
