from __future__ import annotations

import pytest

from compilagent.core.backend import BackendRegistry


class _FakeBackend:
    id = "fake"
    artifact_stages: tuple[str, ...] = ()


def test_register_and_get_round_trip():
    registry = BackendRegistry()
    registry.register("fake", _FakeBackend)
    assert registry.ids() == ["fake"]
    backend = registry.get("fake")
    assert backend.id == "fake"


def test_register_duplicate_raises():
    registry = BackendRegistry()
    registry.register("fake", _FakeBackend)
    with pytest.raises(ValueError):
        registry.register("fake", _FakeBackend)


def test_get_unknown_raises():
    registry = BackendRegistry()
    with pytest.raises(KeyError):
        registry.get("missing")


def test_clear_removes_all():
    registry = BackendRegistry()
    registry.register("fake", _FakeBackend)
    registry.clear()
    assert registry.ids() == []
