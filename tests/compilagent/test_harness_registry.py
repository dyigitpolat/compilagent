from __future__ import annotations

import pytest

from compilagent.harness.registry import HarnessRegistry


class _FakeHarness:
    id = "fake"
    supported_providers: tuple[str, ...] = ("anthropic",)


def test_register_and_get_round_trip():
    registry = HarnessRegistry()
    registry.register("fake", _FakeHarness)
    assert registry.ids() == ["fake"]
    harness = registry.get("fake")
    assert harness.id == "fake"


def test_register_duplicate_raises():
    registry = HarnessRegistry()
    registry.register("fake", _FakeHarness)
    with pytest.raises(ValueError):
        registry.register("fake", _FakeHarness)


def test_get_unknown_raises():
    registry = HarnessRegistry()
    with pytest.raises(KeyError):
        registry.get("missing")
