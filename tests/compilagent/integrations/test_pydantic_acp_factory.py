"""Tests for the pydantic-acp shim (selector + factory)."""

from __future__ import annotations

from dataclasses import dataclass, field

from compilagent.harness.registry import harness_registry


@dataclass
class _FakeAcpSession:
    cwd: str = "/tmp"
    config_values: dict = field(default_factory=dict)


@dataclass
class _StubHarness:
    id: str = "stub"
    supported_providers: tuple[str, ...] = ("test",)

    async def run(self, request):
        from compilagent.harness.base import StreamEvent, StreamEventKind

        yield StreamEvent(kind=StreamEventKind.RUN_FINISHED, text="ok")


def test_selector_pulls_options_from_registry():
    harness_registry.register("stub", _StubHarness)

    from compilagent.integrations.pydantic_acp.selector import (
        AcpHarnessSelector,
        selected_harness,
    )

    selector = AcpHarnessSelector()
    options = selector.get_config_options(_FakeAcpSession(), agent=None)
    assert options  # non-empty
    if options:
        ids = [o.value for o in options[0].options]
        assert "stub" in ids

    session = _FakeAcpSession(config_values={"harness": "stub"})
    assert selected_harness(session) == "stub"


def test_selector_set_config_option_validates_against_registry():
    harness_registry.register("stub", _StubHarness)

    from compilagent.integrations.pydantic_acp.selector import (
        HARNESS_CONFIG_ID,
        AcpHarnessSelector,
    )

    selector = AcpHarnessSelector()
    session = _FakeAcpSession(config_values={})
    # unknown id rejected
    assert selector.set_config_option(session, None, HARNESS_CONFIG_ID, "missing") is None
    # known id accepted
    result = selector.set_config_option(session, None, HARNESS_CONFIG_ID, "stub")
    assert result is not None
    assert session.config_values[HARNESS_CONFIG_ID] == "stub"
