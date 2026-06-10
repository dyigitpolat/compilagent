"""The process-global request throttle in archetype_harnesses._llm.

P0 probe finding #7: Mistral 429s above 2 concurrent requests; stable at a
1.2 s global min-interval between request starts. `reserve_request_start`
implements the slot-reservation half of that contract.
"""

from __future__ import annotations

import pytest

from compilagent.integrations.archetype_harnesses import _llm


@pytest.fixture(autouse=True)
def _fresh_throttle(monkeypatch):
    monkeypatch.setattr(_llm, "_next_start", [0.0])


def test_slots_are_spaced_by_the_min_interval():
    waits = [_llm.reserve_request_start(0.5) for _ in range(3)]
    assert waits[0] == 0.0
    assert waits[1] == pytest.approx(0.5, abs=0.05)
    assert waits[2] == pytest.approx(1.0, abs=0.05)


def test_zero_interval_never_waits():
    assert all(_llm.reserve_request_start(0.0) == 0.0 for _ in range(5))


def test_defaults_match_the_probe_protocol():
    assert _llm.DEFAULT_MIN_INTERVAL_S == 1.2
    assert _llm.DEFAULT_MAX_CONCURRENT == 2
