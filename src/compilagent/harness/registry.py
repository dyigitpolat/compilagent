"""Process-wide harness registry.

Each harness integration self-registers at import time, e.g.

    from compilagent.harness import harness_registry
    from .harness import PydanticAIHarness
    harness_registry.register("pydantic_ai", PydanticAIHarness)

Settings carry the harness id as a plain string; the registry resolves it.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Harness


class HarnessRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Harness]] = {}

    def register(self, harness_id: str, factory: Callable[[], Harness]) -> None:
        if harness_id in self._factories:
            raise ValueError(f"Harness `{harness_id}` is already registered.")
        self._factories[harness_id] = factory

    def get(self, harness_id: str) -> Harness:
        if harness_id not in self._factories:
            known = sorted(self._factories.keys())
            raise KeyError(
                f"Unknown harness `{harness_id}`. Registered: {known or '(none)'}."
            )
        return self._factories[harness_id]()

    def ids(self) -> list[str]:
        return sorted(self._factories.keys())

    def clear(self) -> None:
        self._factories.clear()


harness_registry = HarnessRegistry()
