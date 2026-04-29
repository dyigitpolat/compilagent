"""ACP `SessionConfigOptionsProvider` over the core's `harness_registry`.

Adding a new harness — first-party or out-of-tree — requires zero changes
here. We read `harness_registry.ids()` at runtime and emit a `select`
config option whose values match those ids.
"""

from __future__ import annotations

from typing import Any

from compilagent.harness.registry import harness_registry
from compilagent.settings import CompilagentSettings

HARNESS_CONFIG_ID = "harness"


def _normalize(value: str) -> str:
    return value.strip().replace("-", "_")


def _default_harness(session: Any) -> str:
    settings = CompilagentSettings.from_env()
    configured = (getattr(session, "config_values", {}) or {}).get(
        HARNESS_CONFIG_ID, settings.harness
    )
    return _normalize(str(configured))


class AcpHarnessSelector:
    """Surfaces the harness registry as an ACP session-config select option."""

    def get_config_options(self, session: Any, agent: Any) -> list:
        del agent
        try:
            from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption
        except Exception:  # noqa: BLE001
            return []

        ids = harness_registry.ids()
        return [
            SessionConfigOptionSelect(
                id=HARNESS_CONFIG_ID,
                name="Harness",
                category="agent",
                description=(
                    "Select the optimizer agentic harness. Values come from "
                    "`compilagent.harness_registry`; install or import any "
                    "harness integration to make it appear here."
                ),
                type="select",
                current_value=_default_harness(session),
                options=[
                    SessionConfigSelectOption(value=h, name=h) for h in ids
                ],
            )
        ]

    def set_config_option(
        self,
        session: Any,
        agent: Any,
        config_id: str,
        value: str | bool,
    ) -> list | None:
        if config_id != HARNESS_CONFIG_ID or not isinstance(value, str):
            return None
        normalized = _normalize(value)
        if normalized not in harness_registry.ids():
            return None
        session.config_values[HARNESS_CONFIG_ID] = normalized
        return self.get_config_options(session, agent)


def selected_harness(session: Any) -> str:
    return _default_harness(session)
