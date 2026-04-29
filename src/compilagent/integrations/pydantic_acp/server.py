"""ACP server entry point.

`run_acp_server()` mounts the per-session agent factory built in
`factory.py` onto pydantic-acp's `run_acp(...)`. The harness the client
picks is read from `selector.AcpHarnessSelector`, which reads
`harness_registry.ids()` at runtime — adding a harness requires zero code
changes here.
"""

from __future__ import annotations

from compilagent.bootstrap import load_entry_point_integrations

from .factory import agent_factory_from_session
from .selector import AcpHarnessSelector


def build_config() -> object:
    """Build the pydantic-acp `AdapterConfig`.

    The config exposes:
      - `MemorySessionStore` for per-session state (in-memory; Phase-3 may
        swap for `FileSessionStore`).
      - `AcpHarnessSelector` so the client can switch harness mid-session.
      - `ThinkingBridge` so reasoning deltas reach the ACP client.
      - `PrepareToolsBridge` is intentionally left off this minimal scaffold;
        the canonical session toolset's `read_only` flags can drive a future
        gating bridge.
    """

    from pydantic_acp import AdapterConfig, MemorySessionStore, ThinkingBridge

    return AdapterConfig(
        session_store=MemorySessionStore(),
        config_options_provider=AcpHarnessSelector(),
        capability_bridges=[ThinkingBridge()],
    )


def run_acp_server() -> None:
    """Start the ACP server.

    Loads any out-of-tree integrations advertised via entry points before
    mounting the agent factory, so harnesses installed via `pip install`
    appear in the harness selector immediately.
    """

    from pydantic_acp import run_acp

    load_entry_point_integrations()
    run_acp(agent_factory=agent_factory_from_session, config=build_config())
