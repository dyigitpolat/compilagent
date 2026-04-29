"""ACP server shim — mounts an `OptimizationSession` over pydantic-acp.

This integration does not register a backend or harness. It plugs into ACP
clients (Zed, Claude Code, …) and lets them drive a per-session optimizer
through whatever harness the user selects from the registry.
"""

from __future__ import annotations

from .factory import agent_factory_from_session
from .selector import HARNESS_CONFIG_ID, AcpHarnessSelector, selected_harness
from .server import build_config, run_acp_server

__all__ = [
    "AcpHarnessSelector",
    "HARNESS_CONFIG_ID",
    "agent_factory_from_session",
    "build_config",
    "run_acp_server",
    "selected_harness",
]
