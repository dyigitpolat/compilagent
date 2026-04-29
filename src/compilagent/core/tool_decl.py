"""Canonical agent-tool declaration.

Tools are declared once as `ToolDecl` records. Harness adapters
(`pydantic_ai`, `claude_agent_sdk`, ...) bind these into their native tool
surfaces — pydantic-ai's `@agent.tool_plain` decorators, the Claude Agent
SDK's MCP `@tool` registrations — without diverging on description, schema,
or read-only / destructive flags.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

ToolHandler = Callable[[dict[str, Any]], str]
ReturnsKind = Literal["json", "text"]


@dataclass(frozen=True, slots=True)
class ToolDecl:
    """One agent-facing tool, declared once and bound by harness adapters.

    `args_schema` is a JSON schema for the args object the harness will
    receive from the model and pass through to `handler`. `handler` is
    synchronous; it may raise `ValueError` to signal bad input — adapters
    translate that into the harness's native retry/error response.
    """

    name: str
    description: str
    args_schema: dict[str, Any]
    handler: ToolHandler
    read_only: bool
    returns_kind: ReturnsKind = "json"
    metadata: dict[str, Any] = field(default_factory=dict)
