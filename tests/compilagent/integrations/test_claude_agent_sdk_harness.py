"""Tests for the Claude Agent SDK harness — without making real SDK calls."""

from __future__ import annotations

from compilagent.core.tool_decl import ToolDecl
from compilagent.harness.registry import harness_registry
from compilagent.toolset import Toolset


def test_self_registration():
    import compilagent.integrations.claude_agent_sdk  # noqa

    assert "claude_agent_sdk" in harness_registry.ids()
    h = harness_registry.get("claude_agent_sdk")
    assert h.id == "claude_agent_sdk"
    assert h.supported_providers == ("anthropic",)


def test_mcp_server_built_from_toolset_not_reflection():
    """Verify the new design: the MCP server iterates Toolset.tools rather
    than introspecting an OptimizerRuntime."""

    from compilagent.integrations.claude_agent_sdk._mcp import (
        SDK_TOOL_PREFIX,
        allowed_tools_for,
        create_optimizer_mcp_server,
    )

    decl_a = ToolDecl(
        name="alpha",
        description="A",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _: "ok",
        read_only=True,
    )
    decl_b = ToolDecl(
        name="beta",
        description="B",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _: "ok",
        read_only=False,
    )
    toolset = Toolset(tools=(decl_a, decl_b))

    # Allowed-tools list reflects the toolset, not a hard-coded list.
    allowed = allowed_tools_for(toolset, include_built_ins=False)
    assert allowed == [f"{SDK_TOOL_PREFIX}alpha", f"{SDK_TOOL_PREFIX}beta"]

    # Constructing the server doesn't crash; it returns whatever the SDK's
    # `create_sdk_mcp_server` returns. (We don't call into it; just make
    # sure the construction succeeds with our adapter.)
    server = create_optimizer_mcp_server(toolset)
    assert server is not None


def test_allowed_tools_includes_builtins_when_requested():
    from compilagent.integrations.claude_agent_sdk._mcp import allowed_tools_for

    toolset = Toolset(
        tools=(
            ToolDecl(
                name="alpha",
                description="A",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                handler=lambda _: "",
                read_only=True,
            ),
        )
    )
    full = allowed_tools_for(toolset, include_built_ins=True)
    assert "Read" in full
    assert "WebFetch" in full
