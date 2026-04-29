"""Build the in-process MCP server from a `Toolset`.

The legacy version reflected over `OptimizerRuntime` and hard-coded a list
of 20+ tools by name. The new version iterates `request.toolset.tools` and
binds each `ToolDecl` directly: backend introspection tools and any future
canonical tools auto-appear without touching this file.
"""

from __future__ import annotations

from typing import Any

from compilagent.toolset import Toolset

SDK_TOOL_PREFIX = "mcp__compilagent__"


def _response(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error_response(exc: Exception) -> dict[str, Any]:
    return {
        "isError": True,
        "content": [
            {"type": "text", "text": f"{type(exc).__name__}: {exc}"}
        ],
    }


def create_optimizer_mcp_server(toolset: Toolset, *, server_name: str = "compilagent") -> Any:
    """Construct an in-process MCP server exposing every `ToolDecl` in `toolset`."""

    import claude_agent_sdk as sdk  # lazy

    tool_decorator = sdk.tool
    create_sdk_mcp_server = sdk.create_sdk_mcp_server
    annotations_cls = sdk.ToolAnnotations

    tools: list = []
    for decl in toolset.tools:
        ann = annotations_cls(
            readOnlyHint=decl.read_only,
            destructiveHint=not decl.read_only,
            openWorldHint=False,
        )

        @tool_decorator(decl.name, decl.description, decl.args_schema, annotations=ann)
        async def _handler(args: dict[str, Any], *, _decl=decl) -> dict[str, Any]:
            try:
                result = _decl.handler(args)
            except ValueError as exc:
                return _error_response(exc)
            except Exception as exc:  # noqa: BLE001
                return _error_response(exc)
            return _response(result)

        tools.append(_handler)

    return create_sdk_mcp_server(name=server_name, tools=tools)


def allowed_tools_for(toolset: Toolset, *, include_built_ins: bool) -> list[str]:
    allowed = [f"{SDK_TOOL_PREFIX}{decl.name}" for decl in toolset.tools]
    if include_built_ins:
        allowed.extend(["Read", "Glob", "Grep", "WebFetch", "WebSearch", "AskUserQuestion"])
    return allowed
