"""`ClaudeAgentSdkHarness` — implements the core `Harness` protocol on the
Claude Agent SDK.

The harness:
  1. Builds an in-process MCP server from `request.toolset` (no reflection).
  2. Configures the SDK with that MCP server and the canonical-tool allowlist.
  3. Drives `client.query(prompt)` and translates each SDK message into a
     core `StreamEvent`.
  4. Closes the SDK client and emits a final `RUN_FINISHED` (or `RUN_FAILED`)
     carrying cost / turn-count metadata from the SDK's `ResultMessage`.

Configuration knobs read from `request.extra`:
  - `max_turns` (int)
  - `max_budget_usd` (float | None)
  - `permission_mode` (str, default "dontAsk")
  - `cwd` (str | Path | None) — defaults to current working dir
  - `read_only` (bool) — if True, only read-only tools advertised
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from compilagent.harness.base import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
)
from compilagent.toolset import Toolset

from ._mcp import allowed_tools_for, create_optimizer_mcp_server
from ._translate import translate_sdk_message

_MCP_SERVER_NAME = "compilagent"


class ClaudeAgentSdkHarness:
    """Harness adapter that drives the Claude Agent SDK."""

    id: str = "claude_agent_sdk"
    supported_providers: tuple[str, ...] = ("anthropic",)

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        try:
            import claude_agent_sdk as sdk
        except ImportError as exc:  # pragma: no cover
            yield StreamEvent(
                kind=StreamEventKind.RUN_FAILED,
                error_type="ImportError",
                error_message=f"claude-agent-sdk is not installed: {exc!r}",
            )
            return

        toolset: Toolset = request.toolset
        if request.extra.get("read_only"):
            toolset = toolset.read_only_subset()

        options = self._build_options(sdk, request, toolset)
        client = sdk.ClaudeSDKClient(options=options)

        cost_usd: float | None = None
        turn_count: int | None = None
        session_id: str | None = None
        final_result: str | None = None

        try:
            await client.connect()
            await client.query(request.user_prompt)
            async for message in client.receive_response():
                # Capture the terminal ResultMessage's metadata for the final event.
                if isinstance(message, sdk.ResultMessage):
                    cost_usd = getattr(message, "total_cost_usd", None)
                    turn_count = getattr(message, "num_turns", None)
                    session_id = getattr(message, "session_id", None)
                    result = getattr(message, "result", None)
                    if isinstance(result, str):
                        final_result = result
                    continue
                for ev in translate_sdk_message(message):
                    yield ev
        except Exception as exc:  # noqa: BLE001
            yield StreamEvent(
                kind=StreamEventKind.RUN_FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return
        finally:
            with contextlib.suppress(Exception):
                await client.disconnect()

        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text=final_result,
            extra={
                "cost_usd": cost_usd,
                "turn_count": turn_count,
                "session_id": session_id,
            },
        )

    def _build_options(
        self,
        sdk: Any,
        request: HarnessRunRequest,
        toolset: Toolset,
    ) -> Any:
        cwd = request.extra.get("cwd")
        max_turns = request.max_turns or int(request.extra.get("max_turns", 24))
        max_budget_usd = request.extra.get("max_budget_usd")
        permission_mode = str(request.extra.get("permission_mode", "dontAsk"))
        read_only = bool(request.extra.get("read_only"))

        # Strip any `anthropic:` prefix; the SDK takes the bare model id.
        model_id = (request.model_id or "").removeprefix("anthropic:") or None

        kwargs: dict[str, Any] = {
            "tools": ["Read", "Glob", "Grep", "WebFetch", "WebSearch", "AskUserQuestion"],
            "allowed_tools": allowed_tools_for(toolset, include_built_ins=not read_only),
            "system_prompt": {
                "type": "preset",
                "preset": "claude_code",
                "append": request.system_instructions,
            },
            "mcp_servers": {
                _MCP_SERVER_NAME: create_optimizer_mcp_server(
                    toolset, server_name=_MCP_SERVER_NAME
                )
            },
            "permission_mode": permission_mode,
            "max_turns": max_turns,
            "setting_sources": ["project"],
        }
        if cwd is not None:
            kwargs["cwd"] = cwd
        if model_id:
            kwargs["model"] = model_id
        if max_budget_usd is not None:
            kwargs["max_budget_usd"] = float(max_budget_usd)
        # Accept either `effort` or no effort knob; the SDK picks defaults.
        effort = request.reasoning_effort
        if effort:
            kwargs["effort"] = effort
        return sdk.ClaudeAgentOptions(**kwargs)
