"""Translate Claude Agent SDK messages into core `StreamEvent`s."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from compilagent.harness.base import StreamEvent, StreamEventKind


def translate_sdk_message(message: Any) -> Iterator[StreamEvent]:
    """Yield zero or more `StreamEvent`s for one SDK message."""

    import claude_agent_sdk as sdk  # lazy

    AssistantMessage = sdk.AssistantMessage
    ResultMessage = sdk.ResultMessage
    TextBlock = sdk.TextBlock
    ThinkingBlock = sdk.ThinkingBlock
    ToolUseBlock = sdk.ToolUseBlock
    ToolResultBlock = sdk.ToolResultBlock

    if isinstance(message, AssistantMessage):
        for block in getattr(message, "content", ()) or ():
            if isinstance(block, ThinkingBlock):
                yield StreamEvent(
                    kind=StreamEventKind.THINKING_DELTA,
                    text=str(getattr(block, "thinking", "") or ""),
                )
            elif isinstance(block, TextBlock):
                yield StreamEvent(
                    kind=StreamEventKind.TEXT_DELTA,
                    text=str(getattr(block, "text", "") or ""),
                )
            elif isinstance(block, ToolUseBlock):
                args = getattr(block, "input", None)
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_CALL,
                    tool_name=getattr(block, "name", None),
                    tool_call_id=getattr(block, "id", None),
                    tool_args=args if isinstance(args, dict) else None,
                )
        return

    if isinstance(message, getattr(sdk, "UserMessage", ())):
        # User messages from the SDK carry tool results back to us.
        for block in getattr(message, "content", ()) or ():
            if isinstance(block, ToolResultBlock):
                content = getattr(block, "content", None)
                tool_call_id = getattr(block, "tool_use_id", None)
                if isinstance(content, list):
                    text_parts = [
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict)
                    ]
                    text = "".join(text_parts)
                else:
                    text = str(content) if content is not None else ""
                is_error = bool(getattr(block, "is_error", False))
                if is_error:
                    yield StreamEvent(
                        kind=StreamEventKind.TOOL_ERROR,
                        tool_call_id=tool_call_id,
                        error_type="ToolError",
                        error_message=text,
                    )
                else:
                    yield StreamEvent(
                        kind=StreamEventKind.TOOL_RESULT,
                        tool_call_id=tool_call_id,
                        tool_result=text,
                    )
        return

    if isinstance(message, ResultMessage):
        # Terminal SDK message; the harness inspects this to build the final
        # `RUN_FINISHED` event with cost/turn metadata.
        return
