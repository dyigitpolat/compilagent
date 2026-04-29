"""Abstract harness contract — one shape both pydantic-ai and the Claude
Agent SDK implement.

The session drives setup (baseline compile, search-space derivation, tool
surface) and then hands control to the harness via `harness.run(request)`.
The harness drives the inner agent loop: it receives the toolset, exposes
each tool through its native surface (decorators / MCP), streams events
back, and dispatches tool calls by invoking `ToolDecl.handler(args)`. The
session reacts to `StreamEvent`s by emitting through its `ObservationSink`.

Vendor types (`pydantic_ai.messages.*`, `claude_agent_sdk.*`) appear only in
the integration adapters — never in this module or the session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from compilagent.toolset import Toolset


class StreamEventKind(StrEnum):
    THINKING_STARTED = "thinking.started"
    THINKING_DELTA = "thinking.delta"
    TEXT_STARTED = "text.started"
    TEXT_DELTA = "text.delta"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Generic event yielded by `Harness.run`.

    Adapters translate vendor message types into this shape so the session
    sees a single event vocabulary regardless of harness identity.
    """

    kind: StreamEventKind
    part_index: int | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    text: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HarnessRunRequest:
    """Everything the harness needs to drive one agent run."""

    toolset: Toolset
    system_instructions: str
    user_prompt: str
    model_id: str
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    max_turns: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """Summary returned by `run_harness` after the stream terminates."""

    final_text: str | None
    elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Harness(Protocol):
    """Abstract agent harness.

    Out-of-tree integrations: implement structurally (no inheritance
    required). Translate vendor message types (pydantic-ai parts, Claude
    Agent SDK messages, etc.) into `StreamEvent`s — that translation is the
    only place vendor types should appear.
    """

    id: str
    """Stable string id this harness registers under (e.g. `"pydantic_ai"`)."""

    supported_providers: tuple[str, ...]
    """Provider prefixes this harness can route, e.g. `("anthropic","mistral")`.

    The session validates that `model_id`'s provider prefix is in this tuple
    before dispatching. Empty tuple means the harness accepts any model id
    (e.g. when it routes through a single fixed provider).
    """

    def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        """Drive one agent run, yielding `StreamEvent`s until the run ends.

        Contract:

          - The final yielded event MUST be `RUN_FINISHED` or `RUN_FAILED`.
          - Tool-call dispatch is the harness's responsibility — it invokes
            `request.toolset.by_name(name).handler(args)` for each tool call
            and yields a `TOOL_RESULT` event carrying the handler's string
            return value.
          - When `handler` raises `ValueError`, the harness MUST yield a
            `TOOL_ERROR` event (not propagate the exception); the agent will
            then see it as a retryable error.
          - Read-only / destructive gating: respect `ToolDecl.read_only`
            when the harness has a notion of permission modes (e.g. ACP's
            prepare-tool, Claude SDK's `allowed_tools`).
          - Configuration: `request.extra` carries harness-specific knobs
            (max_budget_usd, permission_mode, retries, …); the harness
            consumes only what it recognises.
        """
        ...
