"""Build a pydantic-ai Agent that wraps an `OptimizationSession` + `Harness`.

The pydantic-acp framework consumes a pydantic-ai `Agent`. We bridge the
core's harness contract by wrapping our async harness loop in a
pydantic-ai `FunctionModel`: ACP feeds us a prompt, we drive the harness,
and we return the concatenated output.

Tool dispatch is owned by the harness (which already invokes
`ToolDecl.handler(args)`). The wrapper Agent registers a single
`run_optimizer` tool that lets the ACP client drive a session with explicit
parameters; the canonical session toolset is **not** re-registered on the
wrapper Agent — it lives inside each per-session `Harness.run`.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from compilagent.harness.base import HarnessRunRequest
from compilagent.harness.registry import harness_registry
from compilagent.session.session import OptimizationSession, run_session
from compilagent.settings import CompilagentSettings
from compilagent.storage.trace_store import TraceStore
from compilagent.storage.workspace import OptimizationWorkspace

from .selector import selected_harness


def _harness_extra_from_settings(settings: CompilagentSettings) -> dict[str, Any]:
    out: dict[str, Any] = dict(settings.harness_extra or {})
    if settings.anthropic_api_key is not None:
        out.setdefault(
            "anthropic_api_key", settings.anthropic_api_key.get_secret_value()
        )
    if settings.mistral_api_key is not None:
        out.setdefault(
            "mistral_api_key", settings.mistral_api_key.get_secret_value()
        )
    if settings.openai_api_key is not None:
        out.setdefault(
            "openai_api_key", settings.openai_api_key.get_secret_value()
        )
    return out


def agent_factory_from_session(session_ctx: Any) -> Any:
    """Construct a pydantic-ai Agent for the given ACP session context.

    The Agent exposes a single `run_optimizer` tool — the ACP client passes
    `workload_id` and (optionally) `max_candidates`/`user_prompt`, and the
    factory resolves the harness, builds an `OptimizationSession`, drives
    `run_session`, and returns the resulting leaderboard JSON.
    """

    from pydantic_ai import Agent

    settings = CompilagentSettings.from_env()
    cwd = Path(getattr(session_ctx, "cwd", Path.cwd()) or Path.cwd())
    workspace = OptimizationWorkspace(session_cwd=cwd).ensure()

    agent: Agent[None, str] = Agent(
        model=settings.model_name,
        instructions=(
            "You are an ACP-mounted optimizer. Use `run_optimizer` to drive "
            "an end-to-end optimization session against a registered "
            "workload. Return the resulting leaderboard JSON to the user."
        ),
    )

    @agent.tool_plain
    async def run_optimizer(
        workload_id: str,
        max_candidates: int = 4,
        user_prompt: str = "",
    ) -> str:
        """Run an optimization session for `workload_id` and return the leaderboard."""

        harness_id = selected_harness(session_ctx)
        if harness_id not in harness_registry.ids():
            raise ValueError(
                f"harness `{harness_id}` is not registered; "
                f"known: {harness_registry.ids()}"
            )
        sink = TraceStore(workspace.root).ensure()
        session = OptimizationSession(
            workload_id=workload_id,
            run_id=f"acp-{uuid.uuid4().hex[:10]}",
            workspace=workspace,
            sink=sink,
            max_candidates=max_candidates,
        )
        request = HarnessRunRequest(
            toolset=session.toolset,
            system_instructions=(
                f"Optimize workload `{workload_id}`. Use the canonical "
                f"8-tool toolset; propose multi-intervention candidates."
            ),
            user_prompt=user_prompt
            or (
                "Inspect the workload and search space; propose 3 candidates; "
                "run them; synthesize findings."
            ),
            model_id=settings.model_name,
            reasoning_effort=settings.reasoning_effort,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            max_turns=int(settings.harness_extra.get("max_turns", 24)),
            extra={
                **_harness_extra_from_settings(settings),
                "cwd": str(cwd),
            },
        )
        harness = harness_registry.get(harness_id)
        await run_session(session=session, harness=harness, request=request)
        session.finalize()
        try:
            rows = json.loads(session.compare_runs())
        except Exception:  # noqa: BLE001
            rows = []
        return json.dumps(
            {
                "workload_id": workload_id,
                "harness": harness_id,
                "leaderboard": rows,
            },
            indent=2,
            default=str,
        )

    return agent
