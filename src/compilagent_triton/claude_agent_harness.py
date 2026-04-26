from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests

from .optimizer_runtime import OptimizerRuntime, optimizer_instructions
from .settings import CompilagentSettings
from .workspace import OptimizationWorkspace

SDK_TOOL_SERVER_NAME = "triton_optimizer"
SDK_TOOL_PREFIX = f"mcp__{SDK_TOOL_SERVER_NAME}__"

READ_ONLY_TOOL_NAMES = (
    "describe_optimizer_surface",
    "list_benchmarks",
    "compile_baseline",
    "inspect_ir",
    "summarize_decisions",
    "inspect_optimization_toolset",
    "propose_candidates",
    "validate_candidate",
    "compare_runs",
    "compare_benchmarks",
)
ALL_TOOL_NAMES = (
    "describe_optimizer_surface",
    "list_benchmarks",
    "register_kernel",
    "start_episode",
    "compile_baseline",
    "inspect_ir",
    "summarize_decisions",
    "record_hypothesis",
    "record_reasoning_summary",
    "inspect_optimization_toolset",
    "propose_candidates",
    "propose_candidate_from_toolset",
    "validate_candidate",
    "run_candidate",
    "run_baseline_benchmark",
    "run_candidate_benchmark",
    "compare_runs",
    "compare_benchmarks",
    "accept_or_reject_candidate",
    "write_report",
)


class ClaudeAgentSdkHarness:
    """Run prompts through Claude Agent SDK while exposing local optimizer tools."""

    def __init__(
        self,
        *,
        runtime: OptimizerRuntime,
        settings: CompilagentSettings,
        mode: str,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.settings = settings
        self.mode = mode
        self._client_factory = client_factory
        self._client: Any | None = None

    async def run(self, prompt: str) -> str:
        self.runtime.emit(
            "agent.prompt_received",
            payload={"harness": "claude_agent_sdk", "mode": self.mode},
        )
        client = await self._ensure_client()
        await client.query(prompt)

        final_result: str | None = None
        assistant_chunks: list[str] = []
        async for message in client.receive_response():
            if hasattr(message, "result"):
                result = getattr(message, "result", None)
                if isinstance(result, str):
                    final_result = result
                self.runtime.emit(
                    "agent.sdk_result",
                    payload={
                        "subtype": getattr(message, "subtype", None),
                        "session_id": getattr(message, "session_id", None),
                        "total_cost_usd": getattr(message, "total_cost_usd", None),
                    },
                )
            else:
                assistant_chunks.extend(_message_text_chunks(message))

        if final_result is not None:
            return final_result
        return "\n".join(chunk for chunk in assistant_chunks if chunk).strip()

    async def close(self) -> None:
        if self._client is not None and hasattr(self._client, "disconnect"):
            await self._client.disconnect()
        self._client = None

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        sdk = _import_sdk()
        client_factory = self._client_factory or sdk.ClaudeSDKClient
        self._client = client_factory(options=self._build_options(sdk))
        if hasattr(self._client, "connect"):
            await self._client.connect()
        return self._client

    def _build_options(self, sdk: Any) -> Any:
        return sdk.ClaudeAgentOptions(
            tools=["Read", "Glob", "Grep", "WebFetch", "WebSearch", "AskUserQuestion"],
            allowed_tools=self._allowed_tools(),
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": optimizer_instructions(
                    self.settings,
                    self.runtime.workspace,
                    harness_label="Claude Agent SDK",
                ),
            },
            mcp_servers={SDK_TOOL_SERVER_NAME: create_optimizer_mcp_server(self.runtime)},
            permission_mode=self._permission_mode(),
            cwd=self.runtime.workspace.session_cwd,
            model=self.settings.claude_sdk_model_name(),
            max_turns=self.settings.claude_sdk_max_turns,
            max_budget_usd=self.settings.claude_sdk_max_budget_usd,
            effort=self.settings.claude_sdk_effort_value(),
            setting_sources=["project"],
        )

    def _allowed_tools(self) -> list[str]:
        custom_tool_names = ALL_TOOL_NAMES if self.mode == "optimize" else READ_ONLY_TOOL_NAMES
        allowed = [f"{SDK_TOOL_PREFIX}{name}" for name in custom_tool_names]
        if self.mode == "optimize":
            allowed.extend(["Read", "Glob", "Grep", "WebFetch", "WebSearch", "AskUserQuestion"])
        return allowed

    def _permission_mode(self) -> str:
        if self.mode == "optimize":
            return self.settings.claude_sdk_permission_mode
        return "dontAsk"


def build_claude_sdk_proxy_agent(
    *,
    settings: CompilagentSettings,
    workspace: OptimizationWorkspace,
    mode: str,
    client_factory: Callable[..., Any] | None = None,
) -> Agent[None, str | DeferredToolRequests]:
    runtime = OptimizerRuntime(
        settings=settings,
        workspace=workspace,
        harness_label="Claude Agent SDK",
    )
    harness = ClaudeAgentSdkHarness(
        runtime=runtime,
        settings=settings,
        mode=mode,
        client_factory=client_factory,
    )

    async def run_claude_sdk(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        del agent_info
        result = await harness.run(_latest_user_prompt(messages))
        return ModelResponse(parts=[TextPart(content=result or "Claude Agent SDK completed.")])

    return Agent(
        FunctionModel(run_claude_sdk, model_name="claude-agent-sdk"),
        name="triton-claude-sdk-optimizer",
        output_type=[str, DeferredToolRequests],
        instructions=optimizer_instructions(
            settings,
            workspace,
            harness_label="Claude Agent SDK",
        ),
    )


def create_optimizer_mcp_server(runtime: OptimizerRuntime) -> Any:
    sdk = _import_sdk()
    tool = sdk.tool
    create_sdk_mcp_server = sdk.create_sdk_mcp_server
    tool_annotations = sdk.ToolAnnotations

    def response(text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}]}

    def error_response(exc: Exception) -> dict[str, Any]:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
        }

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            method = getattr(runtime, name)
            return response(method(**args))
        except Exception as exc:
            return error_response(exc)

    tools = []

    def add_tool(
        name: str,
        description: str,
        schema: dict[str, Any],
        *,
        read_only: bool,
    ) -> None:
        @tool(
            name,
            description,
            schema,
            annotations=tool_annotations(
                readOnlyHint=read_only,
                destructiveHint=not read_only,
                openWorldHint=False,
            ),
        )
        async def handler(args: dict[str, Any], *, _name: str = name) -> dict[str, Any]:
            return await call(_name, args)

        tools.append(handler)

    add_tool(
        "describe_optimizer_surface",
        OptimizerRuntime.describe_optimizer_surface.__doc__ or "",
        {"type": "object", "properties": {}, "required": []},
        read_only=True,
    )
    add_tool(
        "list_benchmarks",
        OptimizerRuntime.list_benchmarks.__doc__ or "",
        {"type": "object", "properties": {}, "required": []},
        read_only=True,
    )
    add_tool(
        "register_kernel",
        OptimizerRuntime.register_kernel.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "kernel_id": {"type": "string"},
                "name": {"type": "string"},
                "path": {"type": "string"},
                "entrypoint": {"type": "string"},
                "shapes_json": {"type": "string", "default": "[]"},
                "dtypes_json": {"type": "string", "default": "[]"},
            },
            "required": ["kernel_id", "name", "path", "entrypoint"],
        },
        read_only=False,
    )
    add_tool(
        "start_episode",
        OptimizerRuntime.start_episode.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "kernel_id": {"type": "string"},
                "objective": {"type": "string"},
                "budget_json": {"type": "string", "default": "{}"},
            },
            "required": ["kernel_id", "objective"],
        },
        read_only=False,
    )
    add_tool(
        "compile_baseline",
        OptimizerRuntime.compile_baseline.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "kernel_id": {"type": "string"},
                "meta_json": {"type": "string", "default": "{}"},
            },
            "required": ["kernel_id"],
        },
        read_only=True,
    )
    add_tool(
        "inspect_ir",
        OptimizerRuntime.inspect_ir.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "stage": {"type": "string", "default": "ttgir"},
                "max_chars": {"type": "integer", "default": 6000},
            },
            "required": ["run_id"],
        },
        read_only=True,
    )
    add_tool(
        "summarize_decisions",
        OptimizerRuntime.summarize_decisions.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "stage": {"type": "string", "default": "ttgir"},
            },
            "required": ["run_id"],
        },
        read_only=True,
    )
    add_tool(
        "record_hypothesis",
        OptimizerRuntime.record_hypothesis.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "statement": {"type": "string"},
                "expected_effect": {"type": "string"},
                "evidence_refs_json": {"type": "string", "default": "[]"},
            },
            "required": ["episode_id", "statement", "expected_effect"],
        },
        read_only=False,
    )
    add_tool(
        "record_reasoning_summary",
        OptimizerRuntime.record_reasoning_summary.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "summary": {"type": "string"},
                "linked_hypothesis_id": {"type": "string"},
                "linked_candidate_id": {"type": "string"},
                "evidence_refs_json": {"type": "string", "default": "[]"},
                "next_step": {"type": "string"},
            },
            "required": ["episode_id", "summary"],
        },
        read_only=False,
    )
    add_tool(
        "inspect_optimization_toolset",
        OptimizerRuntime.inspect_optimization_toolset.__doc__ or "",
        {
            "type": "object",
            "properties": {"kernel_id": {"type": "string"}},
            "required": ["kernel_id"],
        },
        read_only=True,
    )
    add_tool(
        "propose_candidates",
        OptimizerRuntime.propose_candidates.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "kernel_id": {"type": "string"},
                "objective": {"type": "string"},
                "budget": {"type": "integer", "default": 2},
                "hypothesis_id": {"type": "string"},
            },
            "required": ["kernel_id", "objective"],
        },
        read_only=True,
    )
    add_tool(
        "propose_candidate_from_toolset",
        OptimizerRuntime.propose_candidate_from_toolset.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "kernel_id": {"type": "string"},
                "kind": {"type": "string"},
                "changes_json": {"type": "string"},
                "description": {"type": "string"},
                "expected_effect": {"type": "string"},
                "hypothesis_id": {"type": "string"},
            },
            "required": ["kernel_id", "kind", "changes_json", "description", "expected_effect"],
        },
        read_only=False,
    )
    add_tool(
        "validate_candidate",
        OptimizerRuntime.validate_candidate.__doc__ or "",
        {
            "type": "object",
            "properties": {"candidate_json": {"type": "string"}},
            "required": ["candidate_json"],
        },
        read_only=True,
    )
    add_tool(
        "run_candidate",
        OptimizerRuntime.run_candidate.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "candidate_json": {"type": "string"},
                "meta_json": {"type": "string", "default": "{}"},
            },
            "required": ["candidate_json"],
        },
        read_only=False,
    )
    add_tool(
        "run_baseline_benchmark",
        OptimizerRuntime.run_baseline_benchmark.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "kernel_id": {"type": "string"},
                "workload_json": {"type": "string", "default": "{}"},
            },
            "required": ["episode_id", "kernel_id"],
        },
        read_only=False,
    )
    add_tool(
        "run_candidate_benchmark",
        OptimizerRuntime.run_candidate_benchmark.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "candidate_json": {"type": "string"},
                "workload_json": {"type": "string", "default": "{}"},
            },
            "required": ["episode_id", "candidate_json"],
        },
        read_only=False,
    )
    add_tool(
        "compare_runs",
        OptimizerRuntime.compare_runs.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "baseline_id": {"type": "string"},
                "candidate_ids_json": {"type": "string"},
            },
            "required": ["baseline_id", "candidate_ids_json"],
        },
        read_only=True,
    )
    add_tool(
        "compare_benchmarks",
        OptimizerRuntime.compare_benchmarks.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "baseline_id": {"type": "string"},
                "candidate_ids_json": {"type": "string"},
            },
            "required": ["episode_id", "baseline_id", "candidate_ids_json"],
        },
        read_only=True,
    )
    add_tool(
        "accept_or_reject_candidate",
        OptimizerRuntime.accept_or_reject_candidate.__doc__ or "",
        {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "candidate_id": {"type": "string"},
                "status": {"type": "string"},
                "rationale": {"type": "string"},
                "evidence_ids_json": {"type": "string", "default": "[]"},
                "compile_only": {"type": "boolean", "default": False},
            },
            "required": ["episode_id", "candidate_id", "status", "rationale"],
        },
        read_only=False,
    )
    add_tool(
        "write_report",
        OptimizerRuntime.write_report.__doc__ or "",
        {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}},
            "required": ["episode_id"],
        },
        read_only=False,
    )

    return create_sdk_mcp_server(name=SDK_TOOL_SERVER_NAME, version="1.0.0", tools=tools)


def _latest_user_prompt(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            parts: list[str] = []
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    parts.append(_user_content_to_text(part.content))
            if parts:
                return "\n".join(parts)
    return "Continue the optimization task."


def _user_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        return "\n".join(_user_content_to_text(item) for item in content)
    text = getattr(content, "content", None)
    if isinstance(text, str):
        return text
    return str(content)


def _message_text_chunks(message: Any) -> list[str]:
    content = getattr(message, "content", None)
    if content is None and hasattr(message, "message"):
        content = getattr(message.message, "content", None)
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    chunks: list[str] = []
    for block in content:
        text = getattr(block, "text", None) or getattr(block, "content", None)
        if isinstance(text, str):
            chunks.append(text)
    return chunks


def _import_sdk() -> Any:
    try:
        import claude_agent_sdk
    except ImportError as exc:
        raise RuntimeError(
            "Claude Agent SDK harness requires `claude-agent-sdk`. "
            "Install with `env/bin/pip install claude-agent-sdk` before selecting this harness."
        ) from exc
    return claude_agent_sdk
