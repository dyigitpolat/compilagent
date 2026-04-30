"""Resolve a `model_id` string + `extra` knobs into a pydantic-ai `Model`.

The harness reads `request.model_id` (e.g. `"anthropic:claude-opus-4-7"`,
`"mistral:mistral-large-latest"`, `"openai:gpt-4o"`, or `"test"`) and
provider API keys from `request.extra` (which the harness adapter is free
to populate from settings before construction).

The returned model is wrapped in a retrying proxy that retries 4xx/5xx HTTP
errors with exponential backoff. Retry parameters come from `extra` keys
(`retry_max_attempts`, `retry_base_seconds`, `retry_max_seconds`).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None) or {}
    if isinstance(headers, dict):
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra is not None:
            try:
                return float(ra)
            except (TypeError, ValueError):
                return None
    return None


def _build_retrying_model(
    wrapped: Any,
    *,
    max_attempts: int,
    base_seconds: float,
    max_seconds: float,
) -> Any:
    """Wrap a pydantic-ai Model in a retrying proxy.

    Built dynamically because pydantic-ai's `WrapperModel` is only available
    at runtime and the constructed proxy must remain a real `Model` so the
    Agent's isinstance check passes.
    """

    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.models.wrapper import WrapperModel

    def _is_retryable(exc: Exception) -> bool:
        return (
            isinstance(exc, ModelHTTPError)
            and getattr(exc, "status_code", None) in _RETRY_STATUS
        )

    async def _sleep(attempt: int, exc: Exception) -> None:
        hint = _retry_after_seconds(exc)
        if hint is not None and hint > 0:
            wait = min(hint, max_seconds)
        else:
            cap = min(max_seconds, base_seconds * (2**attempt))
            wait = random.uniform(base_seconds, max(base_seconds, cap))
        await asyncio.sleep(wait)

    class RetryingModel(WrapperModel):
        async def request(self, messages, model_settings, model_request_parameters):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await self.wrapped.request(
                        messages, model_settings, model_request_parameters
                    )
                except Exception as exc:
                    last_exc = exc
                    if not _is_retryable(exc) or attempt == max_attempts - 1:
                        raise
                    await _sleep(attempt, exc)
            assert last_exc is not None
            raise last_exc

        @asynccontextmanager
        async def request_stream(
            self,
            messages,
            model_settings,
            model_request_parameters,
            run_context=None,
        ):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    async with self.wrapped.request_stream(
                        messages,
                        model_settings,
                        model_request_parameters,
                        run_context,
                    ) as resp:
                        yield resp
                        return
                except Exception as exc:
                    last_exc = exc
                    if not _is_retryable(exc) or attempt == max_attempts - 1:
                        raise
                    await _sleep(attempt, exc)
            assert last_exc is not None
            raise last_exc

    return RetryingModel(wrapped)


def resolve_model(model_id: str, extra: Mapping[str, Any]) -> Any:
    """Build a pydantic-ai Model from a `provider:model-name` id.

    Recognised provider prefixes: `anthropic:`, `mistral:`, `openai:`. Any
    other id is passed straight to pydantic-ai (which lets it fall back to
    its registry of named models, including `test`/`test-model`).

    `extra` keys consumed:
      - `anthropic_api_key` / `mistral_api_key` / `openai_api_key`
      - `retry_max_attempts` (int, default 8)
      - `retry_base_seconds` (float, default 2.0)
      - `retry_max_seconds` (float, default 60.0)
    """

    name = model_id or ""

    if name in {"test", "test-model"}:
        from pydantic_ai.models.test import TestModel

        return TestModel()

    base: Any
    if name.startswith("anthropic:") and extra.get("anthropic_api_key"):
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        base = AnthropicModel(
            name.removeprefix("anthropic:"),
            provider=AnthropicProvider(api_key=str(extra["anthropic_api_key"])),
        )
    elif name.startswith("mistral:") and extra.get("mistral_api_key"):
        from pydantic_ai.models.mistral import MistralModel
        from pydantic_ai.providers.mistral import MistralProvider

        base = MistralModel(
            name.removeprefix("mistral:"),
            provider=MistralProvider(api_key=str(extra["mistral_api_key"])),
        )
    elif name.startswith("openai:") and extra.get("openai_api_key"):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        base = OpenAIChatModel(
            name.removeprefix("openai:"),
            provider=OpenAIProvider(api_key=str(extra["openai_api_key"])),
        )
    else:
        # Let pydantic-ai resolve by name (handles its own provider routing
        # via env vars when no key was supplied to us).
        base = name

    return _build_retrying_model(
        base,
        max_attempts=int(extra.get("retry_max_attempts", 8)),
        base_seconds=float(extra.get("retry_base_seconds", 2.0)),
        max_seconds=float(extra.get("retry_max_seconds", 60.0)),
    )


_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "extra_high": "max",
    "extra-high": "max",
    "max": "max",
}


def resolve_model_settings(
    model_id: str,
    *,
    reasoning_effort: str | None,
    max_tokens: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    """Build the `model_settings` dict pydantic-ai consumes.

    Translates our `reasoning_effort` to Anthropic's adaptive-thinking
    `output_config.effort` (exposed as `anthropic_effort`). When effort
    is set on Anthropic models, temperature must be 1.0 per Anthropic's
    API. Picking a model that supports the chosen reasoning_effort is
    the caller's responsibility — each harness advertises a tuple of
    known-good model strings via `Harness.example_models`, surfaced in
    the observation UI's model dropdown.
    """

    out: dict[str, Any] = {}
    if max_tokens is not None:
        out["max_tokens"] = int(max_tokens)

    raw = (reasoning_effort or "").strip().lower().replace("-", "_")
    effort = _EFFORT_MAP.get(raw)
    is_anthropic = (model_id or "").startswith("anthropic:")
    if is_anthropic and effort is not None:
        out["anthropic_effort"] = effort
        out["temperature"] = 1.0
    elif temperature is not None:
        out["temperature"] = float(temperature)

    # Prompt caching: a multi-turn agent loop that re-sends the same system
    # prompt + tool definitions + a growing transcript every turn pays
    # full input rate on every retransmission without `cache_control`.
    # On long Inductor/Triton sessions (24+ turns × KB-sized artifacts),
    # caching cuts input cost by ~10× and is the single biggest lever we
    # have over Anthropic billing.
    #
    # Three breakpoints (3 of Anthropic's 4 slots):
    #   - tool_definitions: 8 canonical tools, never change within a run
    #   - instructions: system prompt, never changes within a run
    #   - anthropic_cache (auto): moves forward across the growing turns
    if is_anthropic:
        out.setdefault("anthropic_cache", True)
        out.setdefault("anthropic_cache_instructions", True)
        out.setdefault("anthropic_cache_tool_definitions", True)
    return out
