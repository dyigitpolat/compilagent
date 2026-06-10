"""Direct (tool-free) chat completions for the archetype harnesses.

The archetypes drive candidate generation with plain chat turns (the tool
protocol is reserved for the canonical session tools), so they need a raw
"messages in → text out" call. Model-string resolution is REUSED from the
pydantic-ai integration (`resolve_model`) rather than reimplemented: the
same `provider:model` prefixes, API-key extras, and Retry-After-aware
retrying proxy apply, so `mistral:mistral-large-latest` works wherever it
works for the `pydantic_ai` harness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Turn = tuple[str, str]
"""One chat turn: ("user" | "assistant", text)."""


def usage_to_dict(usage: Any) -> dict[str, int]:
    """Normalise pydantic-ai `RequestUsage` to the harness metadata shape
    (`request_tokens` / `response_tokens` / `total_tokens` — matching what
    the pydantic_ai harness reports, so suite rows pick tokens up the same
    way for every harness)."""

    request = int(getattr(usage, "input_tokens", 0) or 0)
    response = int(getattr(usage, "output_tokens", 0) or 0)
    return {
        "request_tokens": request,
        "response_tokens": response,
        "total_tokens": request + response,
    }


class DirectChatLLM:
    """Thin async chat-completion wrapper over a resolved pydantic-ai model."""

    def __init__(self, model_id: str, extra: Mapping[str, Any]):
        self._model_id = model_id
        self._extra = dict(extra or {})
        self._model: Any = None

    def _resolve(self) -> Any:
        if self._model is None:
            # Deliberate cross-integration reuse (ticket E4): the archetype
            # harnesses must be model-agnostic via the exact same resolution
            # path as the pydantic_ai harness.
            from compilagent.integrations.pydantic_ai._model import resolve_model

            self._model = resolve_model(self._model_id, self._extra)
        return self._model

    async def generate(
        self,
        *,
        history: Sequence[Turn],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, int]]:
        """One chat completion over `history`; returns (text, usage_dict)."""

        from pydantic_ai.direct import model_request
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            SystemPromptPart,
            TextPart,
            UserPromptPart,
        )

        messages: list[Any] = []
        first_user = True
        for role, text in history:
            if role == "assistant":
                messages.append(ModelResponse(parts=[TextPart(content=text)]))
                continue
            parts: list[Any] = []
            if first_user and system:
                parts.append(SystemPromptPart(content=system))
            parts.append(UserPromptPart(content=text))
            messages.append(ModelRequest(parts=parts))
            first_user = False

        settings: dict[str, Any] = {"temperature": float(temperature)}
        if max_tokens is not None:
            settings["max_tokens"] = int(max_tokens)

        response = await model_request(
            self._resolve(), messages, model_settings=settings
        )
        text = "".join(
            part.content
            for part in response.parts
            if isinstance(part, TextPart)
        )
        return text, usage_to_dict(response.usage)
