"""Direct (tool-free) chat completions for the archetype harnesses.

The archetypes drive candidate generation with plain chat turns (the tool
protocol is reserved for the canonical session tools), so they need a raw
"messages in → text out" call. Model-string resolution is REUSED from the
pydantic-ai integration (`resolve_model`) rather than reimplemented: the
same `provider:model` prefixes, API-key extras, and Retry-After-aware
retrying proxy apply, so `mistral:mistral-large-latest` works wherever it
works for the `pydantic_ai` harness.

Rate limiting (P0 probe finding #7: ≥5 concurrent Mistral requests →
HTTP 429; stable at 2 workers with a 1.2 s global min-interval): every
`DirectChatLLM.generate` call passes through a PROCESS-GLOBAL throttle —
request starts are spaced ≥ `llm_min_interval_s` apart (default 1.2 s,
shared across all instances/harnesses in the process) and at most
`llm_max_concurrent` requests (default 2) are in flight per event loop.
Both knobs are configurable through `HarnessRunRequest.extra`.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

Turn = tuple[str, str]
"""One chat turn: ("user" | "assistant", text)."""

DEFAULT_MIN_INTERVAL_S = 1.2
DEFAULT_MAX_CONCURRENT = 2

_throttle_lock = threading.Lock()
_next_start = [0.0]  # monotonic time of the next allowed request start
_loop_semaphores: dict[int, asyncio.Semaphore] = {}


def reserve_request_start(min_interval: float) -> float:
    """Atomically reserve the next request-start slot.

    Returns the number of seconds the caller must wait before starting its
    request. Slots are spaced `min_interval` apart process-wide (the probe
    pattern), so N queued callers start at t, t+i, t+2i, ...
    """

    with _throttle_lock:
        now = time.monotonic()
        start_at = max(now, _next_start[0])
        _next_start[0] = start_at + float(min_interval)
        return max(0.0, start_at - now)


def _concurrency_gate(limit: int) -> asyncio.Semaphore:
    """Per-event-loop semaphore capping in-flight requests."""

    loop_id = id(asyncio.get_running_loop())
    sem = _loop_semaphores.get(loop_id)
    if sem is None:
        # One driver run uses one loop; stale entries from closed loops are
        # harmless (the dict stays tiny within a process lifetime).
        sem = asyncio.Semaphore(int(limit))
        _loop_semaphores[loop_id] = sem
    return sem


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
        # Caller-supplied provider-specific settings (e.g. anthropic_effort)
        # merge last so explicit experiment configuration always wins.
        overrides = self._extra.get("model_settings")
        if isinstance(overrides, dict):
            settings.update(overrides)
            # A None value DELETES the key: e.g. claude-fable-5 rejects
            # `temperature` as deprecated, so {"temperature": null} strips
            # it from every call regardless of harness defaults.
            for key in [k for k, v in settings.items() if v is None]:
                del settings[key]

        min_interval = float(
            self._extra.get("llm_min_interval_s", DEFAULT_MIN_INTERVAL_S)
        )
        max_concurrent = int(
            self._extra.get("llm_max_concurrent", DEFAULT_MAX_CONCURRENT)
        )
        from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

        # Rate-limit resilience (T1-lite finding): provider 429s must not
        # kill the episode — token-per-minute limits can fire regardless of
        # request spacing, so the call itself retries with exponential
        # backoff (jittered, Retry-After honored when present) and only
        # re-raises after `llm_429_retries` exhausted attempts.
        retries = int(self._extra.get("llm_429_retries", 8))
        async with _concurrency_gate(max_concurrent):
            for attempt in range(retries + 1):
                wait = reserve_request_start(min_interval)
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    response = await model_request(
                        self._resolve(), messages, model_settings=settings
                    )
                    break
                except ModelAPIError as err:
                    status = getattr(err, "status_code", None)
                    timed_out = not isinstance(err, ModelHTTPError) and (
                        "timed out" in str(err).lower()
                        or "timeout" in str(err).lower()
                    )
                    retryable = (
                        status == 429
                        or (isinstance(status, int) and status >= 500)
                        or timed_out
                    )
                    if not retryable or attempt == retries:
                        raise
                    retry_after = None
                    body = getattr(err, "body", None)
                    if isinstance(body, Mapping):
                        retry_after = body.get("retry_after")
                    delay = (
                        float(retry_after)
                        if retry_after
                        else min(90.0, 5.0 * (2.0**attempt))
                    )
                    await asyncio.sleep(
                        delay + (time.monotonic() * 997.0) % 2.0
                    )
        text = "".join(
            part.content
            for part in response.parts
            if isinstance(part, TextPart)
        )
        return text, usage_to_dict(response.usage)
