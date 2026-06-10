"""Archetype harnesses (ticket E4): `archetype_sr` and `archetype_bon`.

Two minimal, published-protocol agent archetypes implemented as core
`Harness`es, so they can be compared against the full tool-using agent under
the SAME session, budget ledger, and hardened verifier:

  - `archetype_sr` — serial refinement (the KernelBench G+E protocol): one
    chat chain; each turn the LLM sees the reference + task + its previous
    candidate + a structured verdict (compile error / failing gate + message
    / "correct but X ms vs Y ms — make it faster") and emits a full module.
    Refinement depth is bounded by the session's `max_candidates` budget
    (the chain stops when `slots_remaining == 0`); keep-best semantics come
    from the session leaderboard.
  - `archetype_bon` — parallel best-of-N: N = `max_candidates` independent
    generations at temperature 1.0 from the same base prompt (no feedback);
    every parseable sample is submitted and the session leaderboard picks
    the best validated candidate.

Both harnesses:

  - are model-agnostic via the SAME model-string resolution the pydantic_ai
    integration uses (`mistral:mistral-large-latest` works identically);
  - generate candidates with direct chat calls but submit them through the
    canonical session tool protocol (`propose_candidate` / `run_candidate`),
    emitting the normal `StreamEvent`s so traces and observation work;
  - accumulate tokens in/out per LLM call into `RUN_FINISHED.extra["usage"]`
    (the same shape the pydantic_ai harness reports), plus `llm_calls`;
  - end each protocol with `compare_runs` + `synthesize_findings` so the
    `DefaultCompletionPolicy` reflection lock is satisfied;
  - respect the budget-ledger rule that a timing invocation only happens for
    gate-passing candidates. With the `triton_source` backend this holds by
    construction (compile() runs all gates before any timing rep), and the
    harness asserts the invariant on every run result it sees from that
    backend.

Prompting mirrors the P0 probe (one in-context vector-add Triton exemplar +
the probe's exact wording) — see `prompts.py`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

from compilagent.harness.base import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
)
from compilagent.session.completion import RunSnapshot
from compilagent.toolset import Toolset

from ._llm import DirectChatLLM, Turn
from .prompts import (
    RETRY_FORMAT_MESSAGE,
    base_prompt,
    extract_code,
    feedback_for_run_result,
    rejection_feedback,
)

GenerateFn = Callable[..., Awaitable[tuple[str, dict[str, int]]]]

_DEFAULT_MAX_TURNS = 12
_DEFAULT_BEST_OF_N = 4


class _ToolCallOutcome:
    __slots__ = ("events", "result", "error")

    def __init__(
        self,
        events: list[StreamEvent],
        result: str | None,
        error: str | None,
    ) -> None:
        self.events = events
        self.result = result
        self.error = error


class _ArchetypeHarnessBase:
    """Shared plumbing: tool dispatch, task context, token accounting."""

    supported_providers: tuple[str, ...] = ("anthropic", "mistral", "openai")
    example_models: tuple[str, ...] = (
        "mistral:mistral-large-latest",
        "anthropic:claude-opus-4-7",
    )

    def __init__(self, generate_fn: GenerateFn | None = None) -> None:
        # Test seam: inject a fake generator; production resolves the model
        # id through the pydantic_ai integration's `resolve_model`.
        self._generate_override = generate_fn

    # ---- LLM ------------------------------------------------------------------

    def _generator(self, request: HarnessRunRequest) -> GenerateFn:
        if self._generate_override is not None:
            return self._generate_override
        return DirectChatLLM(request.model_id, request.extra).generate

    @staticmethod
    def _accumulate(total: dict[str, int], usage: dict[str, int]) -> None:
        for key in ("request_tokens", "response_tokens", "total_tokens"):
            total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)

    # ---- tools ----------------------------------------------------------------

    @staticmethod
    def _call_tool(
        toolset: Toolset,
        name: str,
        args: dict[str, Any],
        call_id: str,
    ) -> _ToolCallOutcome:
        """Invoke a session tool, packaging the canonical event pair.

        `ValueError` (the retryable tool-error contract) becomes a
        TOOL_ERROR event + an error string the caller can fold into
        feedback; other exceptions propagate (and end the run as
        RUN_FAILED).
        """

        events = [
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL,
                tool_name=name,
                tool_call_id=call_id,
                tool_args=args,
            )
        ]
        try:
            result = toolset.by_name(name).invoke(args)
        except ValueError as exc:
            events.append(
                StreamEvent(
                    kind=StreamEventKind.TOOL_ERROR,
                    tool_name=name,
                    tool_call_id=call_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            return _ToolCallOutcome(events, None, str(exc))
        events.append(
            StreamEvent(
                kind=StreamEventKind.TOOL_RESULT,
                tool_name=name,
                tool_call_id=call_id,
                tool_result=result,
            )
        )
        return _ToolCallOutcome(events, result, None)

    def _load_task_context(
        self, toolset: Toolset
    ) -> tuple[list[StreamEvent], dict[str, Any]]:
        """inspect_workload (+ read_reference_source when the backend ships
        it) → {reference_source, task_description, banned_patterns, ...}."""

        events: list[StreamEvent] = []
        outcome = self._call_tool(toolset, "inspect_workload", {}, "ctx-1")
        events.extend(outcome.events)
        info = json.loads(outcome.result or "{}")
        workload = info.get("workload") or {}
        metadata = workload.get("metadata") or {}
        context: dict[str, Any] = {
            "workload_id": workload.get("id", ""),
            "backend_id": info.get("backend_id", ""),
            "task_description": workload.get("description", ""),
            "reference_source": str(
                metadata.get("reference_module_source", "") or ""
            ),
            "banned_patterns": list(metadata.get("banned_patterns", []) or []),
            "baseline_median_ms": (info.get("baseline_timing") or {}).get(
                "median_ms"
            ),
            # Cross-run policy hints (rationale strings) surfaced by
            # `inspect_workload` — CASCADE's C6 injects these into prompts.
            "prior_hints": [
                str(h.get("rationale", "") or "")
                for h in (info.get("prior_hints") or [])
                if isinstance(h, dict)
            ],
        }
        if "read_reference_source" in toolset.names():
            outcome = self._call_tool(
                toolset, "read_reference_source", {}, "ctx-2"
            )
            events.extend(outcome.events)
            if outcome.result:
                ref = json.loads(outcome.result)
                context["reference_source"] = ref.get(
                    "reference_module_source", context["reference_source"]
                )
                context["banned_patterns"] = list(
                    ref.get("banned_patterns", context["banned_patterns"])
                )
                context["task_description"] = ref.get(
                    "task_description", context["task_description"]
                )
        return events, context

    @staticmethod
    def _check_ledger_invariant(
        context: dict[str, Any], run_result: dict[str, Any]
    ) -> None:
        """Budget-ledger invariant (triton_source): gate-failing candidates
        are never timed, because compile() runs all gates before any timing
        rep. Asserted here so a regression in the backend surfaces loudly."""

        if context.get("backend_id") != "triton_source":
            return
        if run_result.get("correctness_ok") is False:
            assert run_result.get("median_ms") is None, (
                "budget-ledger violation: a gate-failing candidate carries a "
                f"timing signal: {run_result}"
            )

    @staticmethod
    def _propose_args(code: str, *, description: str) -> dict[str, Any]:
        return {
            "interventions": [
                {
                    "target_kind": "source_replace",
                    "target_selector": "kernel_source",
                    "payload": {"module_source": code},
                    "rationale": description,
                }
            ],
            "description": description,
            "expected_effect": "",
        }

    def _submit_candidate(
        self,
        toolset: Toolset,
        context: dict[str, Any],
        code: str,
        *,
        description: str,
        call_prefix: str,
    ) -> tuple[list[StreamEvent], dict[str, Any] | None, str | None]:
        """propose_candidate + run_candidate for one module source.

        Returns ``(events, run_result, error)`` where exactly one of
        `run_result` / `error` is set. The budget-ledger invariant is
        asserted on every parsed result (triton_source backends only).
        """

        events: list[StreamEvent] = []
        outcome = self._call_tool(
            toolset,
            "propose_candidate",
            self._propose_args(code, description=description),
            f"{call_prefix}-propose",
        )
        events.extend(outcome.events)
        if outcome.error is not None:
            return events, None, outcome.error
        candidate_id = json.loads(outcome.result or "{}").get("id")
        outcome = self._call_tool(
            toolset,
            "run_candidate",
            {"candidate_id": candidate_id},
            f"{call_prefix}-run",
        )
        events.extend(outcome.events)
        if outcome.error is not None:
            return events, None, outcome.error
        result = json.loads(outcome.result or "{}")
        self._check_ledger_invariant(context, result)
        return events, result, None

    def _reflect(self, toolset: Toolset) -> list[StreamEvent]:
        """compare_runs + synthesize_findings — the DefaultCompletionPolicy
        refuses to close the run until both reflection tools have fired."""

        events: list[StreamEvent] = []
        events.extend(self._call_tool(toolset, "compare_runs", {}, "refl-1").events)
        events.extend(
            self._call_tool(toolset, "synthesize_findings", {}, "refl-2").events
        )
        return events

    # ---- outer run wrapper ------------------------------------------------------

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        try:
            async for event in self._run_protocol(request):
                yield event
        except Exception as exc:  # noqa: BLE001
            yield StreamEvent(
                kind=StreamEventKind.RUN_FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    async def _run_protocol(
        self, request: HarnessRunRequest
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError
        yield  # pragma: no cover


class ArchetypeSerialRefinementHarness(_ArchetypeHarnessBase):
    """`archetype_sr` — serial refinement (KernelBench G+E)."""

    id: str = "archetype_sr"

    async def _run_protocol(
        self, request: HarnessRunRequest
    ) -> AsyncIterator[StreamEvent]:
        toolset = request.toolset
        generate = self._generator(request)
        usage_total: dict[str, int] = {}
        llm_calls = 0
        part = 0

        ctx_events, context = self._load_task_context(toolset)
        for event in ctx_events:
            yield event

        history: list[Turn] = [
            (
                "user",
                base_prompt(
                    reference_source=context["reference_source"],
                    task_description=context["task_description"],
                    banned_patterns=context["banned_patterns"],
                ),
            )
        ]
        max_turns = request.max_turns or _DEFAULT_MAX_TURNS
        slots_remaining: int | None = None
        last_result: dict[str, Any] | None = None

        for turn in range(max_turns):
            # Probe-mirroring temperatures: 0.7 on the first generation,
            # 0.5 on refinements.
            text, usage = await generate(
                history=history,
                system=request.system_instructions or None,
                temperature=0.7 if turn == 0 else 0.5,
                max_tokens=request.max_tokens,
            )
            llm_calls += 1
            self._accumulate(usage_total, usage)
            yield StreamEvent(kind=StreamEventKind.TEXT_STARTED, part_index=part)
            yield StreamEvent(
                kind=StreamEventKind.TEXT_DELTA, part_index=part, text=text
            )
            part += 1

            code = extract_code(text)
            if code is None:
                history += [("assistant", text), ("user", RETRY_FORMAT_MESSAGE)]
                continue

            outcome = self._call_tool(
                toolset,
                "propose_candidate",
                self._propose_args(code, description=f"archetype_sr turn {turn}"),
                f"sr-propose-{turn}",
            )
            for event in outcome.events:
                yield event
            if outcome.error is not None:
                history += [
                    ("assistant", text),
                    ("user", rejection_feedback(outcome.error)),
                ]
                continue
            candidate_id = json.loads(outcome.result or "{}").get("id")

            outcome = self._call_tool(
                toolset,
                "run_candidate",
                {"candidate_id": candidate_id},
                f"sr-run-{turn}",
            )
            for event in outcome.events:
                yield event
            if outcome.error is not None:
                history += [
                    ("assistant", text),
                    ("user", rejection_feedback(outcome.error)),
                ]
                continue

            result = json.loads(outcome.result or "{}")
            self._check_ledger_invariant(context, result)
            last_result = result
            slots_remaining = result.get("slots_remaining")
            history += [
                ("assistant", text),
                ("user", feedback_for_run_result(result)),
            ]
            if slots_remaining == 0:
                break

        for event in self._reflect(toolset):
            yield event
        final = (
            f"archetype_sr finished: {llm_calls} generation(s), "
            f"slots_remaining={slots_remaining}, "
            f"last_speedup={last_result.get('speedup_vs_baseline') if last_result else None}"
        )
        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text=final,
            extra={"usage": dict(usage_total), "llm_calls": llm_calls},
        )

    def build_continuation_request(
        self,
        previous: HarnessRunRequest,
        snapshot: RunSnapshot,
    ) -> HarnessRunRequest:
        """Continuations restart the chain with a fresh conversation (the
        documented norm — no transcript resumption across iterations); the
        session leaderboard keeps everything learned so far."""

        return replace(
            previous,
            user_prompt=(
                "Continue refining: "
                f"{max(0, snapshot.max_candidates - snapshot.successful_count)} "
                "validated slot(s) remain."
            ),
        )


class ArchetypeBestOfNHarness(_ArchetypeHarnessBase):
    """`archetype_bon` — parallel best-of-N, no feedback."""

    id: str = "archetype_bon"

    #: Generation temperature is part of the archetype's definition.
    TEMPERATURE: float = 1.0

    async def _run_protocol(
        self, request: HarnessRunRequest
    ) -> AsyncIterator[StreamEvent]:
        toolset = request.toolset
        generate = self._generator(request)
        usage_total: dict[str, int] = {}
        part = 0

        ctx_events, context = self._load_task_context(toolset)
        for event in ctx_events:
            yield event

        n = int(request.extra.get("max_candidates", _DEFAULT_BEST_OF_N))
        prompt: list[Turn] = [
            (
                "user",
                base_prompt(
                    reference_source=context["reference_source"],
                    task_description=context["task_description"],
                    banned_patterns=context["banned_patterns"],
                ),
            )
        ]

        # N independent samples from the SAME base prompt, temperature 1.0.
        generations = await asyncio.gather(
            *(
                generate(
                    history=prompt,
                    system=request.system_instructions or None,
                    temperature=self.TEMPERATURE,
                    max_tokens=request.max_tokens,
                )
                for _ in range(n)
            )
        )
        llm_calls = len(generations)
        submitted = 0
        for idx, (text, usage) in enumerate(generations):
            self._accumulate(usage_total, usage)
            yield StreamEvent(kind=StreamEventKind.TEXT_STARTED, part_index=part)
            yield StreamEvent(
                kind=StreamEventKind.TEXT_DELTA, part_index=part, text=text
            )
            part += 1

            code = extract_code(text)
            if code is None:
                continue
            outcome = self._call_tool(
                toolset,
                "propose_candidate",
                self._propose_args(code, description=f"archetype_bon sample {idx}"),
                f"bon-propose-{idx}",
            )
            for event in outcome.events:
                yield event
            if outcome.error is not None:
                continue
            candidate_id = json.loads(outcome.result or "{}").get("id")
            outcome = self._call_tool(
                toolset,
                "run_candidate",
                {"candidate_id": candidate_id},
                f"bon-run-{idx}",
            )
            for event in outcome.events:
                yield event
            if outcome.error is None:
                self._check_ledger_invariant(
                    context, json.loads(outcome.result or "{}")
                )
                submitted += 1

        for event in self._reflect(toolset):
            yield event
        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text=(
                f"archetype_bon finished: {llm_calls} independent sample(s), "
                f"{submitted} ran; the leaderboard picks the best validated."
            ),
            extra={"usage": dict(usage_total), "llm_calls": llm_calls},
        )

    def build_continuation_request(
        self,
        previous: HarnessRunRequest,
        snapshot: RunSnapshot,
    ) -> HarnessRunRequest:
        """A continuation samples only the REMAINING budget (fresh
        independent generations — still no feedback)."""

        remaining = max(0, snapshot.max_candidates - snapshot.successful_count)
        extra = {**dict(previous.extra), "max_candidates": remaining}
        return replace(
            previous,
            user_prompt=f"Best-of-N continuation: sample {remaining} more.",
            extra=extra,
        )
