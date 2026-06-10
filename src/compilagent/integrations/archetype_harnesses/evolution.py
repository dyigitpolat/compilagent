"""`archetype_evo` — population-based evolutionary archetype (ticket E5).

Inspired by EvoEngineer's evolve-with-an-archive protocol. Documented
deviations from the source system (v0 simplifications):

  - SINGLE model: one chat model plays seeder, crossover, and mutation
    roles (EvoEngineer ensembles roles across models).
  - NO soft-verifier: every parseable child goes straight to the hardened
    gates via the canonical `propose_candidate`/`run_candidate` tool
    protocol — there is no LLM pre-screen of children before hardware
    evaluation.
  - in-run archive only (no cross-task persistence; that is CASCADE's C6).
  - fixed operators: each generation produces two children from the
    contrastive parent pair — one crossover(best, divergent) and one
    mutation(best).

Protocol:

  1. SEED: P=4 independent generations at temperature 1.0 with
     menu-dropout prompt variation (each sample sees a random subset of
     `OPTIMIZATION_MENU`), all submitted through the canonical tool
     protocol.
  2. GENERATIONS until the validated-candidate budget is exhausted (or
     `max_turns` generations): parent selection from the in-run archive —
     top-K=4 by validated speedup plus 1 divergent member (the entry
     textually farthest from the best by identifier-token Jaccard
     distance, preferring validated entries); the crossover/mutation
     prompts present the (best-overall, divergent) contrastive PAIR with
     their measured results; children are validated/timed via the
     canonical tool protocol. When no validated parent exists yet, the
     generation falls back to one fresh menu-dropout seed sample.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any

from compilagent.harness.base import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
)
from compilagent.session.completion import RunSnapshot

from .harness import _DEFAULT_MAX_TURNS, _ArchetypeHarnessBase
from .prompts import base_prompt

_DEFAULT_POPULATION = 4
_DEFAULT_TOP_K = 4


# --------------------------------------------------------------- archive


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One evaluated candidate in the in-run archive."""

    candidate_id: str
    source: str
    speedup: float | None
    correctness_ok: bool | None
    summary: str
    """One-line measured-result description used in contrastive prompts."""

    @property
    def validated(self) -> bool:
        return self.speedup is not None and self.correctness_ok is not False


def _identifier_tokens(source: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source))


def token_set_distance(a: str, b: str) -> float:
    """Jaccard distance over identifier tokens — cheap structural
    divergence proxy for picking the archive's divergent member."""

    ta, tb = _identifier_tokens(a), _identifier_tokens(b)
    if not ta and not tb:
        return 0.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def select_parents(
    entries: list[ArchiveEntry], *, top_k: int = _DEFAULT_TOP_K
) -> tuple[list[ArchiveEntry], ArchiveEntry | None]:
    """Archive parent selection: top-K by validated speedup + 1 divergent.

    The divergent member maximizes `token_set_distance` from the best
    entry; validated entries are preferred for that slot, falling back to
    any non-best entry (a divergent failure still carries contrast).
    Returns ``([], None)`` when nothing has validated yet.
    """

    validated = sorted(
        (e for e in entries if e.validated),
        key=lambda e: e.speedup or 0.0,
        reverse=True,
    )
    top = validated[:top_k]
    if not top:
        return [], None
    best = top[0]
    pool = [e for e in validated if e is not best] or [
        e for e in entries if e is not best
    ]
    if not pool:
        return top, None
    divergent = max(pool, key=lambda e: token_set_distance(best.source, e.source))
    return top, divergent


def _result_summary(result: dict[str, Any]) -> str:
    if not result.get("compile_ok"):
        return f"failed to run: {str(result.get('compile_diagnostics'))[:200]}"
    if result.get("correctness_ok") is False:
        gate_lines = [
            w for w in (result.get("compile_warnings") or []) if "FAILED" in str(w)
        ]
        return f"failed correctness gate: {' '.join(gate_lines)[:200]}"
    speedup = result.get("speedup_vs_baseline")
    median = result.get("median_ms")
    if isinstance(speedup, (int, float)) and isinstance(median, (int, float)):
        # "baseline" not "eager": the comparison point is the backend's
        # empty-plan compile (eager reference on triton_source, the stock
        # compiler heuristics on lever backends).
        return f"CORRECT, {median:.4f} ms, speedup {speedup:.3f}x vs baseline"
    return "ran but produced no timing signal"


def contrastive_pair_prompt(
    *,
    reference_source: str,
    task_description: str,
    banned_patterns: list[str],
    best: ArchiveEntry,
    divergent: ArchiveEntry,
    mode: str,
) -> str:
    """Crossover/mutation prompt presenting the contrastive parent pair
    with their measured results."""

    instruction = {
        "crossover": (
            "Produce a CROSSOVER child: combine the strengths of parent A "
            "and parent B into one faster kernel."
        ),
        "mutation": (
            "Produce a MUTATION child: keep parent A's working core but "
            "change one significant implementation decision (tiling, "
            "vectorization width, pass structure, layout) to try to beat it."
        ),
    }[mode]
    return (
        base_prompt(
            reference_source=reference_source,
            task_description=task_description,
            banned_patterns=banned_patterns,
        )
        + f"""
Two parent implementations from the current population, with measured results:

Parent A (best overall) — {best.summary}:
```python
{best.source}
```

Parent B (divergent) — {divergent.summary}:
```python
{divergent.source}
```

{instruction}
Output exactly ONE fenced python code block with the full module, nothing else.
"""
    )


# --------------------------------------------------------------- harness


class ArchetypeEvolutionHarness(_ArchetypeHarnessBase):
    """`archetype_evo` — seed population + archive-guided generations."""

    id: str = "archetype_evo"

    SEED_TEMPERATURE: float = 1.0
    CHILD_TEMPERATURE: float = 0.8

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
        codec = context["codec"]

        rng = random.Random(request.extra.get("seed"))
        budget = int(request.extra.get("max_candidates", _DEFAULT_POPULATION))
        population = int(
            request.extra.get("population_size", min(_DEFAULT_POPULATION, budget))
        )
        top_k = int(request.extra.get("top_k", _DEFAULT_TOP_K))
        max_generations = request.max_turns or _DEFAULT_MAX_TURNS

        archive: list[ArchiveEntry] = []
        slots_remaining: int | None = None

        def _seed_prompt() -> str:
            return codec.menu_dropout_prompt(context, rng=rng)

        # ---- 1. SEED round: P diverse parallel samples at temp 1.0 ----
        seed_prompts = [_seed_prompt() for _ in range(population)]
        generations = await asyncio.gather(
            *(
                generate(
                    history=[("user", prompt)],
                    system=request.system_instructions or None,
                    temperature=self.SEED_TEMPERATURE,
                    max_tokens=request.max_tokens,
                )
                for prompt in seed_prompts
            )
        )
        llm_calls += len(generations)
        for idx, (text, usage) in enumerate(generations):
            self._accumulate(usage_total, usage)
            yield StreamEvent(kind=StreamEventKind.TEXT_STARTED, part_index=part)
            yield StreamEvent(
                kind=StreamEventKind.TEXT_DELTA, part_index=part, text=text
            )
            part += 1
            code = codec.extract(text)
            if code is None:
                continue
            events, result, _error = self._submit_candidate(
                toolset,
                context,
                code,
                description=f"archetype_evo seed {idx}",
                call_prefix=f"evo-seed-{idx}",
            )
            for event in events:
                yield event
            if result is None:
                continue
            archive.append(
                ArchiveEntry(
                    candidate_id=str(result.get("candidate_id")),
                    source=code,
                    speedup=result.get("speedup_vs_baseline"),
                    correctness_ok=result.get("correctness_ok"),
                    summary=_result_summary(result),
                )
            )
            slots_remaining = result.get("slots_remaining", slots_remaining)

        # ---- 2. GENERATIONS until budget (or generation cap) ----
        generation = 0
        while (
            (slots_remaining is None or slots_remaining > 0)
            and generation < max_generations
        ):
            generation += 1
            top, divergent = select_parents(archive, top_k=top_k)
            if top:
                best = top[0]
                child_prompts = [
                    (
                        mode,
                        codec.contrastive_pair_prompt(
                            context,
                            best=best,
                            divergent=divergent or best,
                            mode=mode,
                        ),
                    )
                    for mode in ("crossover", "mutation")
                ]
            else:
                # Nothing validated yet: re-seed instead of breeding.
                child_prompts = [("seed", _seed_prompt())]

            for mode, prompt in child_prompts:
                if slots_remaining == 0:
                    break
                text, usage = await generate(
                    history=[("user", prompt)],
                    system=request.system_instructions or None,
                    temperature=(
                        self.SEED_TEMPERATURE
                        if mode == "seed"
                        else self.CHILD_TEMPERATURE
                    ),
                    max_tokens=request.max_tokens,
                )
                llm_calls += 1
                self._accumulate(usage_total, usage)
                yield StreamEvent(
                    kind=StreamEventKind.TEXT_STARTED, part_index=part
                )
                yield StreamEvent(
                    kind=StreamEventKind.TEXT_DELTA, part_index=part, text=text
                )
                part += 1
                code = codec.extract(text)
                if code is None:
                    continue
                tag = f"gen {generation} {mode}"
                events, result, _error = self._submit_candidate(
                    toolset,
                    context,
                    code,
                    description=f"archetype_evo {tag}",
                    call_prefix=f"evo-g{generation}-{mode}",
                )
                for event in events:
                    yield event
                if result is None:
                    continue
                archive.append(
                    ArchiveEntry(
                        candidate_id=str(result.get("candidate_id")),
                        source=code,
                        speedup=result.get("speedup_vs_baseline"),
                        correctness_ok=result.get("correctness_ok"),
                        summary=_result_summary(result),
                    )
                )
                slots_remaining = result.get("slots_remaining", slots_remaining)

        for event in self._reflect(toolset):
            yield event
        best_entry = max(
            (e for e in archive if e.validated),
            key=lambda e: e.speedup or 0.0,
            default=None,
        )
        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text=(
                f"archetype_evo finished: {llm_calls} generation call(s), "
                f"{generation} generation(s) after a population of "
                f"{population}, archive size {len(archive)}, best "
                f"{best_entry.speedup if best_entry else None}"
            ),
            extra={
                "usage": dict(usage_total),
                "llm_calls": llm_calls,
                "archive_size": len(archive),
                "generations": generation,
            },
        )

    def build_continuation_request(
        self,
        previous: HarnessRunRequest,
        snapshot: RunSnapshot,
    ) -> HarnessRunRequest:
        """Continuations restart with a fresh population sized to the
        remaining budget (in-run archive does not survive — the session
        leaderboard keeps everything learned so far)."""

        remaining = max(0, snapshot.max_candidates - snapshot.successful_count)
        extra = {
            **dict(previous.extra),
            "max_candidates": remaining,
            "population_size": max(1, min(_DEFAULT_POPULATION, remaining)),
        }
        return dc_replace(
            previous,
            user_prompt=f"Evolution continuation: {remaining} validated slot(s) remain.",
            extra=extra,
        )
