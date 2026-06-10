"""`archetype_band` — UCB1 bandit over optimization-strategy arms (E5).

Inspired by KernelBand, simplified. Documented deviations from the source
system (v0 simplifications):

  - NO hardware-profile state: KernelBand keys its bandit on NCU profile
    clusters; NCU is blocked on this machine (`RmProfilingAdminOnly=1`),
    so the arms are 5 fixed named optimization strategies with prompt
    templates instead of profile-conditioned actions.
  - reward = validated speedup delta vs the incumbent at pull time,
    clipped at 0 (failures and regressions reward 0) — raw deltas, no
    reward normalization.
  - single incumbent (greedy hill-climb on the best validated candidate),
    no candidate population.

Protocol: each pull selects an arm by UCB1 (every arm is pulled once
first, in declaration order), generates ONE candidate conditioned on the
incumbent source + the chosen strategy template, and submits it through
the canonical `propose_candidate`/`run_candidate` tool protocol. Pulls
continue until the validated-candidate budget is exhausted (or
`max_turns` pulls).
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace as dc_replace

from compilagent.harness.base import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
)
from compilagent.session.completion import RunSnapshot

from .harness import _DEFAULT_MAX_TURNS, _ArchetypeHarnessBase
from .prompts import base_prompt, extract_code

#: The 5 named optimization-strategy arms (name, prompt template).
STRATEGY_ARMS: tuple[tuple[str, str], ...] = (
    (
        "vectorize",
        "VECTORIZE: process more elements per program — widen tl.arange "
        "blocks, use wide coalesced loads/stores, cover several rows or "
        "columns per program id.",
    ),
    (
        "tile_resize",
        "TILE-RESIZE: change the tile/block decomposition — adjust BLOCK "
        "sizes along each dim, num_warps, and num_stages to better fit "
        "this shape and the GPU's occupancy limits.",
    ),
    (
        "fuse_loops",
        "FUSE-LOOPS: merge separate loops/kernels over the same data into "
        "one — compute running statistics online (e.g. streaming "
        "max+sum) so each element is visited once.",
    ),
    (
        "reduce_passes",
        "REDUCE-PASSES: cut the number of passes over global memory — "
        "keep intermediates in registers, avoid materializing "
        "temporaries, eliminate redundant loads.",
    ),
    (
        "relayout",
        "RELAYOUT: change the data access order/layout — iterate along "
        "the contiguous dimension, transpose the tiling, or stage tiles "
        "so global accesses coalesce.",
    ),
)


def ucb1_select(
    *,
    pulls: Sequence[int],
    total_rewards: Sequence[float],
    exploration: float = math.sqrt(2.0),
) -> int:
    """UCB1 arm selection (Auer et al. 2002).

    Any arm with zero pulls is selected first (in index order); afterwards
    the arm maximizing ``mean_i + c * sqrt(ln(N) / n_i)`` wins, where
    ``N = sum(pulls)``. Ties resolve to the lowest index (`max` is
    first-wins on ties).
    """

    if len(pulls) != len(total_rewards) or not pulls:
        raise ValueError("pulls and total_rewards must be equal-length, non-empty")
    for i, n in enumerate(pulls):
        if n == 0:
            return i
    total = sum(pulls)
    scores = [
        total_rewards[i] / pulls[i]
        + exploration * math.sqrt(math.log(total) / pulls[i])
        for i in range(len(pulls))
    ]
    return max(range(len(scores)), key=scores.__getitem__)


def strategy_prompt(
    *,
    reference_source: str,
    task_description: str,
    banned_patterns: list[str],
    incumbent_source: str,
    incumbent_speedup: float,
    strategy_text: str,
) -> str:
    """One pull's prompt: incumbent + the chosen strategy arm."""

    return (
        base_prompt(
            reference_source=reference_source,
            task_description=task_description,
            banned_patterns=banned_patterns,
        )
        + f"""
Current incumbent implementation (validated, speedup {incumbent_speedup:.3f}x vs eager):
```python
{incumbent_source}
```

Apply exactly this optimization strategy to the incumbent:
{strategy_text}

Output exactly ONE fenced python code block with the full improved module, nothing else.
"""
    )


class ArchetypeBanditHarness(_ArchetypeHarnessBase):
    """`archetype_band` — UCB1 over 5 named strategy arms."""

    id: str = "archetype_band"

    TEMPERATURE: float = 0.7

    async def _run_protocol(
        self, request: HarnessRunRequest
    ) -> AsyncIterator[StreamEvent]:
        toolset = request.toolset
        generate = self._generator(request)
        usage_total: dict[str, int] = {}
        llm_calls = 0

        ctx_events, context = self._load_task_context(toolset)
        for event in ctx_events:
            yield event

        max_pulls = request.max_turns or _DEFAULT_MAX_TURNS
        pulls = [0] * len(STRATEGY_ARMS)
        total_rewards = [0.0] * len(STRATEGY_ARMS)

        # Incumbent starts as the reference itself (speedup 1.0 by
        # definition); the first validated improvement replaces it.
        incumbent_source = context["reference_source"]
        incumbent_speedup = 1.0
        slots_remaining: int | None = None

        for pull in range(max_pulls):
            if slots_remaining == 0:
                break
            arm = ucb1_select(pulls=pulls, total_rewards=total_rewards)
            arm_name, arm_text = STRATEGY_ARMS[arm]
            prompt = strategy_prompt(
                reference_source=context["reference_source"],
                task_description=context["task_description"],
                banned_patterns=context["banned_patterns"],
                incumbent_source=incumbent_source,
                incumbent_speedup=incumbent_speedup,
                strategy_text=arm_text,
            )
            text, usage = await generate(
                history=[("user", prompt)],
                system=request.system_instructions or None,
                temperature=self.TEMPERATURE,
                max_tokens=request.max_tokens,
            )
            llm_calls += 1
            self._accumulate(usage_total, usage)
            # One text part per pull — `pull` doubles as the part index.
            yield StreamEvent(kind=StreamEventKind.TEXT_STARTED, part_index=pull)
            yield StreamEvent(
                kind=StreamEventKind.TEXT_DELTA, part_index=pull, text=text
            )

            reward = 0.0
            pulls[arm] += 1
            code = extract_code(text)
            if code is not None:
                events, result, _error = self._submit_candidate(
                    toolset,
                    context,
                    code,
                    description=f"archetype_band pull {pull} arm {arm_name}",
                    call_prefix=f"band-p{pull}-{arm_name}",
                )
                for event in events:
                    yield event
                if result is not None:
                    slots_remaining = result.get(
                        "slots_remaining", slots_remaining
                    )
                    speedup = result.get("speedup_vs_baseline")
                    if (
                        result.get("successful")
                        and isinstance(speedup, (int, float))
                    ):
                        reward = max(0.0, float(speedup) - incumbent_speedup)
                        if float(speedup) > incumbent_speedup:
                            incumbent_source = code
                            incumbent_speedup = float(speedup)
            total_rewards[arm] += reward

        for event in self._reflect(toolset):
            yield event
        arm_stats = {
            name: {
                "pulls": pulls[i],
                "mean_reward": (
                    total_rewards[i] / pulls[i] if pulls[i] else 0.0
                ),
            }
            for i, (name, _) in enumerate(STRATEGY_ARMS)
        }
        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text=(
                f"archetype_band finished: {llm_calls} pull(s), incumbent "
                f"speedup {incumbent_speedup:.3f}x, arms "
                f"{ {k: v['pulls'] for k, v in arm_stats.items()} }"
            ),
            extra={
                "usage": dict(usage_total),
                "llm_calls": llm_calls,
                "arms": arm_stats,
                "incumbent_speedup": incumbent_speedup,
            },
        )

    def build_continuation_request(
        self,
        previous: HarnessRunRequest,
        snapshot: RunSnapshot,
    ) -> HarnessRunRequest:
        """Continuations restart the bandit with fresh arm statistics (the
        documented norm — no state survives a continuation; the session
        leaderboard keeps the validated winners)."""

        remaining = max(0, snapshot.max_candidates - snapshot.successful_count)
        return dc_replace(
            previous,
            user_prompt=f"Bandit continuation: {remaining} validated slot(s) remain.",
        )
