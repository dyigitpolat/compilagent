"""`cascade` — CASCADE v0, the composite harness (ticket E6).

Composes the ingredient table of 04_composite_harness_design.md §2/§3 with
the following v0 substitutions (one per ingredient, each independently
disableable through the harness config — see `CascadeConfig`):

  C1  plan-then-implement: each serial turn first asks for a PLAN (one
      named optimization + a short justification, no code — in lever mode
      the plan names exactly ONE lever change), then an IMPLEMENT call
      producing the full candidate (module in source mode, intervention
      JSON in lever mode); the plan string is recorded in the Intervention
      rationale (via the propose description). Disabled → single generate
      per turn, SR-style.
  C2  staged feedback: a gate-failing candidate feeds back ONLY the
      failing gate + error; a correct candidate feeds back its timing
      delta vs the incumbent plus the timing-history table (no NCU yet —
      profiling is blocked on this machine). Disabled → the E4
      `feedback_for_run_result` verdict text.
  C3  incumbent-delta context: every serial prompt carries the current
      best candidate source + its timing + the last attempt's delta.
  C4  budget split: a seed round of k0=4 parallel samples at temperature
      1.0 (menu-dropout variation), then serial refinement with the
      remaining budget. Disabled → all-serial. Continuations skip the
      seed round (depth-only with the remaining budget).
  C5  two-phase eval: gates-before-timing already holds with
      triton_source (compile() runs all gates before any timing rep).
      v0 substitution: the fused gate+timing sandbox cannot skip timing
      for gate-passing candidates without core changes, so ALL seed
      candidates are evaluated, the model judge-ranks the gate-passers
      by PREDICTED gain (sources only, no timing shown), and only the
      judge's top-m=2 are admitted to incumbent selection / the timing
      history (emulating the budget split at selection level). Judge-rank
      vs actual-speedup pairs are recorded into the run metadata for the
      T1 side-measurement — which is exactly why the actual timings of
      all gate-passers are kept.
  C6  skill memory: rule injection happens in the prompt from the
      session's `prior_hints` (populated by `ExperimentLogPolicy.consult`
      — see `skill_memory.py`); the write path is the E9 observe hook on
      the same policy. The harness toggle controls prompt injection only;
      the policy is attached at session construction (driver's job).
  C7  novelty filter: a proposed candidate whose normalized form matches a
      previously REJECTED candidate this run is refused WITHOUT any tool
      call (zero budget), with explicit feedback to the model. The
      normalization is codec-owned (D9): source mode strips comments/
      whitespace via tokenize; lever mode canonicalizes the intervention
      JSON (rationale dropped, interventions sorted) so reworded/permuted
      duplicates collide.
  C8  keep/revert deadband: accept only >1% timing improvement over the
      incumbent; 2 consecutive non-improvements → ONE re-seed from the
      archive's contrastive pair (C10); 2 more (4 consecutive total) →
      finalize early. Failures count as non-improvements. Disabled →
      any strict improvement is accepted and the loop never stops early.
  C10 contrastive re-seed: the plateau re-seed presents the
      (best-overall, divergent) archive pair via the evolution
      integration's contrastive prompt. C10 fires on C8's plateau signal,
      so with C8 disabled it never triggers (documented dependency; 04 §2
      defines the plateau counter as C10's trigger).

T2 axis bundles map onto the toggles as: proposal={c1,c10},
feedback={c2,c3}, budget={c4,c5,c8}, memory={c6,c7} — `CascadeConfig`
accepts both individual ingredient names and bundle names in
``extra["cascade"]["disable"]``.
"""

from __future__ import annotations

import asyncio
import io
import json
import random
import re
import tokenize
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, fields
from dataclasses import replace as dc_replace
from typing import Any

from compilagent.harness.base import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
)
from compilagent.session.completion import RunSnapshot

from .evolution import (
    ArchiveEntry,
    _result_summary,
    select_parents,
)
from .harness import _DEFAULT_MAX_TURNS, _ArchetypeHarnessBase

# ------------------------------------------------------------------ config

#: T2 ablation axis bundles (06_experimental_plan.md T2).
AXIS_BUNDLES: dict[str, tuple[str, ...]] = {
    "proposal": ("c1", "c10"),
    "feedback": ("c2", "c3"),
    "budget": ("c4", "c5", "c8"),
    "memory": ("c6", "c7"),
}

_INGREDIENT_FIELDS: dict[str, str] = {
    "c1": "c1_plan_then_implement",
    "c2": "c2_staged_feedback",
    "c3": "c3_incumbent_context",
    "c4": "c4_seed_round",
    "c5": "c5_judge_ranking",
    "c6": "c6_skill_memory",
    "c7": "c7_novelty_filter",
    "c8": "c8_deadband",
    "c10": "c10_contrastive_reseed",
}


@dataclass(frozen=True, slots=True)
class CascadeConfig:
    """Per-run CASCADE configuration; every ingredient is a toggle."""

    c1_plan_then_implement: bool = True
    c2_staged_feedback: bool = True
    c3_incumbent_context: bool = True
    c4_seed_round: bool = True
    c5_judge_ranking: bool = True
    c6_skill_memory: bool = True
    c7_novelty_filter: bool = True
    c8_deadband: bool = True
    c10_contrastive_reseed: bool = True
    seed_count: int = 4
    """k0 — parallel seed-round samples (C4)."""
    judge_top_m: int = 2
    """m — judge-ranked candidates admitted past the seed round (C5)."""
    deadband_pct: float = 1.0
    """C8: accept only > this % improvement over the incumbent."""
    plateau_reseed: int = 2
    """C8: consecutive non-improvements before the single C10 re-seed."""
    plateau_stop: int = 4
    """C8: consecutive non-improvements before early finalize."""

    @classmethod
    def from_extra(cls, extra: Mapping[str, Any]) -> CascadeConfig:
        """Build from ``extra["cascade"]``: field overrides plus a
        ``disable`` list of ingredient names ("c1") or bundle names
        ("proposal")."""

        raw = dict(extra.get("cascade") or {})
        disable = raw.pop("disable", ())
        if isinstance(disable, str):
            disable = [d for d in disable.split(",") if d.strip()]
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in valid}
        config = cls(**kwargs)
        toggles_off: dict[str, bool] = {}
        for item in disable:
            key = str(item).strip().lower()
            for ingredient in AXIS_BUNDLES.get(key, (key,)):
                field_name = _INGREDIENT_FIELDS.get(ingredient)
                if field_name is None:
                    raise ValueError(
                        f"unknown cascade ingredient/bundle `{item}`; known: "
                        f"{sorted(_INGREDIENT_FIELDS)} + {sorted(AXIS_BUNDLES)}"
                    )
                toggles_off[field_name] = False
        return dc_replace(config, **toggles_off) if toggles_off else config

    def serialize(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ------------------------------------------------------------- C7 novelty


def normalize_module_source(source: str) -> str:
    """Normalize module text for the novelty filter: drop comments and
    whitespace/blank-line variation while preserving token structure
    (NEWLINE/INDENT/DEDENT become structural markers so differently
    nested code never collides)."""

    try:
        out: list[str] = []
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING):
                continue
            if tok.type == tokenize.NEWLINE:
                out.append(";")
            elif tok.type == tokenize.INDENT:
                out.append(">")
            elif tok.type == tokenize.DEDENT:
                out.append("<")
            else:
                out.append(tok.string)
        return " ".join(out)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        # Unparseable text: fall back to comment-strip + whitespace collapse.
        stripped = re.sub(r"#[^\n]*", "", source)
        return " ".join(stripped.split())


# ------------------------------------------------------------- C8 deadband


class DeadbandController:
    """C8 keep/revert accounting.

    `consider()` returns True (accept) only for a strict
    >`deadband_pct`% improvement over the incumbent; everything else —
    smaller wins, regressions, gate failures (speedup None) — counts as
    one consecutive non-improvement. The counter is NOT reset by the
    re-seed itself: re-seed fires once at `plateau_reseed` consecutive
    non-improvements, early stop at `plateau_stop` (i.e. "2 more").
    Any accepted improvement resets the counter.
    """

    def __init__(
        self,
        *,
        deadband_pct: float = 1.0,
        plateau_reseed: int = 2,
        plateau_stop: int = 4,
    ) -> None:
        self.deadband_pct = float(deadband_pct)
        self.plateau_reseed = int(plateau_reseed)
        self.plateau_stop = int(plateau_stop)
        self.non_improvements = 0
        self.reseed_used = False

    def consider(self, speedup: float | None, incumbent_speedup: float) -> bool:
        threshold = incumbent_speedup * (1.0 + self.deadband_pct / 100.0)
        accepted = speedup is not None and float(speedup) > threshold
        if accepted:
            self.non_improvements = 0
        else:
            self.non_improvements += 1
        return accepted

    @property
    def should_reseed(self) -> bool:
        return (
            not self.reseed_used
            and self.non_improvements >= self.plateau_reseed
        )

    def mark_reseeded(self) -> None:
        self.reseed_used = True

    @property
    def should_stop(self) -> bool:
        return self.non_improvements >= self.plateau_stop


# -------------------------------------------------------------- prompts


def rules_block(prior_hints: list[str]) -> str:
    """C6 — constraint rules / prior results injected by the policy."""

    if not prior_hints:
        return ""
    lines = "\n".join(f"- {hint}" for hint in prior_hints)
    return (
        "\nKnown constraints and prior results (distilled from earlier "
        f"runs — respect these):\n{lines}\n"
    )


def incumbent_block(
    incumbent: ArchiveEntry | None,
    *,
    last_delta_pct: float | None,
) -> str:
    """C3 — current best source + timing + last attempt's delta."""

    if incumbent is None:
        return (
            "\nNo candidate has validated yet — the incumbent is the "
            "PyTorch reference itself (speedup 1.000x by definition).\n"
        )
    delta_line = (
        f"Your last attempt was {last_delta_pct:+.2f}% vs this incumbent.\n"
        if last_delta_pct is not None
        else ""
    )
    return f"""
Current incumbent (best validated candidate) — {incumbent.summary}:
```python
{incumbent.source}
```
{delta_line}"""


def staged_feedback(
    result: dict[str, Any],
    *,
    incumbent_speedup: float,
    history: list[dict[str, Any]],
) -> str:
    """C2 — feedback content matched to the candidate's state.

    Gate-failing → failing gate + error ONLY. Correct → timing delta vs
    the incumbent + the timing-history table (no NCU columns yet).
    """

    if not result.get("compile_ok"):
        return (
            "Previous attempt FAILED to run. Error:\n"
            f"{result.get('compile_diagnostics')}"
        )
    if result.get("correctness_ok") is False:
        gate_lines = [
            w for w in (result.get("compile_warnings") or []) if "FAILED" in str(w)
        ]
        detail = " ".join(gate_lines) or f"max_abs_diff={result.get('max_abs_diff')}"
        return f"Previous attempt FAILED a correctness gate: {detail}"
    speedup = result.get("speedup_vs_baseline")
    median = result.get("median_ms")
    delta_pct = (
        (float(speedup) / incumbent_speedup - 1.0) * 100.0
        if isinstance(speedup, (int, float)) and incumbent_speedup
        else None
    )
    rows = "\n".join(
        f"  {h['candidate_id']}: {h['median_ms']:.4f} ms, "
        f"{h['speedup']:.3f}x, {'ACCEPTED' if h['accepted'] else 'reverted'}"
        for h in history
        if isinstance(h.get("median_ms"), (int, float))
        and isinstance(h.get("speedup"), (int, float))
    )
    return (
        "Previous attempt was CORRECT: "
        f"{median} ms, speedup {speedup}x vs baseline "
        + (
            f"({delta_pct:+.2f}% vs the incumbent).\n"
            if delta_pct is not None
            else ".\n"
        )
        + (f"Timing history this run:\n{rows}\n" if rows else "")
    )


def plan_prompt(context_block: str) -> str:
    """C1 phase 1 — one named optimization, justified, NO code."""

    return (
        context_block
        + "\nState a PLAN for the next candidate: name exactly ONE "
        "optimization to apply (e.g. 'widen rows-per-program to 4', "
        "'switch to online softmax') and justify it in at most 3 "
        "sentences. Do NOT write any code yet."
    )


def implement_prompt(context_block: str, plan: str) -> str:
    """C1 phase 2 — implement exactly the stated plan."""

    return (
        context_block
        + f"\nYour plan for this attempt:\n{plan}\n\n"
        "Implement exactly this plan. Output exactly ONE fenced python "
        "code block with the full module, nothing else."
    )


def judge_prompt(
    task_description: str, candidates: list[tuple[str, str]]
) -> str:
    """C5 — rank gate-passing seeds by PREDICTED gain (no timings shown)."""

    blocks = "\n\n".join(
        f"Candidate {cid}:\n```python\n{source}\n```"
        for cid, source in candidates
    )
    ids = [cid for cid, _ in candidates]
    return f"""You are judging Triton kernel candidates for this task: {task_description}

All candidates below passed the correctness gates. Rank them by PREDICTED
performance gain (fastest first), considering memory-access patterns,
parallelism, and passes over global memory.

{blocks}

Respond with ONLY a JSON array of candidate ids, best first, e.g.
["{ids[0]}", ...]. No other text."""


def parse_judge_ranking(text: str, candidate_ids: list[str]) -> list[str]:
    """Parse the judge's JSON array; fall back to submission order on any
    malformation. Unknown ids are dropped, missing ids appended in order."""

    match = re.search(r"\[.*?\]", text, re.DOTALL)
    ranked: list[str] = []
    if match:
        try:
            parsed = json.loads(match.group(0))
            ranked = [str(x) for x in parsed if str(x) in candidate_ids]
        except (json.JSONDecodeError, TypeError):
            ranked = []
    seen = set(ranked)
    return ranked + [cid for cid in candidate_ids if cid not in seen]


# --------------------------------------------------------------- harness


class CascadeHarness(_ArchetypeHarnessBase):
    """CASCADE v0 — composite of C1–C10 (minus C9, which is the always-on
    hardened verifier of the triton_source backend itself)."""

    id: str = "cascade"

    SEED_TEMPERATURE: float = 1.0
    PLAN_TEMPERATURE: float = 0.5
    IMPLEMENT_TEMPERATURE: float = 0.5
    JUDGE_TEMPERATURE: float = 0.2

    async def _run_protocol(
        self, request: HarnessRunRequest
    ) -> AsyncIterator[StreamEvent]:
        toolset = request.toolset
        generate = self._generator(request)
        config = CascadeConfig.from_extra(request.extra)
        usage_total: dict[str, int] = {}
        llm_calls = 0
        part = 0

        ctx_events, context = self._load_task_context(toolset)
        for event in ctx_events:
            yield event
        codec = context["codec"]

        rng = random.Random(request.extra.get("seed"))
        max_turns = request.max_turns or _DEFAULT_MAX_TURNS
        is_continuation = bool(request.extra.get("cascade_continuation"))

        archive: list[ArchiveEntry] = []
        rejected_normalized: set[str] = set()
        novelty_hits = 0
        history: list[dict[str, Any]] = []
        judge_meta: dict[str, Any] = {"ranking": [], "pairs": []}
        incumbent: ArchiveEntry | None = None
        incumbent_speedup = 1.0  # the reference, by definition
        last_feedback: str | None = None
        last_delta_pct: float | None = None
        slots_remaining: int | None = None
        deadband = DeadbandController(
            deadband_pct=config.deadband_pct,
            plateau_reseed=config.plateau_reseed,
            plateau_stop=config.plateau_stop,
        )
        accepted_count = 0

        async def _gen(prompt: str, temperature: float) -> str:
            nonlocal llm_calls
            text, usage = await generate(
                history=[("user", prompt)],
                system=request.system_instructions or None,
                temperature=temperature,
                max_tokens=request.max_tokens,
            )
            llm_calls += 1
            self._accumulate(usage_total, usage)
            return text

        def _text_events(text: str) -> list[StreamEvent]:
            nonlocal part
            events = [
                StreamEvent(kind=StreamEventKind.TEXT_STARTED, part_index=part),
                StreamEvent(
                    kind=StreamEventKind.TEXT_DELTA, part_index=part, text=text
                ),
            ]
            part += 1
            return events

        def _entry(code: str, result: dict[str, Any]) -> ArchiveEntry:
            return ArchiveEntry(
                candidate_id=str(result.get("candidate_id")),
                source=code,
                speedup=result.get("speedup_vs_baseline"),
                correctness_ok=result.get("correctness_ok"),
                summary=_result_summary(result),
            )

        def _record(result: dict[str, Any], accepted: bool) -> None:
            history.append(
                {
                    "candidate_id": str(result.get("candidate_id")),
                    "median_ms": result.get("median_ms"),
                    "speedup": result.get("speedup_vs_baseline"),
                    "accepted": accepted,
                }
            )

        # ---- C4: parallel seed round (skipped on continuations) ----
        if config.c4_seed_round and not is_continuation:
            seed_prompts = []
            for _ in range(config.seed_count):
                prompt = codec.menu_dropout_prompt(context, rng=rng)
                if config.c6_skill_memory:
                    prompt += rules_block(context.get("prior_hints") or [])
                seed_prompts.append(prompt)
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
            seed_entries: list[tuple[str, dict[str, Any]]] = []
            for idx, (text, usage) in enumerate(generations):
                self._accumulate(usage_total, usage)
                for event in _text_events(text):
                    yield event
                code = codec.extract(text)
                if code is None:
                    continue
                if (
                    config.c7_novelty_filter
                    and codec.normalize(code) in rejected_normalized
                ):
                    novelty_hits += 1
                    continue  # zero budget spent on rejected re-introductions
                events, result, _error = self._submit_candidate(
                    toolset,
                    context,
                    code,
                    description=f"cascade seed {idx}",
                    call_prefix=f"cascade-seed-{idx}",
                )
                for event in events:
                    yield event
                if result is None:
                    rejected_normalized.add(codec.normalize(code))
                    continue
                slots_remaining = result.get("slots_remaining", slots_remaining)
                archive.append(_entry(code, result))
                if result.get("successful"):
                    seed_entries.append((code, result))
                else:
                    rejected_normalized.add(codec.normalize(code))

            # ---- C5: judge-rank gate-passing seeds by predicted gain ----
            eligible = seed_entries
            if config.c5_judge_ranking and len(seed_entries) > 1:
                ids = [str(r.get("candidate_id")) for _, r in seed_entries]
                text = await _gen(
                    codec.judge_prompt(
                        context["task_description"],
                        [
                            (str(r.get("candidate_id")), code)
                            for code, r in seed_entries
                        ],
                    ),
                    self.JUDGE_TEMPERATURE,
                )
                for event in _text_events(text):
                    yield event
                ranking = parse_judge_ranking(text, ids)
                judge_meta["ranking"] = ranking
                by_id = {str(r.get("candidate_id")): (c, r) for c, r in seed_entries}
                judge_meta["pairs"] = [
                    {
                        "candidate_id": cid,
                        "judge_rank": rank + 1,
                        "actual_speedup": by_id[cid][1].get("speedup_vs_baseline"),
                    }
                    for rank, cid in enumerate(ranking)
                ]
                eligible = [by_id[cid] for cid in ranking[: config.judge_top_m]]

            for code, result in eligible:
                _record(result, accepted=False)
                speedup = result.get("speedup_vs_baseline")
                if isinstance(speedup, (int, float)) and speedup > incumbent_speedup:
                    incumbent = _entry(code, result)
                    incumbent_speedup = float(speedup)
            if incumbent is not None:
                for h in history:
                    if h["candidate_id"] == incumbent.candidate_id:
                        h["accepted"] = True
                accepted_count += 1

        # ---- serial refinement loop ----
        for turn in range(max_turns):
            if slots_remaining == 0:
                break
            if config.c8_deadband and deadband.should_stop:
                break

            # C10: one contrastive re-seed on the first plateau.
            reseeding = (
                config.c8_deadband
                and config.c10_contrastive_reseed
                and deadband.should_reseed
                and bool(select_parents(archive)[0])
            )
            if reseeding:
                top, divergent = select_parents(archive)
                prompt = codec.contrastive_pair_prompt(
                    context,
                    best=top[0],
                    divergent=divergent or top[0],
                    mode="crossover",
                )
                if config.c6_skill_memory:
                    prompt += rules_block(context.get("prior_hints") or [])
                deadband.mark_reseeded()
                text = await _gen(prompt, self.SEED_TEMPERATURE)
                for event in _text_events(text):
                    yield event
                code = codec.extract(text)
                plan_text = "re-seed from archive contrastive pair (plateau)"
            else:
                context_block = codec.base_prompt(context)
                if config.c6_skill_memory:
                    context_block += rules_block(context.get("prior_hints") or [])
                if config.c3_incumbent_context:
                    context_block += codec.incumbent_block(
                        incumbent, last_delta_pct=last_delta_pct
                    )
                if last_feedback is not None:
                    context_block += f"\n{last_feedback}\n"

                if config.c1_plan_then_implement:
                    plan_text = (
                        await _gen(
                            codec.plan_prompt(context_block),
                            self.PLAN_TEMPERATURE,
                        )
                    ).strip()
                    for event in _text_events(plan_text):
                        yield event
                    text = await _gen(
                        codec.implement_prompt(context_block, plan_text),
                        self.IMPLEMENT_TEMPERATURE,
                    )
                else:
                    plan_text = f"cascade serial turn {turn}"
                    text = await _gen(
                        context_block + codec.propose_instruction,
                        self.IMPLEMENT_TEMPERATURE,
                    )
                for event in _text_events(text):
                    yield event
                code = codec.extract(text)

            if code is None:
                last_feedback = codec.missing_candidate_feedback
                continue

            # C7: refuse rejected re-introductions without spending budget.
            normalized = codec.normalize(code)
            if config.c7_novelty_filter and normalized in rejected_normalized:
                novelty_hits += 1
                last_feedback = codec.novelty_feedback
                continue

            events, result, error = self._submit_candidate(
                toolset,
                context,
                code,
                # C1: the plan string becomes the Intervention rationale.
                description=plan_text[:400],
                call_prefix=f"cascade-t{turn}",
            )
            for event in events:
                yield event
            if result is None:
                rejected_normalized.add(normalized)
                last_feedback = codec.rejection_feedback(error or "rejected")
                if config.c8_deadband:
                    deadband.consider(None, incumbent_speedup)
                continue

            slots_remaining = result.get("slots_remaining", slots_remaining)
            archive.append(_entry(code, result))
            speedup = result.get("speedup_vs_baseline")
            successful = bool(result.get("successful"))
            if not successful:
                rejected_normalized.add(normalized)

            if config.c8_deadband:
                accepted = successful and deadband.consider(
                    speedup if isinstance(speedup, (int, float)) else None,
                    incumbent_speedup,
                )
            else:
                accepted = (
                    successful
                    and isinstance(speedup, (int, float))
                    and float(speedup) > incumbent_speedup
                )
            if successful:
                _record(result, accepted)
            last_delta_pct = (
                (float(speedup) / incumbent_speedup - 1.0) * 100.0
                if isinstance(speedup, (int, float)) and incumbent_speedup
                else None
            )
            if accepted:
                incumbent = _entry(code, result)
                incumbent_speedup = float(speedup)
                accepted_count += 1

            # C2: staged feedback for the next turn.
            if config.c2_staged_feedback:
                last_feedback = staged_feedback(
                    result,
                    incumbent_speedup=incumbent_speedup,
                    history=history,
                )
            else:
                last_feedback = codec.feedback_for_run_result(result)

        for event in self._reflect(toolset):
            yield event
        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text=(
                f"cascade finished: {llm_calls} LLM call(s), incumbent "
                f"speedup {incumbent_speedup:.3f}x, accepted "
                f"{accepted_count}, plateau {deadband.non_improvements}, "
                f"reseed_used={deadband.reseed_used}"
            ),
            extra={
                "usage": dict(usage_total),
                "llm_calls": llm_calls,
                "judge": judge_meta,
                "cascade_config": config.serialize(),
                "incumbent_speedup": incumbent_speedup,
                "accepted_count": accepted_count,
                "reseed_used": deadband.reseed_used,
                "early_stop": (
                    config.c8_deadband and deadband.should_stop
                ),
                "novelty_rejections": novelty_hits,
            },
        )

    def build_continuation_request(
        self,
        previous: HarnessRunRequest,
        snapshot: RunSnapshot,
    ) -> HarnessRunRequest:
        """Continuations skip the seed round (C4's split already spent the
        width budget) and refine serially with the remaining budget."""

        remaining = max(0, snapshot.max_candidates - snapshot.successful_count)
        extra = {
            **dict(previous.extra),
            "max_candidates": remaining,
            "cascade_continuation": True,
        }
        return dc_replace(
            previous,
            user_prompt=(
                f"CASCADE continuation: {remaining} validated slot(s) remain; "
                "serial refinement only."
            ),
            extra=extra,
        )
