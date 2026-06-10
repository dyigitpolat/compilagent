"""`archetype_ma` — H-MA, profiler-in-the-loop Coder+Judge (CudaForge-inspired).

Implements the CudaForge serial two-role refinement protocol (Zhang et al.,
arXiv:2511.01884; card B3 in research_artifacts/02_baseline_cards.md) on
top of the compilagent session contract. Each round:

  1. the Coder emits a full Triton module — round 1 from the SAME base
     prompt + vector-add exemplar as `archetype_sr` (reused from
     `prompts.py`); later rounds from a role-specific correction or
     optimization prompt;
  2. the module goes through the canonical `propose_candidate` /
     `run_candidate` tool protocol (gates run before any timing rep — the
     budget-ledger invariant is asserted on every result);
  3. a GATE-PASSING candidate is NCU-profiled via
     `triton_source._internal.ncu_profile` (curated ~12-metric subset; the
     profile runs under the same GPU lease pool as the sandbox, replays
     charged to the lease);
  4. the Judge (same model, temperature 0.2) receives the candidate source
     + measured timing + the curated metrics block and returns a strict
     one-issue JSON analysis — optimization mode: {bottleneck,
     optimization_method, modification_plan}; correction mode:
     {critical_issue, why_it_matters, minimal_fix_hint};
  5. the next Coder turn is Markovian (CudaForge's "lightweight memory"):
     it sees only the task context, the INCUMBENT (best validated) kernel
     + its timing, and the latest Judge analysis — never the conversation
     history. Correction turns see the failed kernel + error log instead.

The round count N is bounded by the session's `max_candidates` budget (the
loop stops when `slots_remaining == 0`); keep-best selection over all
rounds is the session leaderboard, matching CudaForge's
best-of-history-archive final answer.

Documented deviations from the CudaForge paper:

  - Action space is Triton source (`ModelNew` modules), NOT CUDA-C
    `load_inline` extensions; the base prompt/exemplar is shared with
    `archetype_sr` for cross-archetype comparability.
  - Verification is this repo's hardened gate set (g1 shape/dtype, g2
    allclose, g3 anti-aliasing, g4 banned-API lint, g5 determinism) plus
    the budget ledger — not compile + 1e-4-vs-reference alone — and N
    comes from the session budget, not a fixed 10/30.
  - ONE model instantiates both roles (CudaForge treats Coder/Judge as
    separable agents, though it also defaults both to o3).
  - The curated NCU subset is ~12 a-priori robust metrics (SM/DRAM
    throughput %, achieved occupancy, L1/L2 hit rates, gld/gst efficiency,
    top warp-stall reasons, registers/thread, duration), not the offline
    Pearson-correlation-derived 24-metric whitelist of their Appendix B.3.
  - NCU may be UNAVAILABLE on this machine (profiling is admin-gated via
    `RmProfilingAdminOnly=1`; the sudoers grant for `sudo -n ncu` may be
    pending): the runner degrades to `metrics={}` with an
    `ncu_unavailable` flag and the loop continues as a no-NCU H-MA — the
    Judge is told profiling is unavailable and analyzes from source +
    timing alone, mirroring CudaForge's own no-NCU ablation. The flag is
    recorded per round and overall in the run metadata (under the `judge`
    key the pilot driver already copies into suite rows).
  - Judge JSON is parsed leniently (first `{...}` object; raw-text
    fallback) instead of hard-failing on malformed output.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from compilagent.harness.base import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
)
from compilagent.session.completion import RunSnapshot

from .harness import _DEFAULT_MAX_TURNS, _ArchetypeHarnessBase
from .prompts import RETRY_FORMAT_MESSAGE, base_prompt, extract_code

#: Test seam: (reference_source, candidate_source) → ncu_profile result dict.
ProfileFn = Callable[..., dict[str, Any]]

OPTIMIZATION_KEYS: tuple[str, ...] = (
    "bottleneck",
    "optimization_method",
    "modification_plan",
)
CORRECTION_KEYS: tuple[str, ...] = (
    "critical_issue",
    "why_it_matters",
    "minimal_fix_hint",
)

_NO_PROFILE_NOTE = (
    "NCU profiling is UNAVAILABLE on this machine for this run — no "
    "hardware counters are attached. Analyze from the kernel source and "
    "the measured timing alone."
)


def _default_profile(
    *, reference_source: str, candidate_source: str
) -> dict[str, Any]:
    # Deliberate cross-integration reuse: profiling is a backend-side
    # capability of triton_source, exactly like the eval sandbox.
    from compilagent.integrations.triton_source._internal.ncu_profile import (
        profile_candidate,
    )

    artifact_dir = Path(tempfile.mkdtemp(prefix="compilagent-ncu-"))
    return profile_candidate(
        reference_source=reference_source,
        candidate_source=candidate_source,
        artifact_dir=artifact_dir,
    )


# ----------------------------------------------------------------- prompts


def timing_line(result: dict[str, Any]) -> str:
    cand_ms = result.get("median_ms")
    speedup = result.get("speedup_vs_baseline")
    if isinstance(cand_ms, (int, float)) and isinstance(speedup, (int, float)):
        return (
            f"measured {cand_ms:.4f} ms vs eager {cand_ms * speedup:.4f} ms "
            f"(speedup {speedup:.3f}x)"
        )
    return "no timing signal recorded"


def error_log_for(result: dict[str, Any]) -> str:
    """ERROR_LOG analogue: compile diagnostics, or the failing gate(s)."""

    if not result.get("compile_ok"):
        return str(result.get("compile_diagnostics") or "unknown failure")
    gate_lines = [
        w for w in (result.get("compile_warnings") or []) if "FAILED" in str(w)
    ]
    return " ".join(gate_lines) or (
        f"correctness gate failed: max_abs_diff={result.get('max_abs_diff')}"
    )


def judge_optimization_prompt(
    *,
    task_description: str,
    candidate_source: str,
    timing: str,
    metrics_block: str,
) -> str:
    """Judge, optimization mode — one bottleneck, strict JSON."""

    profile_part = metrics_block or _NO_PROFILE_NOTE
    return f"""You are a GPU-kernel optimization specialist acting as the JUDGE in a
Coder/Judge loop. You never write code; you diagnose exactly ONE hardware
bottleneck per round.

Task: {task_description}

The Triton candidate below PASSED all correctness gates; {timing}.

Candidate module:
```python
{candidate_source}
```

{profile_part}

Identify the single highest-impact bottleneck (e.g. memory-bound,
occupancy-limited, stall-limited, launch-overhead, uncoalesced access),
citing the 3-4 most important metrics when they are available. Respond with
ONLY a JSON object, no other text:
{{"bottleneck": "<=30 words", "optimization_method": "<=35 words", "modification_plan": "<=35 words"}}"""


def judge_correction_prompt(
    *,
    task_description: str,
    candidate_source: str,
    error_log: str,
) -> str:
    """Judge, correction mode — one critical issue, strict JSON."""

    return f"""You are a senior Triton/PyTorch extension developer acting as the JUDGE
in a Coder/Judge loop. You never write code; you diagnose exactly ONE
critical issue per round.

Task: {task_description}

The Triton candidate below FAILED.

Candidate module:
```python
{candidate_source}
```

ERROR LOG:
{error_log}

Identify the single highest-impact issue causing the failure. Respond with
ONLY a JSON object, no other text:
{{"critical_issue": "<=20 words", "why_it_matters": "<=35 words", "minimal_fix_hint": "<=20 words"}}"""


def coder_optimization_prompt(
    *,
    base: str,
    incumbent_source: str,
    incumbent_timing: str,
    analysis: str,
) -> str:
    """Coder, optimization mode: incumbent + Judge strategy, no history."""

    return f"""{base}
Current incumbent (best validated candidate so far) — {incumbent_timing}:
```python
{incumbent_source}
```

Judge analysis of the latest profiled run:
{analysis}

Strictly apply the judge's STRATEGY above to produce the next, faster
module (same public behavior: identical output dtype/shape, same parameter
names). Output exactly ONE fenced python code block with the full module,
nothing else."""


def coder_correction_prompt(
    *,
    base: str,
    failed_source: str,
    error_log: str,
    analysis: str,
) -> str:
    """Coder, correction mode: failed kernel + error log + Judge fix hint."""

    return f"""{base}
Your previous candidate FAILED:
```python
{failed_source}
```

ERROR LOG:
{error_log}

Judge diagnosis (fix exactly this one issue):
{analysis}

Produce the full corrected module. Output exactly ONE fenced python code
block with the full module, nothing else."""


def parse_judge_analysis(
    text: str, expected_keys: tuple[str, ...]
) -> dict[str, str]:
    """First `{...}` JSON object, filtered to the expected keys (British
    `optimisation_*` spellings normalized); raw-text fallback on any
    malformation — a documented deviation from CudaForge's strict JSON."""

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            normalized = {
                str(k).strip().lower().replace("optimisation", "optimization"): v
                for k, v in parsed.items()
            }
            out = {
                key: str(normalized[key])
                for key in expected_keys
                if key in normalized
            }
            if out:
                return out
    return {"analysis": text.strip()[:600]}


def format_analysis(analysis: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in analysis.items())


# ----------------------------------------------------------------- harness


class ArchetypeProfilerMAHarness(_ArchetypeHarnessBase):
    """`archetype_ma` — serial Coder+Judge with NCU profiler in the loop."""

    id: str = "archetype_ma"

    #: First generation mirrors archetype_sr; refinements run cooler; the
    #: Judge runs at the cascade judge temperature.
    CODER_FIRST_TEMPERATURE: float = 0.7
    CODER_TEMPERATURE: float = 0.5
    JUDGE_TEMPERATURE: float = 0.2

    def __init__(
        self,
        generate_fn: Any | None = None,
        profile_fn: ProfileFn | None = None,
    ) -> None:
        super().__init__(generate_fn)
        # Test seam: inject a fake profiler; production runs ncu_profile.
        self._profile_override = profile_fn

    def _profiler(self) -> ProfileFn:
        return self._profile_override or _default_profile

    async def _run_protocol(
        self, request: HarnessRunRequest
    ) -> AsyncIterator[StreamEvent]:
        toolset = request.toolset
        generate = self._generator(request)
        profile = self._profiler()
        usage_total: dict[str, int] = {}
        llm_calls = 0
        part = 0

        ctx_events, context = self._load_task_context(toolset)
        for event in ctx_events:
            yield event

        base = base_prompt(
            reference_source=context["reference_source"],
            task_description=context["task_description"],
            banned_patterns=context["banned_patterns"],
        )
        task = context["task_description"]
        max_turns = request.max_turns or _DEFAULT_MAX_TURNS

        incumbent_code: str | None = None
        incumbent_result: dict[str, Any] | None = None
        incumbent_speedup: float = 1.0  # the reference, by definition
        rounds_meta: list[dict[str, Any]] = []
        profiled_rounds = 0
        any_unavailable = False
        slots_remaining: int | None = None
        coder_prompt = base
        retry_appended = False

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

        for round_idx in range(max_turns):
            text = await _gen(
                coder_prompt,
                self.CODER_FIRST_TEMPERATURE
                if round_idx == 0
                else self.CODER_TEMPERATURE,
            )
            for event in _text_events(text):
                yield event

            code = extract_code(text)
            if code is None:
                if not retry_appended:
                    coder_prompt += f"\n\n{RETRY_FORMAT_MESSAGE}"
                    retry_appended = True
                continue
            retry_appended = False

            events, result, error = self._submit_candidate(
                toolset,
                context,
                code,
                description=f"archetype_ma round {round_idx}",
                call_prefix=f"ma-r{round_idx}",
            )
            for event in events:
                yield event

            if result is None:
                # propose/run rejection (lint, missing ModelNew, ...): the
                # rejection text is the ERROR_LOG of a correction round.
                error_log = error or "candidate rejected before reaching the GPU"
                judge_text = await _gen(
                    judge_correction_prompt(
                        task_description=task,
                        candidate_source=code,
                        error_log=error_log,
                    ),
                    self.JUDGE_TEMPERATURE,
                )
                for event in _text_events(judge_text):
                    yield event
                analysis = parse_judge_analysis(judge_text, CORRECTION_KEYS)
                rounds_meta.append(
                    {
                        "round": round_idx,
                        "mode": "correction",
                        "candidate_id": None,
                        "speedup": None,
                        "analysis": analysis,
                    }
                )
                coder_prompt = coder_correction_prompt(
                    base=base,
                    failed_source=code,
                    error_log=error_log,
                    analysis=format_analysis(analysis),
                )
                continue

            slots_remaining = result.get("slots_remaining", slots_remaining)
            successful = bool(result.get("successful"))
            speedup = result.get("speedup_vs_baseline")
            round_meta: dict[str, Any] = {
                "round": round_idx,
                "candidate_id": result.get("candidate_id"),
                "speedup": speedup,
            }
            if successful and (
                incumbent_code is None
                or (
                    isinstance(speedup, (int, float))
                    and float(speedup) > incumbent_speedup
                )
            ):
                incumbent_code = code
                incumbent_result = result
                if isinstance(speedup, (int, float)):
                    incumbent_speedup = float(speedup)

            if slots_remaining == 0:
                # Budget exhausted: no Coder turn will consume a Judge
                # analysis, so neither the profile nor the Judge fires.
                rounds_meta.append({**round_meta, "mode": "final"})
                break

            if successful:
                ncu = profile(
                    reference_source=context["reference_source"],
                    candidate_source=code,
                )
                profiled_rounds += 1
                unavailable = bool(ncu.get("ncu_unavailable"))
                any_unavailable = any_unavailable or unavailable
                # Local import keeps triton_source out of this module's
                # import-time dependencies (test environments without it).
                from compilagent.integrations.triton_source._internal.ncu_profile import (  # noqa: E501
                    format_metrics_block,
                )

                judge_text = await _gen(
                    judge_optimization_prompt(
                        task_description=task,
                        candidate_source=code,
                        timing=timing_line(result),
                        metrics_block=format_metrics_block(ncu),
                    ),
                    self.JUDGE_TEMPERATURE,
                )
                for event in _text_events(judge_text):
                    yield event
                analysis = parse_judge_analysis(judge_text, OPTIMIZATION_KEYS)
                rounds_meta.append(
                    {
                        **round_meta,
                        "mode": "optimization",
                        "ncu_unavailable": unavailable,
                        "ncu_reason": ncu.get("reason"),
                        "ncu_invocation": ncu.get("invocation"),
                        "dominant_kernel": ncu.get("dominant_kernel"),
                        "metrics_count": len(ncu.get("metrics") or {}),
                        "analysis": analysis,
                    }
                )
                coder_prompt = coder_optimization_prompt(
                    base=base,
                    incumbent_source=incumbent_code or code,
                    incumbent_timing=timing_line(incumbent_result or result),
                    analysis=format_analysis(analysis),
                )
            else:
                error_log = error_log_for(result)
                judge_text = await _gen(
                    judge_correction_prompt(
                        task_description=task,
                        candidate_source=code,
                        error_log=error_log,
                    ),
                    self.JUDGE_TEMPERATURE,
                )
                for event in _text_events(judge_text):
                    yield event
                analysis = parse_judge_analysis(judge_text, CORRECTION_KEYS)
                rounds_meta.append(
                    {**round_meta, "mode": "correction", "analysis": analysis}
                )
                coder_prompt = coder_correction_prompt(
                    base=base,
                    failed_source=code,
                    error_log=error_log,
                    analysis=format_analysis(analysis),
                )

        for event in self._reflect(toolset):
            yield event
        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text=(
                f"archetype_ma finished: {llm_calls} LLM call(s), "
                f"{len(rounds_meta)} round(s), {profiled_rounds} profiled, "
                f"ncu_unavailable="
                f"{any_unavailable if profiled_rounds else None}, "
                f"incumbent speedup {incumbent_speedup:.3f}x"
            ),
            extra={
                "usage": dict(usage_total),
                "llm_calls": llm_calls,
                # Under the `judge` key so run_pilot's existing metadata
                # pass-through records the NCU state in every suite row.
                "judge": {
                    "rounds": rounds_meta,
                    "profiled_rounds": profiled_rounds,
                    # None when nothing was profiled (no gate-passer): the
                    # NCU path never ran, so availability is unknown.
                    "ncu_unavailable": (
                        any_unavailable if profiled_rounds else None
                    ),
                },
                "incumbent_speedup": incumbent_speedup,
            },
        )

    def build_continuation_request(
        self,
        previous: HarnessRunRequest,
        snapshot: RunSnapshot,
    ) -> HarnessRunRequest:
        """Continuations restart the Markovian chain (no transcript to
        resume by construction); the session leaderboard keeps everything
        learned so far."""

        remaining = max(0, snapshot.max_candidates - snapshot.successful_count)
        return replace(
            previous,
            user_prompt=(
                f"Continue the Coder/Judge loop: {remaining} validated "
                "slot(s) remain."
            ),
        )
