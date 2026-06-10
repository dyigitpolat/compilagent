"""Candidate-representation codecs — the cross-space seam (ticket D9).

One codec per decision-space family bundles HOW a candidate is asked for
(prompt builders), parsed out of LLM text, normalized for the novelty
filter, and converted into canonical `propose_candidate` arguments. The
archetype protocols (SR chain, best-of-N fan-out, evolution archive, UCB1
bandit, CASCADE composite) call only this interface, so the identical
harness topology / feedback / memory / acceptance logic runs on:

  - SOURCE mode (`triton_source`): a candidate is a full kernel module
    (one `source_replace` intervention) — delegates to the existing P0-
    probe-mirroring builders in `prompts.py` / `evolution.py` /
    `cascade.py`, so source-space behavior is byte-identical to before.
  - LEVER mode (`torch_inductor` config knobs, `triton` MLIR pass
    pipeline, or any backend whose derived SearchSpace is NOT the single
    free-form `kernel_source` lever): a candidate is a JSON list of typed
    interventions ``{"target": {"kind", "selector"}, "payload",
    "rationale"}`` validated by `Backend.validate_intervention` — see
    `lever_prompts.py`.

Mode selection (`codec_for_context`) keys off the session's derived
SearchSpace as surfaced by `inspect_search_space`: lever mode iff levers
exist and none of them is a `source_replace` lever. An empty/absent
search space defaults to source mode (back-compat with minimal backends
and test fakes).
"""

from __future__ import annotations

from typing import Any, Protocol

from . import lever_prompts
from . import prompts as source_prompts


class CandidateCodec(Protocol):
    """Structural interface both codecs implement (documentation aid)."""

    name: str
    retry_format_message: str
    missing_candidate_feedback: str
    novelty_feedback: str
    propose_instruction: str
    strategy_arms: tuple[tuple[str, str], ...]

    def extract(self, text: str) -> str | None: ...
    def propose_args(self, candidate: str, *, description: str) -> dict[str, Any]: ...
    def normalize(self, candidate: str) -> str: ...
    def rejection_feedback(self, error: str) -> str: ...
    def feedback_for_run_result(self, result: dict[str, Any]) -> str: ...
    def base_prompt(self, context: dict[str, Any]) -> str: ...
    def menu_dropout_prompt(
        self, context: dict[str, Any], *, rng: Any, keep_probability: float = 0.5
    ) -> str: ...
    def contrastive_pair_prompt(
        self, context: dict[str, Any], *, best: Any, divergent: Any, mode: str
    ) -> str: ...
    def strategy_prompt(
        self,
        context: dict[str, Any],
        *,
        incumbent: str,
        incumbent_speedup: float,
        strategy_text: str,
    ) -> str: ...
    def incumbent_seed(self, context: dict[str, Any]) -> str: ...
    def incumbent_block(
        self, incumbent: Any, *, last_delta_pct: float | None
    ) -> str: ...
    def plan_prompt(self, context_block: str) -> str: ...
    def implement_prompt(self, context_block: str, plan: str) -> str: ...
    def judge_prompt(
        self, task_description: str, candidates: list[tuple[str, str]]
    ) -> str: ...


class SourceCodec:
    """Full-module candidates (the original E4-E6 behavior, unchanged)."""

    name = "source"
    retry_format_message = source_prompts.RETRY_FORMAT_MESSAGE
    missing_candidate_feedback = (
        "Your previous reply contained no python code block. "
        "Output exactly one fenced python code block."
    )
    novelty_feedback = (
        "Novelty filter: that module is identical (modulo "
        "comments/whitespace) to a candidate already REJECTED "
        "this run. Propose a structurally different approach."
    )
    propose_instruction = (
        "\nPropose the next improved module. Output exactly "
        "ONE fenced python code block with the full module, "
        "nothing else."
    )

    @property
    def strategy_arms(self) -> tuple[tuple[str, str], ...]:
        from .bandit import STRATEGY_ARMS

        return STRATEGY_ARMS

    def extract(self, text: str) -> str | None:
        return source_prompts.extract_code(text)

    def propose_args(self, candidate: str, *, description: str) -> dict[str, Any]:
        return {
            "interventions": [
                {
                    "target_kind": "source_replace",
                    "target_selector": "kernel_source",
                    "payload": {"module_source": candidate},
                    "rationale": description,
                }
            ],
            "description": description,
            "expected_effect": "",
        }

    def normalize(self, candidate: str) -> str:
        from .cascade import normalize_module_source

        return normalize_module_source(candidate)

    def rejection_feedback(self, error: str) -> str:
        return source_prompts.rejection_feedback(error)

    def feedback_for_run_result(self, result: dict[str, Any]) -> str:
        return source_prompts.feedback_for_run_result(result)

    def base_prompt(self, context: dict[str, Any]) -> str:
        return source_prompts.base_prompt(
            reference_source=context["reference_source"],
            task_description=context["task_description"],
            banned_patterns=context["banned_patterns"],
        )

    def menu_dropout_prompt(
        self, context: dict[str, Any], *, rng: Any, keep_probability: float = 0.5
    ) -> str:
        return source_prompts.menu_dropout_prompt(
            reference_source=context["reference_source"],
            task_description=context["task_description"],
            banned_patterns=context["banned_patterns"],
            rng=rng,
            keep_probability=keep_probability,
        )

    def contrastive_pair_prompt(
        self, context: dict[str, Any], *, best: Any, divergent: Any, mode: str
    ) -> str:
        from .evolution import contrastive_pair_prompt

        return contrastive_pair_prompt(
            reference_source=context["reference_source"],
            task_description=context["task_description"],
            banned_patterns=context["banned_patterns"],
            best=best,
            divergent=divergent,
            mode=mode,
        )

    def strategy_prompt(
        self,
        context: dict[str, Any],
        *,
        incumbent: str,
        incumbent_speedup: float,
        strategy_text: str,
    ) -> str:
        from .bandit import strategy_prompt

        return strategy_prompt(
            reference_source=context["reference_source"],
            task_description=context["task_description"],
            banned_patterns=context["banned_patterns"],
            incumbent_source=incumbent,
            incumbent_speedup=incumbent_speedup,
            strategy_text=strategy_text,
        )

    def incumbent_seed(self, context: dict[str, Any]) -> str:
        # The reference module itself (speedup 1.0 by definition).
        return str(context.get("reference_source") or "")

    def incumbent_block(
        self, incumbent: Any, *, last_delta_pct: float | None
    ) -> str:
        from .cascade import incumbent_block

        return incumbent_block(incumbent, last_delta_pct=last_delta_pct)

    def plan_prompt(self, context_block: str) -> str:
        from .cascade import plan_prompt

        return plan_prompt(context_block)

    def implement_prompt(self, context_block: str, plan: str) -> str:
        from .cascade import implement_prompt

        return implement_prompt(context_block, plan)

    def judge_prompt(
        self, task_description: str, candidates: list[tuple[str, str]]
    ) -> str:
        from .cascade import judge_prompt

        return judge_prompt(task_description, candidates)


class LeverCodec:
    """Typed-intervention candidates over the derived SearchSpace."""

    name = "lever"
    retry_format_message = lever_prompts.RETRY_FORMAT_MESSAGE
    missing_candidate_feedback = lever_prompts.MISSING_CANDIDATE_FEEDBACK
    novelty_feedback = lever_prompts.NOVELTY_FEEDBACK
    propose_instruction = lever_prompts.PROPOSE_INSTRUCTION
    strategy_arms = lever_prompts.STRATEGY_ARMS

    def extract(self, text: str) -> str | None:
        return lever_prompts.extract_interventions(text)

    def propose_args(self, candidate: str, *, description: str) -> dict[str, Any]:
        return lever_prompts.propose_args(candidate, description=description)

    def normalize(self, candidate: str) -> str:
        return lever_prompts.normalize_interventions(candidate)

    def rejection_feedback(self, error: str) -> str:
        return lever_prompts.rejection_feedback(error)

    def feedback_for_run_result(self, result: dict[str, Any]) -> str:
        return lever_prompts.feedback_for_run_result(result)

    def base_prompt(self, context: dict[str, Any]) -> str:
        return lever_prompts.base_prompt(context)

    def menu_dropout_prompt(
        self, context: dict[str, Any], *, rng: Any, keep_probability: float = 0.5
    ) -> str:
        return lever_prompts.menu_dropout_prompt(
            context, rng=rng, keep_probability=keep_probability
        )

    def contrastive_pair_prompt(
        self, context: dict[str, Any], *, best: Any, divergent: Any, mode: str
    ) -> str:
        return lever_prompts.contrastive_pair_prompt(
            context, best=best, divergent=divergent, mode=mode
        )

    def strategy_prompt(
        self,
        context: dict[str, Any],
        *,
        incumbent: str,
        incumbent_speedup: float,
        strategy_text: str,
    ) -> str:
        return lever_prompts.strategy_prompt(
            context,
            incumbent=incumbent,
            incumbent_speedup=incumbent_speedup,
            strategy_text=strategy_text,
        )

    def incumbent_seed(self, context: dict[str, Any]) -> str:
        # The empty plan IS the baseline (Plan() is the identity element).
        return "[]"

    def incumbent_block(
        self, incumbent: Any, *, last_delta_pct: float | None
    ) -> str:
        return lever_prompts.incumbent_block(
            incumbent, last_delta_pct=last_delta_pct
        )

    def plan_prompt(self, context_block: str) -> str:
        return lever_prompts.plan_prompt(context_block)

    def implement_prompt(self, context_block: str, plan: str) -> str:
        return lever_prompts.implement_prompt(context_block, plan)

    def judge_prompt(
        self, task_description: str, candidates: list[tuple[str, str]]
    ) -> str:
        return lever_prompts.judge_prompt(task_description, candidates)


SOURCE_CODEC = SourceCodec()
LEVER_CODEC = LeverCodec()


def codec_for_context(context: dict[str, Any]) -> SourceCodec | LeverCodec:
    """Pick the candidate representation from the derived SearchSpace.

    LEVER mode iff the space advertises levers and none of them is the
    free-form `source_replace` kernel-source lever; everything else
    (including an empty space) stays in SOURCE mode.
    """

    levers = list(context.get("levers") or [])
    if levers and all(
        (lever.get("target") or {}).get("kind") != "source_replace"
        for lever in levers
    ):
        return LEVER_CODEC
    return SOURCE_CODEC
