"""Unit tests for the E5 archetypes: `archetype_evo` + `archetype_band`.

No LLM, no GPU — a scripted generator and a fake session toolset drive the
protocols, mirroring `test_archetype_harnesses.py`. Covers the UCB1
arm-selection math, archive parent selection (top-K + divergent), and the
two protocol loops end-to-end through the canonical event stream.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest

from compilagent.core.tool_decl import ToolDecl
from compilagent.harness.base import HarnessRunRequest, StreamEventKind
from compilagent.harness.registry import harness_registry
from compilagent.integrations.archetype_harnesses.bandit import (
    STRATEGY_ARMS,
    ArchetypeBanditHarness,
    ucb1_select,
)
from compilagent.integrations.archetype_harnesses.evolution import (
    ArchetypeEvolutionHarness,
    ArchiveEntry,
    select_parents,
    token_set_distance,
)
from compilagent.toolset import Toolset

# ----------------------------------------------------------- shared fakes


def _decl(name: str, handler) -> ToolDecl:
    return ToolDecl(
        name=name,
        description=name,
        args_schema={"type": "object", "properties": {}},
        handler=handler,
        read_only=False,
    )


class _FakeSession:
    """Scripted tool surface mimicking the session contract."""

    def __init__(self, run_results: list[dict[str, Any]]):
        self.run_results = list(run_results)
        self.proposed: list[dict[str, Any]] = []
        self.ran: list[str] = []
        self.reflections: list[str] = []

    def toolset(self) -> Toolset:
        def inspect_workload(**_kw) -> str:
            return json.dumps(
                {
                    "workload": {
                        "id": "softmax_4096",
                        "description": "Row-wise softmax.",
                        "metadata": {},
                    },
                    "backend_id": "triton_source",
                    "baseline_timing": {"median_ms": 1.0},
                }
            )

        def read_reference_source(**_kw) -> str:
            return json.dumps(
                {
                    "workload_id": "softmax_4096",
                    "task_description": "Row-wise softmax.",
                    "reference_module_source": "class Model: ...",
                    "banned_patterns": ["softmax"],
                }
            )

        def propose_candidate(**kwargs) -> str:
            self.proposed.append(kwargs)
            return json.dumps({"id": f"cand-{len(self.proposed)}"})

        def run_candidate(**kwargs) -> str:
            self.ran.append(str(kwargs.get("candidate_id")))
            return json.dumps(self.run_results.pop(0))

        def compare_runs(**_kw) -> str:
            self.reflections.append("compare_runs")
            return "[]"

        def synthesize_findings(**_kw) -> str:
            self.reflections.append("synthesize_findings")
            return "{}"

        return Toolset(
            tools=tuple(
                _decl(name, fn)
                for name, fn in [
                    ("inspect_workload", inspect_workload),
                    ("read_reference_source", read_reference_source),
                    ("propose_candidate", propose_candidate),
                    ("run_candidate", run_candidate),
                    ("compare_runs", compare_runs),
                    ("synthesize_findings", synthesize_findings),
                ]
            )
        )


def _scripted_generator(texts: list[str], record: list[dict[str, Any]]):
    queue = list(texts)

    async def generate(*, history, system=None, temperature=0.7, max_tokens=None):
        record.append({"history": list(history), "temperature": temperature})
        return queue.pop(0), {
            "request_tokens": 100,
            "response_tokens": 50,
            "total_tokens": 150,
        }

    return generate


def _module(code: str) -> str:
    return f"```python\n{code}\n```"


def _request(toolset: Toolset, **extra) -> HarnessRunRequest:
    return HarnessRunRequest(
        toolset=toolset,
        system_instructions="be brief",
        user_prompt="optimise",
        model_id="mistral:mistral-large-latest",
        extra=extra,
    )


def _drive(harness, request):
    async def _collect():
        return [ev async for ev in harness.run(request)]

    return asyncio.run(_collect())


def _result(*, compile_ok=True, correctness_ok=True, median_ms=0.5, speedup=2.0,
            slots_remaining=0, successful=None, candidate_id="cand-x") -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "compile_ok": compile_ok,
        "median_ms": median_ms,
        "speedup_vs_baseline": speedup,
        "correctness_ok": correctness_ok,
        "compile_diagnostics": None if compile_ok else "NameError: tl",
        "compile_warnings": [],
        "successful": (
            bool(compile_ok and correctness_ok) if successful is None else successful
        ),
        "slots_remaining": slots_remaining,
        "max_abs_diff": 0.0,
    }


# ----------------------------------------------------------------- registry


def test_registration_installs_evo_and_band():
    import compilagent.integrations.archetype_harnesses  # noqa: F401

    assert "archetype_evo" in harness_registry.ids()
    assert "archetype_band" in harness_registry.ids()


# --------------------------------------------------------------- UCB1 math


def test_ucb1_pulls_every_arm_once_first_in_order():
    assert ucb1_select(pulls=[0, 0, 0], total_rewards=[0, 0, 0]) == 0
    assert ucb1_select(pulls=[1, 0, 0], total_rewards=[0.5, 0, 0]) == 1
    assert ucb1_select(pulls=[1, 1, 0], total_rewards=[0.5, 0.1, 0]) == 2


def test_ucb1_exploits_higher_mean_at_equal_pulls():
    # Equal pulls → equal bonus → pure mean comparison.
    assert ucb1_select(pulls=[2, 2], total_rewards=[1.0, 0.5]) == 0
    assert ucb1_select(pulls=[3, 3, 3], total_rewards=[0.1, 0.9, 0.5]) == 1


def test_ucb1_exploration_bonus_promotes_underpulled_arm():
    # Hand-computed: pulls=[10,2], rewards=[5.0,0.9], N=12, c=sqrt(2):
    #   score0 = 0.50 + sqrt(2*ln12/10) = 0.50 + 0.7050... ≈ 1.205
    #   score1 = 0.45 + sqrt(2*ln12/2)  = 0.45 + 1.5764... ≈ 2.026
    assert ucb1_select(pulls=[10, 2], total_rewards=[5.0, 0.9]) == 1
    # Sanity against the closed form:
    c = math.sqrt(2.0)
    s0 = 5.0 / 10 + c * math.sqrt(math.log(12) / 10)
    s1 = 0.9 / 2 + c * math.sqrt(math.log(12) / 2)
    assert s1 > s0


def test_ucb1_zero_exploration_is_pure_greedy():
    assert ucb1_select(
        pulls=[10, 2], total_rewards=[5.0, 0.9], exploration=0.0
    ) == 0


def test_ucb1_rejects_malformed_inputs():
    with pytest.raises(ValueError):
        ucb1_select(pulls=[], total_rewards=[])
    with pytest.raises(ValueError):
        ucb1_select(pulls=[1, 2], total_rewards=[0.5])


# ------------------------------------------------- archive parent selection


def _entry(cid: str, source: str, speedup, correctness_ok=True) -> ArchiveEntry:
    return ArchiveEntry(
        candidate_id=cid,
        source=source,
        speedup=speedup,
        correctness_ok=correctness_ok,
        summary=f"speedup {speedup}",
    )


def test_select_parents_orders_top_k_by_validated_speedup():
    entries = [
        _entry("a", "alpha beta gamma", 1.2),
        _entry("b", "alpha beta gamma delta", 2.0),
        _entry("c", "alpha beta", 1.5),
        _entry("d", "alpha beta gamma", None),  # never timed — not validated
        _entry("e", "alpha beta gamma", 3.0, correctness_ok=False),  # gate-fail
    ]
    top, _ = select_parents(entries, top_k=2)
    assert [e.candidate_id for e in top] == ["b", "c"]


def test_select_parents_divergent_is_farthest_validated_from_best():
    best = _entry("best", "import triton\nrow_max = tl.max(x)", 2.0)
    near = _entry("near", "import triton\nrow_max = tl.max(x) + 0", 1.5)
    far = _entry("far", "totally different identifiers everywhere", 1.2)
    top, divergent = select_parents([near, best, far], top_k=4)
    assert top[0] is best
    assert divergent is far
    # The metric itself is sane.
    assert token_set_distance(best.source, far.source) > token_set_distance(
        best.source, near.source
    )


def test_select_parents_divergent_falls_back_to_failures():
    best = _entry("best", "alpha beta", 2.0)
    failed = _entry("failed", "epsilon zeta", None, correctness_ok=False)
    top, divergent = select_parents([best, failed], top_k=4)
    assert [e.candidate_id for e in top] == ["best"]
    assert divergent is failed  # only contrast available


def test_select_parents_empty_until_something_validates():
    failed = _entry("failed", "epsilon", None, correctness_ok=False)
    assert select_parents([failed], top_k=4) == ([], None)


# -------------------------------------------------------------- archetype_evo


def test_evo_seeds_then_breeds_contrastive_children():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.2, slots_remaining=1, candidate_id="cand-1"),
            _result(correctness_ok=False, median_ms=None, speedup=None,
                    slots_remaining=1, candidate_id="cand-2"),
            _result(speedup=1.5, slots_remaining=0, candidate_id="cand-3"),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeEvolutionHarness(
        generate_fn=_scripted_generator(
            [
                _module("seed_one = 'vectorized rows'"),
                _module("seed_two = 'online softmax accumulator'"),
                _module("child_one = 'crossover'"),
            ],
            record,
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=2, population_size=2, seed=7),
    )

    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    # 2 seeds + 1 crossover child (mutation skipped: budget hit 0).
    assert len(record) == 3
    assert len(session.proposed) == 3
    # Seed round: temp 1.0 single-turn menu-dropout prompts.
    assert record[0]["temperature"] == 1.0
    assert record[1]["temperature"] == 1.0
    assert "Focus especially" in record[0]["history"][0][1]
    # Child prompt: contrastive PAIR with measured results. The divergent
    # slot falls back to the gate-failing seed (only other entry).
    child_prompt = record[2]["history"][0][1]
    assert record[2]["temperature"] == ArchetypeEvolutionHarness.CHILD_TEMPERATURE
    assert "Parent A (best overall)" in child_prompt
    assert "speedup 1.200x" in child_prompt
    assert "seed_one = 'vectorized rows'" in child_prompt
    assert "Parent B (divergent)" in child_prompt
    assert "seed_two = 'online softmax accumulator'" in child_prompt
    assert "failed correctness gate" in child_prompt
    assert "CROSSOVER" in child_prompt
    # Reflection lock + accounting.
    assert session.reflections == ["compare_runs", "synthesize_findings"]
    assert events[-1].extra["llm_calls"] == 3
    assert events[-1].extra["usage"]["total_tokens"] == 450
    assert events[-1].extra["archive_size"] == 3


def test_evo_reseeds_when_nothing_validated_yet():
    session = _FakeSession(
        run_results=[
            _result(compile_ok=False, correctness_ok=None, median_ms=None,
                    speedup=None, slots_remaining=1, candidate_id="cand-1"),
            _result(speedup=1.3, slots_remaining=0, candidate_id="cand-2"),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeEvolutionHarness(
        generate_fn=_scripted_generator(
            [_module("broken = 1"), _module("fixed = 2")], record
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=1, population_size=1, seed=3),
    )
    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    assert len(record) == 2
    # The fallback sample is a fresh menu-dropout seed, not a breeding prompt.
    assert record[1]["temperature"] == 1.0
    assert "Focus especially" in record[1]["history"][0][1]
    assert "Parent A" not in record[1]["history"][0][1]


def test_evo_continuation_resizes_population_to_remaining_budget():
    from compilagent.session.completion import RunSnapshot

    harness = ArchetypeEvolutionHarness(generate_fn=_scripted_generator([], []))
    previous = _request(Toolset(tools=()), max_candidates=8, population_size=4)
    snap = RunSnapshot(
        successful_count=6,
        failed_attempts=2,
        max_candidates=8,
        max_failed_attempts=24,
        tools_called=frozenset(),
        iteration=0,
        max_continuations=4,
        harness_failed=False,
        best_speedup=1.4,
    )
    nxt = harness.build_continuation_request(previous, snap)
    assert nxt.extra["max_candidates"] == 2
    assert nxt.extra["population_size"] == 2


# ------------------------------------------------------------- archetype_band


def test_band_pulls_arms_by_ucb1_and_tracks_incumbent():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.2, slots_remaining=1, candidate_id="cand-1"),
            _result(correctness_ok=False, median_ms=None, speedup=None,
                    slots_remaining=1, candidate_id="cand-2"),
            _result(speedup=1.5, slots_remaining=0, candidate_id="cand-3"),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeBanditHarness(
        generate_fn=_scripted_generator(
            [
                _module("pull_one = 'wide loads'"),
                _module("pull_two = 'tiles'"),
                _module("pull_three = 'fused'"),
            ],
            record,
        )
    )
    events = _drive(harness, _request(session.toolset(), max_candidates=2))

    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    assert len(record) == 3
    # Init phase pulls arms in declaration order.
    assert "VECTORIZE" in record[0]["history"][0][1]
    assert "TILE-RESIZE" in record[1]["history"][0][1]
    assert "FUSE-LOOPS" in record[2]["history"][0][1]
    # Pull 1 conditions on the reference incumbent (speedup 1.0)…
    assert "speedup 1.000x" in record[0]["history"][0][1]
    # …pull 2 conditions on the validated winner of pull 1.
    assert "pull_one = 'wide loads'" in record[1]["history"][0][1]
    assert "speedup 1.200x" in record[1]["history"][0][1]
    # Rewards are validated speedup deltas vs the incumbent at pull time.
    arms = events[-1].extra["arms"]
    assert arms["vectorize"]["pulls"] == 1
    assert arms["vectorize"]["mean_reward"] == pytest.approx(0.2)
    assert arms["tile_resize"]["pulls"] == 1
    assert arms["tile_resize"]["mean_reward"] == 0.0
    assert arms["fuse_loops"]["pulls"] == 1
    assert arms["fuse_loops"]["mean_reward"] == pytest.approx(0.3)
    assert arms["reduce_passes"]["pulls"] == 0
    assert arms["relayout"]["pulls"] == 0
    assert events[-1].extra["incumbent_speedup"] == pytest.approx(1.5)
    assert session.reflections == ["compare_runs", "synthesize_findings"]
    assert events[-1].extra["llm_calls"] == 3


def test_band_stops_at_max_turns_without_budget_signal():
    """Unparseable outputs never call run_candidate; the pull cap bounds
    the loop."""

    session = _FakeSession(run_results=[])
    record: list[dict[str, Any]] = []
    harness = ArchetypeBanditHarness(
        generate_fn=_scripted_generator(["no code here"] * 3, record)
    )
    request = HarnessRunRequest(
        toolset=session.toolset(),
        system_instructions="",
        user_prompt="go",
        model_id="mistral:mistral-large-latest",
        max_turns=3,
        extra={"max_candidates": 2},
    )
    events = _drive(harness, request)
    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    assert len(record) == 3
    assert session.ran == []
    # Failed pulls still count as pulls with zero reward.
    arms = events[-1].extra["arms"]
    assert sum(v["pulls"] for v in arms.values()) == 3


def test_band_has_five_named_strategy_arms():
    names = [name for name, _ in STRATEGY_ARMS]
    assert names == [
        "vectorize", "tile_resize", "fuse_loops", "reduce_passes", "relayout",
    ]
    assert all(template.strip() for _, template in STRATEGY_ARMS)
