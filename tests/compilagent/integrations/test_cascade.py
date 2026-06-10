"""Unit tests for the CASCADE v0 composite harness (ticket E6).

No LLM, no GPU — scripted generators and a fake session toolset, mirroring
the other archetype tests. Covers the novelty-filter normalization, the
keep/revert deadband (accept >1%, plateau re-seed, early finalize), the
judge-rank metadata, and — critically for the T2 ablation — that every
ingredient toggle actually disables its code path.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from compilagent.core.tool_decl import ToolDecl
from compilagent.harness.base import HarnessRunRequest, StreamEventKind
from compilagent.harness.registry import harness_registry
from compilagent.integrations.archetype_harnesses.cascade import (
    AXIS_BUNDLES,
    CascadeConfig,
    CascadeHarness,
    DeadbandController,
    normalize_module_source,
    parse_judge_ranking,
)
from compilagent.toolset import Toolset

# ----------------------------------------------------------- shared fakes

_HINTS = [
    "constraint rule: tl.sum has NO mask keyword — mask at load time.",
    "constraint rule: GEMM kernels need explicit OOB masking on the K-tail.",
]


def _decl(name: str, handler) -> ToolDecl:
    return ToolDecl(
        name=name,
        description=name,
        args_schema={"type": "object", "properties": {}},
        handler=handler,
        read_only=False,
    )


class _FakeSession:
    def __init__(self, run_results: list[dict[str, Any]], prior_hints=None):
        self.run_results = list(run_results)
        self.prior_hints = list(prior_hints or [])
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
                    "prior_hints": [
                        {"rationale": h, "confidence": 0.5, "interventions": []}
                        for h in self.prior_hints
                    ],
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
            result = dict(self.run_results.pop(0))
            result["candidate_id"] = str(kwargs.get("candidate_id"))
            return json.dumps(result)

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


def _request(
    toolset: Toolset, *, max_candidates: int = 4, cascade=None, max_turns=None
) -> HarnessRunRequest:
    extra: dict[str, Any] = {"max_candidates": max_candidates, "seed": 11}
    if cascade is not None:
        extra["cascade"] = cascade
    return HarnessRunRequest(
        toolset=toolset,
        system_instructions="be brief",
        user_prompt="optimise",
        model_id="mistral:mistral-large-latest",
        max_turns=max_turns,
        extra=extra,
    )


def _drive(harness, request):
    async def _collect():
        return [ev async for ev in harness.run(request)]

    return asyncio.run(_collect())


def _result(*, compile_ok=True, correctness_ok=True, median_ms=0.5, speedup=2.0,
            slots_remaining=0, successful=None) -> dict[str, Any]:
    return {
        "compile_ok": compile_ok,
        "median_ms": median_ms,
        "speedup_vs_baseline": speedup,
        "correctness_ok": correctness_ok,
        "compile_diagnostics": None if compile_ok else "NameError: tl",
        "compile_warnings": (
            [] if correctness_ok is not False
            else ["g2_allclose FAILED: drifted"]
        ),
        "successful": (
            bool(compile_ok and correctness_ok) if successful is None else successful
        ),
        "slots_remaining": slots_remaining,
        "max_abs_diff": 0.0,
    }


def _finished(events):
    assert events[-1].kind is StreamEventKind.RUN_FINISHED, (
        events[-1].error_message
    )
    return events[-1]


# ----------------------------------------------------------------- config


def test_registration_installs_cascade():
    import compilagent.integrations.archetype_harnesses  # noqa: F401

    assert "cascade" in harness_registry.ids()


def test_config_disable_accepts_ingredients_and_bundles():
    cfg = CascadeConfig.from_extra({"cascade": {"disable": ["c1", "c10"]}})
    assert not cfg.c1_plan_then_implement
    assert not cfg.c10_contrastive_reseed
    assert cfg.c2_staged_feedback

    cfg = CascadeConfig.from_extra({"cascade": {"disable": "budget"}})
    assert not cfg.c4_seed_round
    assert not cfg.c5_judge_ranking
    assert not cfg.c8_deadband
    assert cfg.c1_plan_then_implement

    assert set(AXIS_BUNDLES) == {"proposal", "feedback", "budget", "memory"}
    with pytest.raises(ValueError):
        CascadeConfig.from_extra({"cascade": {"disable": ["c99"]}})


def test_config_field_overrides():
    cfg = CascadeConfig.from_extra(
        {"cascade": {"seed_count": 2, "judge_top_m": 1, "deadband_pct": 2.0}}
    )
    assert cfg.seed_count == 2
    assert cfg.judge_top_m == 1
    assert cfg.deadband_pct == 2.0


# --------------------------------------------------------- novelty filter


def test_normalize_strips_comments_and_whitespace():
    a = "import torch\n\ndef f(x):\n    # a comment\n    return x + 1\n"
    b = "import torch\ndef f(x):\n    return x + 1  # different comment\n"
    assert normalize_module_source(a) == normalize_module_source(b)


def test_normalize_preserves_real_differences():
    a = "def f(x):\n    return x + 1\n"
    b = "def f(x):\n    return x + 2\n"
    assert normalize_module_source(a) != normalize_module_source(b)
    # Different nesting must not collide.
    c = "def f(x):\n    if x:\n        return 1\n    return 2\n"
    d = "def f(x):\n    if x:\n        return 1\n        return 2\n"
    assert normalize_module_source(c) != normalize_module_source(d)


def test_normalize_survives_unparseable_text():
    assert normalize_module_source("def broken(:\n  # hm\n  pass") != ""


# --------------------------------------------------------------- deadband


def test_deadband_accepts_only_above_one_percent():
    db = DeadbandController(deadband_pct=1.0)
    assert db.consider(1.02, 1.0) is True          # +2% → accept
    assert db.consider(1.025, 1.02) is False        # +0.49% → revert
    assert db.consider(None, 1.02) is False         # failure → non-improve
    assert db.non_improvements == 2


def test_deadband_plateau_reseed_once_then_stop():
    db = DeadbandController(deadband_pct=1.0, plateau_reseed=2, plateau_stop=4)
    db.consider(1.0, 1.5)
    assert not db.should_reseed
    db.consider(1.0, 1.5)
    assert db.should_reseed and not db.should_stop
    db.mark_reseeded()
    assert not db.should_reseed  # single re-seed
    db.consider(1.0, 1.5)
    assert not db.should_stop
    db.consider(1.0, 1.5)
    assert db.should_stop


def test_deadband_improvement_resets_plateau():
    db = DeadbandController(deadband_pct=1.0)
    db.consider(1.0, 1.5)
    db.consider(1.0, 1.5)
    assert db.non_improvements == 2
    assert db.consider(1.6, 1.5) is True
    assert db.non_improvements == 0


# ------------------------------------------------------------ judge parse


def test_parse_judge_ranking_happy_path_and_fallback():
    ids = ["cand-1", "cand-2", "cand-3"]
    assert parse_judge_ranking('["cand-2", "cand-3", "cand-1"]', ids) == [
        "cand-2", "cand-3", "cand-1",
    ]
    # Unknown ids dropped, missing appended in submission order.
    assert parse_judge_ranking('["cand-9", "cand-2"]', ids) == [
        "cand-2", "cand-1", "cand-3",
    ]
    # Garbage → submission order.
    assert parse_judge_ranking("no json here", ids) == ids


# ------------------------------------------------------------- full loop


def test_cascade_full_loop_seed_judge_then_plan_implement():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.2, slots_remaining=2),
            _result(speedup=1.4, slots_remaining=1),
            _result(speedup=1.5, slots_remaining=0),
        ],
        prior_hints=_HINTS,
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [
                _module("seed_a = 'rows'"),
                _module("seed_b = 'online'"),
                '["cand-2", "cand-1"]',                  # judge
                "PLAN: widen rows-per-program to 4 — bandwidth bound.",
                _module("child = 'wider rows'"),
            ],
            record,
        )
    )
    events = _drive(
        harness,
        _request(
            session.toolset(),
            max_candidates=3,
            cascade={"seed_count": 2},
        ),
    )
    finished = _finished(events)

    # 2 seeds (temp 1.0) + 1 judge + plan + implement = 5 LLM calls.
    assert len(record) == 5
    assert record[0]["temperature"] == 1.0
    assert record[1]["temperature"] == 1.0
    # C6: skill rules injected into seed prompts.
    assert "Known constraints" in record[0]["history"][0][1]
    assert "tl.sum has NO mask" in record[0]["history"][0][1]
    # C5: judge saw both gate-passers, no timings.
    judge_p = record[2]["history"][0][1]
    assert "Candidate cand-1" in judge_p and "Candidate cand-2" in judge_p
    assert "PREDICTED" in judge_p and "1.2" not in judge_p
    # C1: plan asked before implement; C3: incumbent context present.
    plan_p = record[3]["history"][0][1]
    assert "State a PLAN" in plan_p
    assert "Current incumbent" in plan_p
    assert "seed_b = 'online'" in plan_p  # judge winner became incumbent
    impl_p = record[4]["history"][0][1]
    assert "widen rows-per-program" in impl_p
    assert "Implement exactly this plan" in impl_p
    # C1: plan string recorded as the proposal description → rationale.
    serial = session.proposed[2]
    assert "widen rows-per-program" in serial["description"]
    assert "widen rows-per-program" in serial["interventions"][0]["rationale"]
    # Judge-rank vs actual-speedup pairs in run metadata (T1 measurement).
    pairs = finished.extra["judge"]["pairs"]
    assert pairs == [
        {"candidate_id": "cand-2", "judge_rank": 1, "actual_speedup": 1.4},
        {"candidate_id": "cand-1", "judge_rank": 2, "actual_speedup": 1.2},
    ]
    assert finished.extra["incumbent_speedup"] == pytest.approx(1.5)
    assert finished.extra["accepted_count"] == 2  # seed winner + child
    assert session.reflections == ["compare_runs", "synthesize_findings"]
    assert finished.extra["llm_calls"] == 5


def test_cascade_novelty_filter_blocks_rejected_duplicates():
    session = _FakeSession(
        run_results=[
            _result(correctness_ok=False, median_ms=None, speedup=None,
                    slots_remaining=2),
            _result(speedup=1.3, slots_remaining=1),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [
                _module("bad = 1\nx = bad"),
                # Same module modulo comments/whitespace → must be refused
                # WITHOUT propose/run.
                _module("bad = 1  # retry\n\nx = bad"),
                _module("good = 2\ny = good"),
            ],
            record,
        )
    )
    events = _drive(
        harness,
        _request(
            session.toolset(),
            max_candidates=2,
            max_turns=3,
            cascade={"disable": ["c1", "c4"]},
        ),
    )
    finished = _finished(events)
    assert len(record) == 3
    assert len(session.proposed) == 2  # the duplicate never reached a tool
    assert finished.extra["novelty_rejections"] == 1
    # The model was told why.
    assert "Novelty filter" in record[2]["history"][0][1]
    # C2 staged feedback after the gate failure was gate+error ONLY.
    assert "g2_allclose FAILED: drifted" in record[1]["history"][0][1]
    assert "Timing history" not in record[1]["history"][0][1]


def test_cascade_deadband_reseeds_then_stops_early():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.5, slots_remaining=9),
            _result(speedup=1.505, slots_remaining=8),   # ≤1% → revert
            _result(speedup=1.49, slots_remaining=7),    # revert → plateau 2
            _result(speedup=1.51, slots_remaining=6),    # reseed child ≤1%
            _result(speedup=1.50, slots_remaining=5),    # plateau 4 → stop
        ]
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [
                _module("a = 'first'"),
                _module("b = 'second'"),
                _module("c = 'third'"),
                _module("d = 'reseed child'"),
                _module("e = 'fifth'"),
            ],
            record,
        )
    )
    events = _drive(
        harness,
        _request(
            session.toolset(),
            max_candidates=10,
            max_turns=8,
            cascade={"disable": ["c1", "c4"]},
        ),
    )
    finished = _finished(events)
    assert len(record) == 5  # early stop, not max_turns
    # The 4th generation is the C10 contrastive re-seed.
    reseed_p = record[3]["history"][0][1]
    assert "Parent A (best overall)" in reseed_p
    assert "a = 'first'" in reseed_p
    assert record[3]["temperature"] == CascadeHarness.SEED_TEMPERATURE
    assert finished.extra["reseed_used"] is True
    assert finished.extra["early_stop"] is True
    assert finished.extra["incumbent_speedup"] == pytest.approx(1.5)
    assert finished.extra["accepted_count"] == 1


# ------------------------------------------------------- ingredient toggles


def test_toggle_c1_off_single_generate_per_turn():
    session = _FakeSession(run_results=[_result(speedup=1.2, slots_remaining=0)])
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator([_module("only = 1")], record)
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=1,
                 cascade={"disable": ["c1", "c4"]}),
    )
    _finished(events)
    assert len(record) == 1
    assert "State a PLAN" not in record[0]["history"][0][1]
    assert "Propose the next improved module" in record[0]["history"][0][1]


def test_toggle_c2_off_uses_plain_verdict_feedback():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.2, median_ms=0.8, slots_remaining=1),
            _result(speedup=1.3, slots_remaining=0),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [_module("first = 1"), _module("second = 2")], record
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=2, max_turns=4,
                 cascade={"disable": ["c1", "c4", "c2"]}),
    )
    _finished(events)
    follow_up = record[1]["history"][0][1]
    assert "CORRECT but" in follow_up           # E4 verdict text
    assert "Timing history" not in follow_up    # staged table disabled


def test_toggle_c2_on_correct_candidate_gets_history_table():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.2, median_ms=0.8, slots_remaining=1),
            _result(speedup=1.3, slots_remaining=0),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [_module("first = 1"), _module("second = 2")], record
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=2, max_turns=4,
                 cascade={"disable": ["c1", "c4"]}),
    )
    _finished(events)
    follow_up = record[1]["history"][0][1]
    assert "Timing history" in follow_up
    assert "vs the incumbent" in follow_up


def test_toggle_c3_off_drops_incumbent_context():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.2, slots_remaining=1),
            _result(speedup=1.3, slots_remaining=0),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [_module("first = 1"), _module("second = 2")], record
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=2, max_turns=4,
                 cascade={"disable": ["c1", "c4", "c3"]}),
    )
    _finished(events)
    assert "Current incumbent" not in record[1]["history"][0][1]


def test_toggle_c4_off_skips_seed_round():
    session = _FakeSession(run_results=[_result(speedup=1.2, slots_remaining=0)])
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            ["PLAN: tile differently.", _module("serial = 1")], record
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=1, cascade={"disable": ["c4"]}),
    )
    finished = _finished(events)
    # No temp-1.0 menu-dropout samples; straight to plan+implement.
    assert len(record) == 2
    assert "Focus especially" not in record[0]["history"][0][1]
    assert "State a PLAN" in record[0]["history"][0][1]
    assert finished.extra["judge"]["pairs"] == []


def test_toggle_c5_off_no_judge_call():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.2, slots_remaining=1),
            _result(speedup=1.4, slots_remaining=0),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [_module("seed_a = 1"), _module("seed_b = 2")], record
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=2,
                 cascade={"seed_count": 2, "disable": ["c5"]}),
    )
    finished = _finished(events)
    assert len(record) == 2  # seeds only — no judge generation
    assert finished.extra["judge"]["ranking"] == []
    # Without the judge, ALL gate-passers stay eligible → best is incumbent.
    assert finished.extra["incumbent_speedup"] == pytest.approx(1.4)


def test_toggle_c6_off_no_rule_injection():
    session = _FakeSession(
        run_results=[_result(speedup=1.2, slots_remaining=0)],
        prior_hints=_HINTS,
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator([_module("seed = 1")], record)
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=1,
                 cascade={"seed_count": 1, "disable": ["c6"]}),
    )
    _finished(events)
    assert "Known constraints" not in record[0]["history"][0][1]


def test_toggle_c7_off_duplicates_are_proposed_again():
    session = _FakeSession(
        run_results=[
            _result(correctness_ok=False, median_ms=None, speedup=None,
                    slots_remaining=1),
            _result(correctness_ok=False, median_ms=None, speedup=None,
                    slots_remaining=1),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [_module("bad = 1"), _module("bad = 1  # same")], record
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=1, max_turns=2,
                 cascade={"disable": ["c1", "c4", "c7"]}),
    )
    finished = _finished(events)
    assert len(session.proposed) == 2  # duplicate spent budget
    assert finished.extra["novelty_rejections"] == 0


def test_toggle_c8_off_accepts_any_strict_improvement():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.005, slots_remaining=1),  # within the deadband
            _result(speedup=1.006, slots_remaining=0),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [_module("tiny = 1"), _module("tinier = 2")], record
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=2, max_turns=4,
                 cascade={"disable": ["c1", "c4", "c8"]}),
    )
    finished = _finished(events)
    assert finished.extra["accepted_count"] == 2
    assert finished.extra["incumbent_speedup"] == pytest.approx(1.006)
    assert finished.extra["early_stop"] is False


def test_toggle_c10_off_plateau_never_reseeds():
    session = _FakeSession(
        run_results=[
            _result(speedup=1.5, slots_remaining=9),
            _result(speedup=1.50, slots_remaining=8),
            _result(speedup=1.50, slots_remaining=7),
            _result(speedup=1.50, slots_remaining=6),
            _result(speedup=1.50, slots_remaining=5),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            [_module(f"v{i} = {i}") for i in range(5)], record
        )
    )
    events = _drive(
        harness,
        _request(session.toolset(), max_candidates=10, max_turns=8,
                 cascade={"disable": ["c1", "c4", "c10"]}),
    )
    finished = _finished(events)
    assert finished.extra["reseed_used"] is False
    assert finished.extra["early_stop"] is True
    assert all(
        "Parent A" not in r["history"][0][1] for r in record
    )


def test_cascade_continuation_skips_seed_round():
    from compilagent.session.completion import RunSnapshot

    harness = CascadeHarness(generate_fn=_scripted_generator([], []))
    previous = _request(Toolset(tools=()), max_candidates=4)
    snap = RunSnapshot(
        successful_count=3,
        failed_attempts=1,
        max_candidates=4,
        max_failed_attempts=12,
        tools_called=frozenset(),
        iteration=0,
        max_continuations=4,
        harness_failed=False,
        best_speedup=1.2,
    )
    nxt = harness.build_continuation_request(previous, snap)
    assert nxt.extra["cascade_continuation"] is True
    assert nxt.extra["max_candidates"] == 1
