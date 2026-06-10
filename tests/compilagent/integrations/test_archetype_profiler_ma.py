"""Unit tests for `archetype_ma` (H-MA, profiler-in-the-loop) — no LLM, no
GPU, no ncu.

A scripted generator plays both roles (Coder code blocks, Judge JSON) and a
stub profiler stands in for the NCU runner, so the tests can check the
CudaForge-shaped round structure: optimization rounds profile the candidate
and feed the Judge timing + metrics; correction rounds feed the error log
and skip profiling; degraded (ncu_unavailable) profiles keep the loop
running and are recorded in the run metadata.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from compilagent.core.tool_decl import ToolDecl
from compilagent.harness.base import HarnessRunRequest, StreamEventKind
from compilagent.harness.registry import harness_registry
from compilagent.integrations.archetype_harnesses.profiler_ma import (
    ArchetypeProfilerMAHarness,
    error_log_for,
    parse_judge_analysis,
    timing_line,
)
from compilagent.integrations.archetype_harnesses.prompts import (
    RETRY_FORMAT_MESSAGE,
)
from compilagent.session.completion import RunSnapshot
from compilagent.toolset import Toolset

_CODE_1 = "```python\nclass ModelNew:\n    VARIANT = 1\n```"
_CODE_2 = "```python\nclass ModelNew:\n    VARIANT = 2\n```"

_JUDGE_OPT = json.dumps(
    {
        "bottleneck": "DRAM-bound: 81% DRAM vs 21% SM throughput",
        "optimization_method": "widen per-program tiles",
        "modification_plan": "process 4 rows per program with BLOCK=256",
    }
)
_JUDGE_CORR = json.dumps(
    {
        "critical_issue": "missing load mask",
        "why_it_matters": "out-of-bounds reads corrupt the tail",
        "minimal_fix_hint": "add mask to tl.load",
    }
)

_PROFILE_OK = {
    "ncu_unavailable": False,
    "reason": None,
    "error": None,
    "invocation": "ncu",
    "metrics": {
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": {
            "value": 81.0,
            "unit": "%",
            "description": "DRAM throughput, % of peak",
        }
    },
    "dominant_kernel": "softmax_kernel",
    "kernel_names": ["softmax_kernel"],
}
_PROFILE_DEGRADED = {
    "ncu_unavailable": True,
    "reason": "permission",
    "error": "ncu: ERR_NVGPUCTRPERM | sudo -n ncu: a password is required",
    "invocation": None,
    "metrics": {},
    "dominant_kernel": None,
    "kernel_names": [],
}


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

        def run_candidate(**_kwargs) -> str:
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


def _scripted_generator(record: list[dict[str, Any]], texts: list[str]):
    queue = list(texts)

    async def generate(*, history, system=None, temperature=0.7, max_tokens=None):
        record.append(
            {
                "prompt": history[-1][1],
                "temperature": temperature,
            }
        )
        return queue.pop(0), {
            "request_tokens": 100,
            "response_tokens": 50,
            "total_tokens": 150,
        }

    return generate


def _stub_profiler(calls: list[dict[str, Any]], profiles: list[dict[str, Any]]):
    queue = list(profiles)

    def profile(*, reference_source: str, candidate_source: str):
        calls.append(
            {
                "reference_source": reference_source,
                "candidate_source": candidate_source,
            }
        )
        return queue.pop(0)

    return profile


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
            slots_remaining=0, warnings=()) -> dict[str, Any]:
    return {
        "candidate_id": "cand-x",
        "compile_ok": compile_ok,
        "median_ms": median_ms,
        "speedup_vs_baseline": speedup,
        "correctness_ok": correctness_ok,
        "compile_diagnostics": None if compile_ok else "NameError: tl",
        "compile_warnings": list(warnings),
        "successful": bool(compile_ok and correctness_ok),
        "slots_remaining": slots_remaining,
        "max_abs_diff": 0.0,
    }


# ----------------------------------------------------------------- registry


def test_registration_installs_archetype_ma():
    import compilagent.integrations.archetype_harnesses  # noqa: F401

    assert "archetype_ma" in harness_registry.ids()
    harness = harness_registry.get("archetype_ma")
    assert harness.id == "archetype_ma"
    assert "mistral:mistral-large-latest" in harness.example_models


# ------------------------------------------------------------------ helpers


def test_parse_judge_analysis_strict_lenient_and_british_spelling():
    strict = parse_judge_analysis(
        "Sure!\n" + _JUDGE_OPT,
        ("bottleneck", "optimization_method", "modification_plan"),
    )
    assert strict["bottleneck"].startswith("DRAM-bound")
    assert set(strict) == {
        "bottleneck", "optimization_method", "modification_plan",
    }

    british = parse_judge_analysis(
        '{"bottleneck": "x", "optimisation_method": "y"}',
        ("bottleneck", "optimization_method"),
    )
    assert british["optimization_method"] == "y"

    fallback = parse_judge_analysis(
        "the kernel is memory bound, no json here", ("bottleneck",)
    )
    assert fallback == {"analysis": "the kernel is memory bound, no json here"}


def test_timing_line_and_error_log_helpers():
    line = timing_line(_result(median_ms=0.5, speedup=2.0))
    assert "0.5000 ms" in line and "1.0000 ms" in line and "2.000x" in line
    assert timing_line({"median_ms": None}) == "no timing signal recorded"

    assert "NameError" in error_log_for(_result(compile_ok=False))
    gate = error_log_for(
        _result(correctness_ok=False, warnings=["g2_allclose FAILED: drift"])
    )
    assert "g2_allclose FAILED" in gate


# ------------------------------------------------------- optimization rounds


def test_optimization_round_profiles_candidate_and_feeds_judge_then_coder():
    session = _FakeSession(
        run_results=[
            _result(slots_remaining=1, speedup=1.2, median_ms=0.8),
            _result(slots_remaining=0, speedup=1.5, median_ms=0.6),
        ]
    )
    record: list[dict[str, Any]] = []
    profile_calls: list[dict[str, Any]] = []
    harness = ArchetypeProfilerMAHarness(
        generate_fn=_scripted_generator(record, [_CODE_1, _JUDGE_OPT, _CODE_2]),
        profile_fn=_stub_profiler(profile_calls, [_PROFILE_OK]),
    )
    events = _drive(harness, _request(session.toolset()))

    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    # Exactly one profile: the gate-passing round-0 candidate; the final
    # round (slots_remaining == 0) is neither profiled nor judged.
    assert len(profile_calls) == 1
    assert profile_calls[0]["reference_source"] == "class Model: ..."
    assert "VARIANT = 1" in profile_calls[0]["candidate_source"]

    # Coder(0.7) → Judge(0.2) → Coder(0.5).
    assert [r["temperature"] for r in record] == [0.7, 0.2, 0.5]
    judge_prompt = record[1]["prompt"]
    assert "VARIANT = 1" in judge_prompt
    assert "speedup 1.200x" in judge_prompt
    assert "DRAM throughput, % of peak" in judge_prompt  # NCU metrics block
    assert "JSON" in judge_prompt

    coder_prompt = record[2]["prompt"]
    assert "Strictly apply" in coder_prompt
    assert "VARIANT = 1" in coder_prompt  # incumbent source
    assert "widen per-program tiles" in coder_prompt  # judge analysis
    assert "0.8000 ms" in coder_prompt  # incumbent timing

    assert session.reflections == ["compare_runs", "synthesize_findings"]
    extra = events[-1].extra
    assert extra["llm_calls"] == 3
    assert extra["usage"]["total_tokens"] == 450
    assert extra["incumbent_speedup"] == 1.5
    judge_meta = extra["judge"]
    assert judge_meta["profiled_rounds"] == 1
    assert judge_meta["ncu_unavailable"] is False
    modes = [r["mode"] for r in judge_meta["rounds"]]
    assert modes == ["optimization", "final"]
    assert judge_meta["rounds"][0]["dominant_kernel"] == "softmax_kernel"
    assert judge_meta["rounds"][0]["analysis"]["bottleneck"].startswith(
        "DRAM-bound"
    )


def test_degraded_profile_keeps_loop_running_and_flags_the_run():
    session = _FakeSession(
        run_results=[
            _result(slots_remaining=1, speedup=1.2),
            _result(slots_remaining=0, speedup=1.3),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeProfilerMAHarness(
        generate_fn=_scripted_generator(record, [_CODE_1, _JUDGE_OPT, _CODE_2]),
        profile_fn=_stub_profiler([], [_PROFILE_DEGRADED]),
    )
    events = _drive(harness, _request(session.toolset()))

    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    judge_prompt = record[1]["prompt"]
    # No-NCU H-MA (the CudaForge no-NCU ablation): the judge is told.
    assert "NCU profiling is UNAVAILABLE" in judge_prompt
    assert "DRAM throughput" not in judge_prompt
    judge_meta = events[-1].extra["judge"]
    assert judge_meta["ncu_unavailable"] is True
    assert judge_meta["rounds"][0]["ncu_unavailable"] is True
    assert judge_meta["rounds"][0]["ncu_reason"] == "permission"


# --------------------------------------------------------- correction rounds


def test_gate_failure_goes_to_correction_judge_without_profiling():
    session = _FakeSession(
        run_results=[
            _result(
                correctness_ok=False,
                median_ms=None,
                speedup=None,
                slots_remaining=2,
                warnings=["g2_allclose FAILED: drifted"],
            ),
            _result(slots_remaining=0, speedup=1.4),
        ]
    )
    record: list[dict[str, Any]] = []
    profile_calls: list[dict[str, Any]] = []
    harness = ArchetypeProfilerMAHarness(
        generate_fn=_scripted_generator(record, [_CODE_1, _JUDGE_CORR, _CODE_2]),
        profile_fn=_stub_profiler(profile_calls, []),
    )
    events = _drive(harness, _request(session.toolset()))

    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    assert profile_calls == []  # gate-failing candidates are never profiled
    judge_prompt = record[1]["prompt"]
    assert "ERROR LOG" in judge_prompt
    assert "g2_allclose FAILED" in judge_prompt
    assert "critical_issue" in judge_prompt

    coder_prompt = record[2]["prompt"]
    assert "FAILED" in coder_prompt
    assert "VARIANT = 1" in coder_prompt  # the failed kernel, not incumbent
    assert "add mask to tl.load" in coder_prompt  # judge's minimal fix hint

    judge_meta = events[-1].extra["judge"]
    assert [r["mode"] for r in judge_meta["rounds"]] == ["correction", "final"]
    # Nothing was profiled, so NCU availability is unknown (not False).
    assert judge_meta["ncu_unavailable"] is None
    assert judge_meta["profiled_rounds"] == 0


def test_propose_rejection_is_a_correction_round_with_zero_run_results():
    session = _FakeSession(run_results=[_result(slots_remaining=0)])
    toolset = session.toolset()
    rejections = {"n": 0}
    original = toolset.by_name("propose_candidate").handler

    def flaky_propose(**kwargs):
        if rejections["n"] == 0:
            rejections["n"] += 1
            raise ValueError("intervention #0 rejected: no ModelNew defined")
        return original(**kwargs)

    tools = tuple(
        _decl("propose_candidate", flaky_propose)
        if t.name == "propose_candidate"
        else t
        for t in toolset.tools
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeProfilerMAHarness(
        generate_fn=_scripted_generator(record, [_CODE_1, _JUDGE_CORR, _CODE_2]),
        profile_fn=_stub_profiler([], []),
    )
    events = _drive(harness, _request(Toolset(tools=tools)))

    kinds = [e.kind for e in events]
    assert StreamEventKind.TOOL_ERROR in kinds  # surfaced, not raised
    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    judge_prompt = record[1]["prompt"]
    assert "rejected" in judge_prompt  # rejection text is the ERROR_LOG
    rounds = events[-1].extra["judge"]["rounds"]
    assert rounds[0]["mode"] == "correction"
    assert rounds[0]["candidate_id"] is None


# ------------------------------------------------------------- loop plumbing


def test_no_code_reply_retries_with_format_message_and_no_judge():
    session = _FakeSession(run_results=[_result(slots_remaining=0)])
    record: list[dict[str, Any]] = []
    harness = ArchetypeProfilerMAHarness(
        generate_fn=_scripted_generator(record, ["no code, sorry", _CODE_1]),
        profile_fn=_stub_profiler([], []),
    )
    events = _drive(harness, _request(session.toolset()))

    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    assert len(record) == 2  # no judge call for a formatless reply
    assert record[1]["prompt"].endswith(RETRY_FORMAT_MESSAGE)
    assert events[-1].extra["llm_calls"] == 2


def test_ledger_assertion_fires_on_timed_gate_failure():
    bad = _result(
        correctness_ok=False, median_ms=0.5, speedup=2.0, slots_remaining=0
    )
    session = _FakeSession(run_results=[bad])
    harness = ArchetypeProfilerMAHarness(
        generate_fn=_scripted_generator([], [_CODE_1]),
        profile_fn=_stub_profiler([], []),
    )
    events = _drive(harness, _request(session.toolset()))
    assert events[-1].kind is StreamEventKind.RUN_FAILED
    assert "budget-ledger" in (events[-1].error_message or "")


def test_continuation_restarts_the_markovian_chain():
    harness = ArchetypeProfilerMAHarness(
        generate_fn=_scripted_generator([], []),
        profile_fn=_stub_profiler([], []),
    )
    previous = _request(Toolset(tools=()), max_candidates=4)
    snap = RunSnapshot(
        successful_count=3,
        failed_attempts=1,
        max_candidates=4,
        max_failed_attempts=12,
        tools_called=frozenset({"compare_runs", "synthesize_findings"}),
        iteration=0,
        max_continuations=4,
        harness_failed=False,
        best_speedup=1.2,
    )
    nxt = harness.build_continuation_request(previous, snap)
    assert "1 validated slot(s) remain" in nxt.user_prompt
    assert nxt.model_id == previous.model_id
