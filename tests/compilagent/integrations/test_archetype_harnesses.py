"""Unit tests for the archetype harnesses (no LLM, no GPU).

A fake toolset stands in for the session: scripted `run_candidate` verdicts
let us check the serial-refinement feedback loop (compile error → failing
gate → "correct but slower"), the best-of-N fan-out, token accounting, and
the reflection-lock calls — all through the canonical event stream.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from compilagent.core.tool_decl import ToolDecl
from compilagent.harness.base import HarnessRunRequest, StreamEventKind
from compilagent.harness.registry import harness_registry
from compilagent.session.completion import RunSnapshot
from compilagent.toolset import Toolset

from compilagent.integrations.archetype_harnesses.harness import (
    ArchetypeBestOfNHarness,
    ArchetypeSerialRefinementHarness,
)
from compilagent.integrations.archetype_harnesses.prompts import (
    EXEMPLAR,
    base_prompt,
    extract_code,
    feedback_for_run_result,
)

_MODULE_TEXT = "Here you go:\n```python\nimport torch.nn as nn\nclass ModelNew(nn.Module):\n    pass\n```"


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


def _stub_generator(record: list[dict[str, Any]]):
    async def generate(*, history, system=None, temperature=0.7, max_tokens=None):
        record.append(
            {
                "history": list(history),
                "system": system,
                "temperature": temperature,
            }
        )
        return _MODULE_TEXT, {
            "request_tokens": 100,
            "response_tokens": 50,
            "total_tokens": 150,
        }

    return generate


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


def test_registration_installs_both_archetypes():
    import compilagent.integrations.archetype_harnesses  # noqa: F401

    assert "archetype_sr" in harness_registry.ids()
    assert "archetype_bon" in harness_registry.ids()
    sr = harness_registry.get("archetype_sr")
    assert "mistral:mistral-large-latest" in sr.example_models


# ------------------------------------------------------------------ prompts


def test_base_prompt_mirrors_probe_wording_and_exemplar():
    prompt = base_prompt(
        reference_source="class Model: ...",
        task_description="Row-wise softmax.",
        banned_patterns=["softmax"],
    )
    assert prompt.startswith(
        "You optimize PyTorch programs by writing custom Triton GPU kernels."
    )
    assert "Output exactly ONE fenced python code block" in prompt
    assert EXEMPLAR in prompt
    assert "add_kernel" in prompt  # the vector-add exemplar
    assert "Banned torch APIs" in prompt


def test_feedback_messages_cover_all_three_verdicts():
    fail = feedback_for_run_result(_result(compile_ok=False))
    assert "failed to run" in fail and "NameError" in fail

    gate = feedback_for_run_result(
        _result(
            correctness_ok=False,
            median_ms=None,
            speedup=None,
            warnings=["g3_no_alias_no_mutation FAILED: output aliases an input"],
        )
    )
    assert "FAILED a correctness gate" in gate and "aliases" in gate

    slower = feedback_for_run_result(_result(median_ms=2.0, speedup=0.5))
    assert "CORRECT but" in slower and "0.500x" in slower


def test_extract_code_takes_largest_fenced_block():
    text = "```python\nshort\n```\nand\n```python\nmuch longer block here\n```"
    assert extract_code(text) == "much longer block here"
    assert extract_code("no code") is None


# -------------------------------------------------------------- archetype_sr


def test_sr_refines_until_budget_met_with_verdict_feedback():
    session = _FakeSession(
        run_results=[
            _result(compile_ok=False, median_ms=None, speedup=None,
                    correctness_ok=None, slots_remaining=2),
            _result(correctness_ok=False, median_ms=None, speedup=None,
                    slots_remaining=2,
                    warnings=["g2_allclose FAILED: drifted"]),
            _result(slots_remaining=1, speedup=1.2),
            _result(slots_remaining=0, speedup=1.5),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeSerialRefinementHarness(generate_fn=_stub_generator(record))
    events = _drive(harness, _request(session.toolset()))

    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    # 4 generations: compile-fail, gate-fail, correct-but-slow, final.
    assert len(record) == 4
    assert len(session.proposed) == 4
    assert session.ran == ["cand-1", "cand-2", "cand-3", "cand-4"]
    # G+E chain: each turn extends ONE conversation with verdict feedback.
    assert record[0]["temperature"] == 0.7
    assert record[1]["temperature"] == 0.5
    h2 = record[1]["history"]
    assert h2[-1][0] == "user" and "failed to run" in h2[-1][1]
    h3 = record[2]["history"]
    assert "FAILED a correctness gate" in h3[-1][1] and "drifted" in h3[-1][1]
    h4 = record[3]["history"]
    assert "CORRECT but" in h4[-1][1]
    # Reflection lock satisfied.
    assert session.reflections == ["compare_runs", "synthesize_findings"]
    # Token accounting accumulated across calls.
    usage = events[-1].extra["usage"]
    assert usage == {
        "request_tokens": 400,
        "response_tokens": 200,
        "total_tokens": 600,
    }
    assert events[-1].extra["llm_calls"] == 4


def test_sr_recovers_from_propose_rejection():
    session = _FakeSession(run_results=[_result(slots_remaining=0)])
    toolset = session.toolset()

    rejections = {"n": 0}
    original = toolset.by_name("propose_candidate").handler

    def flaky_propose(**kwargs):
        if rejections["n"] == 0:
            rejections["n"] += 1
            raise ValueError("intervention #0 rejected by backend: no ModelNew")
        return original(**kwargs)

    tools = tuple(
        _decl("propose_candidate", flaky_propose) if t.name == "propose_candidate" else t
        for t in toolset.tools
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeSerialRefinementHarness(generate_fn=_stub_generator(record))
    events = _drive(harness, _request(Toolset(tools=tools)))

    kinds = [e.kind for e in events]
    assert StreamEventKind.TOOL_ERROR in kinds  # surfaced, not raised
    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    # Second generation saw the rejection feedback.
    assert "rejected" in record[1]["history"][-1][1]


def test_sr_ledger_assertion_fires_on_timed_gate_failure():
    bad = _result(correctness_ok=False, median_ms=0.5, speedup=2.0,
                  slots_remaining=0)
    session = _FakeSession(run_results=[bad])
    harness = ArchetypeSerialRefinementHarness(
        generate_fn=_stub_generator([])
    )
    events = _drive(harness, _request(session.toolset()))
    assert events[-1].kind is StreamEventKind.RUN_FAILED
    assert "budget-ledger" in (events[-1].error_message or "")


# ------------------------------------------------------------- archetype_bon


def test_bon_submits_n_independent_samples_at_temperature_one():
    session = _FakeSession(
        run_results=[
            _result(slots_remaining=2, speedup=1.1),
            _result(slots_remaining=1, speedup=0.9),
            _result(slots_remaining=0, speedup=1.4),
        ]
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeBestOfNHarness(generate_fn=_stub_generator(record))
    events = _drive(harness, _request(session.toolset(), max_candidates=3))

    assert len(record) == 3
    assert all(r["temperature"] == 1.0 for r in record)
    # No feedback: every sample sees exactly the same single-turn prompt.
    assert all(len(r["history"]) == 1 for r in record)
    assert record[0]["history"] == record[1]["history"] == record[2]["history"]
    assert len(session.proposed) == 3
    assert session.reflections == ["compare_runs", "synthesize_findings"]
    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    assert events[-1].extra["llm_calls"] == 3
    assert events[-1].extra["usage"]["total_tokens"] == 450


def test_bon_continuation_samples_only_remaining_budget():
    harness = ArchetypeBestOfNHarness(generate_fn=_stub_generator([]))
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
    assert nxt.extra["max_candidates"] == 1


def test_run_failure_is_an_event_not_an_exception():
    def boom(**_kw) -> str:
        raise RuntimeError("session exploded")

    toolset = Toolset(tools=(_decl("inspect_workload", boom),))
    harness = ArchetypeBestOfNHarness(generate_fn=_stub_generator([]))
    events = _drive(harness, _request(toolset, max_candidates=1))
    assert events[-1].kind is StreamEventKind.RUN_FAILED
    assert "session exploded" in (events[-1].error_message or "")


# --------------------------------------------------- session-level smoke (CPU)


@pytest.mark.parametrize("harness_cls", [
    ArchetypeSerialRefinementHarness,
    ArchetypeBestOfNHarness,
])
def test_archetypes_drive_a_real_session_with_a_fake_backend(
    tmp_path, harness_cls
):
    """End-to-end against a real OptimizationSession (fake source backend)."""

    from dataclasses import dataclass, field
    from pathlib import Path as _Path
    from collections.abc import Sequence

    from compilagent.core.analysis import (
        Analysis,
        CompileResult,
        CorrectnessResult,
        DeviceCapability,
        TimingResult,
    )
    from compilagent.core.backend import backend_registry
    from compilagent.core.plan import Plan, ValidationResult
    from compilagent.core.search_space import SearchSpace
    from compilagent.core.workload import (
        BenchmarkBudget,
        ToleranceConfig,
        WorkloadInstance,
        WorkloadKind,
        WorkloadSpec,
    )
    from compilagent.core.workload_registry import register_workload
    from compilagent.session.session import OptimizationSession, run_session
    from compilagent.storage.workspace import OptimizationWorkspace

    @dataclass
    class _FakeSourceBackend:
        id: str = "fake_source"
        artifact_stages: tuple[str, ...] = ("module_source",)

        def device_capability(self):
            return DeviceCapability(
                arch="cpu", capability_int=None, name="Fake",
                memory_total_bytes=None, memory_peak_bandwidth_gbps=None,
            )

        def analyze(self, workload, *, baseline_artifacts):
            return Analysis(summary={"kind": workload.kind.value})

        def derive_search_space(self, workload, analysis):
            return SearchSpace(workload_id=workload.id, backend_id=self.id)

        def validate_intervention(self, intervention):
            if intervention.target.kind == "source_replace":
                return ValidationResult(ok=True)
            return ValidationResult(ok=False, errors=("only source_replace",))

        def interpret_plan(self, plan):
            return plan

        def apply_intervention(self, plan, intervention):
            return Plan(interventions=plan.interventions + (intervention,))

        def compile(self, workload, plan, *, artifact_dir, pass_callback=None):
            return CompileResult(ok=True, elapsed_ms=1.0)

        def time_workload(self, workload, plan, *, warmup, repetitions, max_seconds=None):
            ms = 10.0 if plan.is_empty else 5.0
            return TimingResult(
                timings_ms=(ms,) * 3, median_ms=ms, p20_ms=ms, p80_ms=ms
            )

        def validate_correctness(self, workload, baseline, candidate, tolerance):
            return CorrectnessResult(ok=True)

        def reset_between_compiles(self, workload):
            return None

        def list_introspection_tools(self) -> Sequence:
            return ()

        def list_artifact_renderers(self) -> Sequence:
            return ()

        def infer_workload_family(self, workload):
            return None

        def objectives_for_candidate(self, workload, plan, compile_result, timing_result):
            return {}

    backend_registry.register("fake_source", _FakeSourceBackend)
    spec = WorkloadSpec(
        id="fake_source_workload",
        title="fake",
        description="fake source workload",
        kind=WorkloadKind.KERNEL,
        backend_id="fake_source",
        tolerance=ToleranceConfig(),
        budget=BenchmarkBudget(warmup=1, repetitions=3, max_seconds=5.0),
        metadata={
            "reference_module_source": "class Model: ...",
            "banned_patterns": ["softmax"],
        },
    )

    @register_workload(spec)
    def _build(s):
        return WorkloadInstance(spec=s, forward=lambda: None)

    session = OptimizationSession(
        workload_id="fake_source_workload",
        workspace=OptimizationWorkspace(session_cwd=tmp_path),
        max_candidates=1,
    )
    harness = harness_cls(generate_fn=_stub_generator([]))
    request = HarnessRunRequest(
        toolset=session.toolset,
        system_instructions="",
        user_prompt="go",
        model_id="mistral:mistral-large-latest",
        extra={"max_candidates": 1},
    )
    result = asyncio.run(
        run_session(session=session, harness=harness, request=request)
    )
    assert result.metadata["completion_reason"] == "budget_met"
    assert result.metadata["usage"]["total_tokens"] > 0
    assert session.budget_state["successful_count"] == 1
