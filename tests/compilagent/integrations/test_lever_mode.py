"""Unit tests for lever mode (ticket D9): backend-generic archetypes.

No LLM, no GPU. Covers codec selection off the derived SearchSpace,
lever-mode prompt construction (catalog, evidence, example JSON),
intervention-JSON parsing including the malformed-JSON retry path through
the SR chain, novelty normalization (C7 over intervention JSON), the
lever-mode skill-rule store (C6), bandit lever arms, and the pilot
workload→backend routing tables.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from compilagent.core.tool_decl import ToolDecl
from compilagent.harness.base import HarnessRunRequest, StreamEventKind
from compilagent.integrations.archetype_harnesses import lever_prompts
from compilagent.integrations.archetype_harnesses.bandit import (
    ArchetypeBanditHarness,
)
from compilagent.integrations.archetype_harnesses.candidate_codec import (
    LEVER_CODEC,
    SOURCE_CODEC,
    codec_for_context,
)
from compilagent.integrations.archetype_harnesses.cascade import CascadeHarness
from compilagent.integrations.archetype_harnesses.harness import (
    ArchetypeSerialRefinementHarness,
)
from compilagent.toolset import Toolset

# ------------------------------------------------------------ shared fakes

_KNOB_LEVER = {
    "id": "knob:inductor.max_autotune",
    "backend_id": "torch_inductor",
    "target": {"kind": "knob", "selector": "inductor.max_autotune"},
    "range": {"kind": "bool", "candidates": [True, False]},
    "default": False,
    "description": "inductor.max_autotune (type=bool).",
    "evidence": {
        "rule": "torch_inductor.knobs",
        "signal": "curated interesting knob",
        "citations": [],
    },
}

_PASS_LEVER = {
    "id": "pass:ttgir:tritongpu-pipeline",
    "backend_id": "triton",
    "target": {"kind": "pass", "selector": "ttgir:tritongpu-pipeline"},
    "range": {
        "kind": "structured_json",
        "examples": [{"action": "skip"}, {"action": "run"}],
        "schema_hint": '{"action": "run"|"skip"|"replace", "args"?: {...}}',
    },
    "default": {"action": "run"},
    "description": "Override the tritongpu-pipeline MLIR pass.",
    "evidence": {
        "rule": "triton.pass_impact",
        "signal": "tritongpu-pipeline ran for 1.20ms and modified IR",
        "citations": [],
    },
}

_SOURCE_LEVER = {
    "id": "kernel_source",
    "backend_id": "triton_source",
    "target": {"kind": "source_replace", "selector": "kernel_source"},
    "range": {"kind": "structured_json", "examples": [], "schema_hint": ""},
    "default": None,
    "description": "full module source",
    "evidence": {"rule": "e1", "signal": "s", "citations": []},
}


def _lever_context(levers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "workload_id": "rmsnorm",
        "workload_kind": "full_model",
        "backend_id": "torch_inductor",
        "task_description": "Optimize Inductor compile decisions for RMSNorm.",
        "reference_source": "",
        "banned_patterns": [],
        "baseline_median_ms": 0.5,
        "device": {"arch": "cuda:sm_120", "name": "RTX PRO 6000"},
        "analysis_summary": {"fx_ops": ["aten.mean", "aten.rsqrt"]},
        "prior_hints": [],
        "levers": levers if levers is not None else [_KNOB_LEVER, _PASS_LEVER],
    }


def _decl(name: str, handler) -> ToolDecl:
    return ToolDecl(
        name=name,
        description=name,
        args_schema={"type": "object", "properties": {}},
        handler=handler,
        read_only=False,
    )


class _FakeLeverSession:
    """Scripted tool surface advertising a knob/pass SearchSpace."""

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
                        "id": "rmsnorm",
                        "kind": "full_model",
                        "description": "RMSNorm module under torch.compile.",
                        "metadata": {},
                    },
                    "backend_id": "torch_inductor",
                    "device": {"arch": "cuda:sm_120", "name": "RTX"},
                    "analysis_summary": {},
                    "baseline_timing": {"median_ms": 0.5},
                }
            )

        def inspect_search_space(**_kw) -> str:
            return json.dumps(
                {
                    "workload_id": "rmsnorm",
                    "backend_id": "torch_inductor",
                    "lever_count": 2,
                    "levers": [_KNOB_LEVER, _PASS_LEVER],
                }
            )

        def propose_candidate(**kwargs) -> str:
            self.proposed.append(kwargs)
            return json.dumps({"id": f"cand-{len(self.proposed)}"})

        def run_candidate(**kwargs) -> str:
            self.ran.append(str(kwargs.get("candidate_id")))
            result = dict(self.run_results.pop(0))
            result.setdefault("candidate_id", str(kwargs.get("candidate_id")))
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
                    ("inspect_search_space", inspect_search_space),
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


def _request(
    toolset: Toolset, *, max_turns: int | None = None, **extra
) -> HarnessRunRequest:
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


def _result(*, compile_ok=True, correctness_ok=True, median_ms=0.4, speedup=1.25,
            slots_remaining=0, successful=None) -> dict[str, Any]:
    return {
        "compile_ok": compile_ok,
        "median_ms": median_ms,
        "speedup_vs_baseline": speedup,
        "correctness_ok": correctness_ok,
        "compile_diagnostics": None if compile_ok else "Inductor compile raised",
        "compile_warnings": [],
        "successful": (
            bool(compile_ok and correctness_ok) if successful is None else successful
        ),
        "slots_remaining": slots_remaining,
        "max_abs_diff": 0.0,
    }


_PLAN_TEXT = (
    "Reasoning...\n```json\n"
    '[{"target": {"kind": "knob", "selector": "inductor.max_autotune"}, '
    '"payload": true, "rationale": "autotune the GEMMs"}]\n```'
)


# -------------------------------------------------------- codec selection


def test_codec_selection_keys_off_the_search_space():
    # NB: compare by `.name` — the integrations conftest reloads modules
    # per test, so singleton identity does not survive.
    assert codec_for_context(_lever_context()).name == LEVER_CODEC.name
    assert codec_for_context(_lever_context(levers=[])).name == "source"
    assert (
        codec_for_context(_lever_context(levers=[_SOURCE_LEVER])).name
        == SOURCE_CODEC.name
    )
    # A mixed space containing the source lever stays in source mode.
    assert (
        codec_for_context(_lever_context(levers=[_SOURCE_LEVER, _KNOB_LEVER])).name
        == "source"
    )


# ------------------------------------------------------ prompt construction


def test_lever_base_prompt_presents_catalog_evidence_and_format():
    prompt = LEVER_CODEC.base_prompt(_lever_context())
    assert prompt.startswith(
        "You are a compiler-heuristic researcher. Replace baseline compiler "
        "decisions with verified faster ones."
    )
    # Lever catalog with kinds, selectors, ranges, defaults, and evidence.
    assert "knob:inductor.max_autotune" in prompt
    assert "pass:ttgir:tritongpu-pipeline" in prompt
    assert "tritongpu-pipeline ran for 1.20ms and modified IR" in prompt
    assert '"action": "run"|"skip"|"replace"' in prompt
    # Baseline-is-the-empty-plan framing + the typed output contract.
    assert "EMPTY intervention list" in prompt
    assert "do NOT write kernel or module code" in prompt
    assert "fenced json code block" in prompt
    # Synthesized example drawn from the catalog itself.
    assert '"selector": "inductor.max_autotune"' in prompt


def test_lever_menu_dropout_thins_the_lever_menu():
    import random

    prompt = LEVER_CODEC.menu_dropout_prompt(
        _lever_context(), rng=random.Random(7)
    )
    assert prompt.startswith("You are a compiler-heuristic researcher.")
    assert "Focus especially on these optimization directions:" in prompt
    kept = [m for m in lever_prompts.OPTIMIZATION_MENU if m in prompt]
    assert kept  # at least one direction survives dropout


def test_lever_strategy_arms_have_source_arity():
    from compilagent.integrations.archetype_harnesses.bandit import STRATEGY_ARMS

    assert len(LEVER_CODEC.strategy_arms) == len(STRATEGY_ARMS) == 5
    names = [name for name, _ in LEVER_CODEC.strategy_arms]
    assert names == [
        "flip_flags", "scale_numeric", "switch_enum", "skip_decision",
        "combine_levers",
    ]


def test_lever_catalog_caps_rendered_levers():
    many = [
        {**_KNOB_LEVER, "target": {"kind": "knob", "selector": f"inductor.k{i}"}}
        for i in range(lever_prompts.MAX_PROMPT_LEVERS + 10)
    ]
    block = lever_prompts.search_space_block(_lever_context(levers=many))
    assert f"({len(many)} total" in block
    assert "+10 more levers not shown" in block


# -------------------------------------------------------- parsing/encoding


def test_extract_interventions_accepts_fenced_nested_and_flat_shapes():
    nested = LEVER_CODEC.extract(_PLAN_TEXT)
    assert nested is not None
    parsed = json.loads(nested)
    assert parsed[0]["target"] == {
        "kind": "knob", "selector": "inductor.max_autotune",
    }
    assert parsed[0]["payload"] is True

    flat = LEVER_CODEC.extract(
        '```json\n[{"target_kind": "pass", "target_selector": '
        '"ttgir:tritongpu-pipeline", "payload": {"action": "skip"}}]\n```'
    )
    assert json.loads(flat)[0]["target"]["kind"] == "pass"

    bare = LEVER_CODEC.extract(
        'Plan: [{"target": {"kind": "knob", "selector": "x"}, "payload": 1}]'
    )
    assert json.loads(bare)[0]["payload"] == 1


def test_extract_interventions_rejects_malformed_payloads():
    assert LEVER_CODEC.extract("no json here") is None
    assert LEVER_CODEC.extract("```json\n{not json}\n```") is None
    # Parseable JSON that is not an intervention list is still a format miss.
    assert LEVER_CODEC.extract('```json\n["cand-1", "cand-2"]\n```') is None
    assert LEVER_CODEC.extract('```json\n[{"payload": 3}]\n```') is None


def test_propose_args_emits_canonical_typed_interventions():
    candidate = LEVER_CODEC.extract(_PLAN_TEXT)
    args = LEVER_CODEC.propose_args(candidate, description="turn 0")
    assert args["interventions"] == [
        {
            "target_kind": "knob",
            "target_selector": "inductor.max_autotune",
            "payload": True,
            "rationale": "autotune the GEMMs",
        }
    ]
    assert args["description"] == "turn 0"


# ----------------------------------------------------- novelty (C7, lever)


def test_normalize_interventions_ignores_rationale_and_order():
    a = json.dumps(
        [
            {"target": {"kind": "knob", "selector": "a"}, "payload": 1,
             "rationale": "first wording"},
            {"target": {"kind": "pass", "selector": "b"},
             "payload": {"action": "skip"}, "rationale": "x"},
        ]
    )
    b = json.dumps(
        [
            {"target": {"kind": "pass", "selector": "b"},
             "payload": {"action": "skip"}, "rationale": "totally reworded"},
            {"target": {"kind": "knob", "selector": "a"}, "payload": 1,
             "rationale": "second wording"},
        ]
    )
    assert LEVER_CODEC.normalize(a) == LEVER_CODEC.normalize(b)

    different = json.dumps(
        [{"target": {"kind": "knob", "selector": "a"}, "payload": 2}]
    )
    assert LEVER_CODEC.normalize(a) != LEVER_CODEC.normalize(different)
    # Unparseable text degrades to whitespace collapse, never raises.
    assert LEVER_CODEC.normalize("{oops") == "{oops"


def test_cascade_novelty_filter_blocks_duplicate_plans_in_lever_mode():
    session = _FakeLeverSession(
        run_results=[_result(compile_ok=False, median_ms=None, speedup=None,
                             correctness_ok=None, slots_remaining=1)]
    )
    reworded_duplicate = _PLAN_TEXT.replace(
        "autotune the GEMMs", "different rationale, same plan"
    )
    record: list[dict[str, Any]] = []
    harness = CascadeHarness(
        generate_fn=_scripted_generator(
            # C4/C5 off → serial turns only; C1 off → one generate per turn.
            [_PLAN_TEXT, reworded_duplicate, _PLAN_TEXT],
            record,
        )
    )
    events = _drive(
        harness,
        _request(
            session.toolset(),
            max_turns=3,
            max_candidates=4,
            seed=3,
            cascade={"disable": "c1,c4,c5,c8"},
        ),
    )
    final = events[-1]
    assert final.kind is StreamEventKind.RUN_FINISHED, final.error_message
    # Turn 0 proposes and gets rejected; turns 1+2 are normalized duplicates
    # of the rejected plan → refused with ZERO further tool calls.
    assert len(session.proposed) == 1
    assert final.extra["novelty_rejections"] == 2
    # The duplicate feedback is the lever-mode wording.
    assert "intervention plan is identical" in record[2]["history"][0][1]


# ------------------------------------------- SR chain in lever mode (retry)


def test_sr_lever_mode_retries_malformed_json_then_submits_typed_plan():
    session = _FakeLeverSession(run_results=[_result(slots_remaining=0)])
    record: list[dict[str, Any]] = []
    harness = ArchetypeSerialRefinementHarness(
        generate_fn=_scripted_generator(
            ["I would tune num_stages (no JSON given).", _PLAN_TEXT], record
        )
    )
    events = _drive(harness, _request(session.toolset()))

    assert events[-1].kind is StreamEventKind.RUN_FINISHED, (
        events[-1].error_message
    )
    # Malformed first reply → retry-format feedback, zero tool calls spent.
    retry_msg = record[1]["history"][-1][1]
    assert "fenced json code block" in retry_msg
    assert len(session.proposed) == 1
    # The submitted candidate is the typed intervention, not source_replace.
    iv = session.proposed[0]["interventions"][0]
    assert iv["target_kind"] == "knob"
    assert iv["target_selector"] == "inductor.max_autotune"
    assert iv["payload"] is True
    # The base prompt was lever-mode.
    assert "Derived decision levers" in record[0]["history"][0][1]
    assert session.reflections == ["compare_runs", "synthesize_findings"]


def test_sr_lever_mode_surfaces_backend_rejection_as_feedback():
    session = _FakeLeverSession(run_results=[_result(slots_remaining=0)])
    toolset = session.toolset()
    original = toolset.by_name("propose_candidate").handler
    rejections = {"n": 0}

    def flaky_propose(**kwargs):
        if rejections["n"] == 0:
            rejections["n"] += 1
            raise ValueError(
                "intervention #0 rejected by backend: unsupported target.kind"
            )
        return original(**kwargs)

    tools = tuple(
        _decl("propose_candidate", flaky_propose)
        if t.name == "propose_candidate" else t
        for t in toolset.tools
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeSerialRefinementHarness(
        generate_fn=_scripted_generator([_PLAN_TEXT, _PLAN_TEXT], record)
    )
    events = _drive(harness, _request(Toolset(tools=tools)))

    kinds = [e.kind for e in events]
    assert StreamEventKind.TOOL_ERROR in kinds  # retryable, not fatal
    assert events[-1].kind is StreamEventKind.RUN_FINISHED
    feedback = record[1]["history"][-1][1]
    assert "rejected before reaching the compiler" in feedback
    assert "unsupported target.kind" in feedback


# ------------------------------------------------------- bandit lever arms


def test_bandit_uses_lever_arms_and_empty_plan_incumbent_in_lever_mode():
    session = _FakeLeverSession(
        run_results=[_result(slots_remaining=0, speedup=1.1)]
    )
    record: list[dict[str, Any]] = []
    harness = ArchetypeBanditHarness(
        generate_fn=_scripted_generator([_PLAN_TEXT], record)
    )
    events = _drive(harness, _request(session.toolset()))

    final = events[-1]
    assert final.kind is StreamEventKind.RUN_FINISHED, final.error_message
    assert set(final.extra["arms"]) == {
        name for name, _ in LEVER_CODEC.strategy_arms
    }
    prompt = record[0]["history"][0][1]
    # First pull: incumbent is the empty plan (the baseline heuristics).
    assert "```json\n[]\n```" in prompt
    assert "FLIP-FLAGS" in prompt  # arm 0 pulled first


# --------------------------------------------------- skill memory (C6, D9)


def test_skill_memory_uses_lever_rule_store_for_lever_backends(tmp_path):
    from compilagent.core.analysis import Analysis, CompileResult
    from compilagent.core.plan import Plan
    from compilagent.core.workload import WorkloadKind, WorkloadSpec
    from compilagent.integrations.archetype_harnesses.skill_memory import (
        LEVER_SEED_RULES,
        P0_SEED_RULES,
        ExperimentLogPolicy,
    )

    spec = WorkloadSpec(
        id="rmsnorm",
        title="rmsnorm",
        description="rmsnorm module",
        kind=WorkloadKind.FULL_MODEL,
        backend_id="torch_inductor",
    )
    policy = ExperimentLogPolicy(tmp_path)
    hints = policy.consult(
        workload=spec, analysis=Analysis(), family=None, arch="cuda:sm_120"
    )

    # Lever store created and seeded from the lever taxonomy; source store
    # untouched.
    assert policy.lever_rules_path.exists()
    assert not policy.rules_path.exists()
    saved = json.loads(policy.lever_rules_path.read_text(encoding="utf-8"))
    assert {r["id"] for r in saved["rules"]} == {rid for rid, _ in LEVER_SEED_RULES}
    rationales = [h.rationale for h in hints]
    assert len(rationales) == len(LEVER_SEED_RULES)
    assert any("tritongpu-pipeline" in r for r in rationales)
    assert not any("tl.sum" in r for r in rationales)  # no P0 leakage

    # observe() on a lever-backend failure distills into the LEVER file.
    policy.observe(
        workload=spec,
        candidate_id="cand-1",
        plan=Plan(),
        compile_result=CompileResult(
            ok=False, diagnostics="Inductor compile raised: KeyError('x')"
        ),
        timing=None,
        correctness=None,
        speedup=None,
        successful=False,
        family=None,
        arch="cuda:sm_120",
    )
    rules = json.loads(policy.lever_rules_path.read_text(encoding="utf-8"))["rules"]
    distilled = [r for r in rules if r["source"] == "distilled"]
    assert len(distilled) == 1 and "Inductor compile raised" in distilled[0]["text"]
    assert not policy.rules_path.exists()

    # Source-space consults still seed/serve the P0 taxonomy independently.
    src_spec = WorkloadSpec(
        id="softmax_4096",
        title="softmax",
        description="row softmax",
        kind=WorkloadKind.KERNEL,
        backend_id="triton_source",
    )
    src_hints = policy.consult(
        workload=src_spec, analysis=Analysis(), family=None, arch="cuda:sm_120"
    )
    assert policy.rules_path.exists()
    saved_src = json.loads(policy.rules_path.read_text(encoding="utf-8"))
    assert {r["id"] for r in saved_src["rules"]} == {rid for rid, _ in P0_SEED_RULES}
    assert any("tl.sum" in h.rationale for h in src_hints)


# -------------------------------------------------- pilot workload routing


def test_pilot_backend_routing_tables():
    from scripts.pilot_workloads import (
        KERNEL_WORKLOAD_IDS,
        MODULE_WORKLOAD_IDS,
        backend_for,
    )

    assert len(MODULE_WORKLOAD_IDS) == len(KERNEL_WORKLOAD_IDS) == 6
    assert all(backend_for(w) == "torch_inductor" for w in MODULE_WORKLOAD_IDS)
    assert all(backend_for(w) == "triton" for w in KERNEL_WORKLOAD_IDS)
    assert backend_for("softmax_4096") == "triton_source"
    assert backend_for("anything_else") == "triton_source"
