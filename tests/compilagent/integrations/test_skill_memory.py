"""Unit tests for `ExperimentLogPolicy` (CASCADE C6 over the E9 hooks).

Covers: rules-file seeding from the P0 failure taxonomy, consult-time rule
injection, the frequency gate on run-distilled rules (a lesson must recur
before it earns prompt space), observe-time distillation/persistence, and
the ExperimentLog `recall` hint.
"""

from __future__ import annotations

import json

from compilagent.core.analysis import Analysis, CompileResult, CorrectnessResult
from compilagent.core.plan import Plan
from compilagent.core.workload import WorkloadKind, WorkloadSpec
from compilagent.integrations.archetype_harnesses.skill_memory import (
    P0_SEED_RULES,
    ExperimentLogPolicy,
)
from compilagent.storage.experiment_log import ExperimentLog


def _spec() -> WorkloadSpec:
    return WorkloadSpec(
        id="softmax_4096",
        title="softmax",
        description="row-wise softmax",
        kind=WorkloadKind.KERNEL,
        backend_id="triton_source",
    )


def _consult(policy: ExperimentLogPolicy):
    return policy.consult(
        workload=_spec(),
        analysis=Analysis(),
        family="reduction",
        arch="cuda:sm_120",
    )


def _observe_failure(policy: ExperimentLogPolicy, *, gate="g2_allclose",
                     diagnostics="g2_allclose: max_abs_diff=0.13"):
    policy.observe(
        workload=_spec(),
        candidate_id="cand-1",
        plan=Plan(),
        compile_result=CompileResult(ok=True),
        timing=None,
        correctness=CorrectnessResult(
            ok=False, failed_at=gate, diagnostics=diagnostics
        ),
        speedup=None,
        successful=False,
        family="reduction",
        arch="cuda:sm_120",
    )


def test_first_consult_seeds_rules_file_from_p0_taxonomy(tmp_path):
    policy = ExperimentLogPolicy(tmp_path)
    hints = _consult(policy)

    assert policy.rules_path.exists()
    saved = json.loads(policy.rules_path.read_text(encoding="utf-8"))
    assert {r["id"] for r in saved["rules"]} == {rid for rid, _ in P0_SEED_RULES}

    rationales = [h.rationale for h in hints]
    assert len(rationales) == len(P0_SEED_RULES)
    assert any("tl.sum has NO mask" in r for r in rationales)
    assert any("K-tail" in r for r in rationales)
    assert any("single-pass" in r for r in rationales)
    assert all(r.startswith("constraint rule: ") for r in rationales)
    assert all(h.suggested_interventions == () for h in hints)


def test_observe_distills_failure_with_frequency_counter(tmp_path):
    policy = ExperimentLogPolicy(tmp_path)
    _observe_failure(policy)

    rules = json.loads(policy.rules_path.read_text(encoding="utf-8"))["rules"]
    distilled = [r for r in rules if r["source"] == "distilled"]
    assert len(distilled) == 1
    assert distilled[0]["frequency"] == 1
    assert distilled[0]["id"].startswith("gate:g2_allclose:")
    assert "numerical drift" in distilled[0]["text"]
    assert "max_abs_diff=0.13" in distilled[0]["text"]

    # Same failure again → frequency increments, no duplicate rule.
    _observe_failure(policy)
    rules = json.loads(policy.rules_path.read_text(encoding="utf-8"))["rules"]
    distilled = [r for r in rules if r["source"] == "distilled"]
    assert len(distilled) == 1
    assert distilled[0]["frequency"] == 2


def test_distilled_rules_injected_only_at_frequency_two(tmp_path):
    policy = ExperimentLogPolicy(tmp_path)
    _observe_failure(policy)
    rationales = [h.rationale for h in _consult(policy)]
    assert not any("numerical drift" in r for r in rationales)  # freq 1

    _observe_failure(policy)
    rationales = [h.rationale for h in _consult(policy)]
    assert any("numerical drift" in r for r in rationales)  # freq 2 — injected


def test_observe_distills_compile_failures_from_diagnostics_head(tmp_path):
    policy = ExperimentLogPolicy(tmp_path)
    policy.observe(
        workload=_spec(),
        candidate_id="cand-2",
        plan=Plan(),
        compile_result=CompileResult(
            ok=False,
            diagnostics="TypeError: sum() got an unexpected keyword "
            "argument 'mask'\n  File \"cand.py\", line 12",
        ),
        timing=None,
        correctness=None,
        speedup=None,
        successful=False,
        family="reduction",
        arch="cuda:sm_120",
    )
    rules = json.loads(policy.rules_path.read_text(encoding="utf-8"))["rules"]
    distilled = [r for r in rules if r["source"] == "distilled"]
    assert len(distilled) == 1
    assert distilled[0]["id"].startswith("compile:")
    assert "unexpected keyword argument 'mask'" in distilled[0]["text"]
    assert "line 12" not in distilled[0]["text"]  # head line only


def test_observe_ignores_successful_candidates(tmp_path):
    policy = ExperimentLogPolicy(tmp_path)
    _consult(policy)  # seed the rules file
    policy.observe(
        workload=_spec(),
        candidate_id="cand-3",
        plan=Plan(),
        compile_result=CompileResult(ok=True),
        timing=None,
        correctness=CorrectnessResult(ok=True),
        speedup=1.4,
        successful=True,
        family="reduction",
        arch="cuda:sm_120",
    )
    rules = json.loads(policy.rules_path.read_text(encoding="utf-8"))["rules"]
    assert all(r["source"] == "p0_seed" for r in rules)


def test_consult_recalls_best_prior_result_via_experiment_log(tmp_path):
    log = ExperimentLog(tmp_path)
    log.append(
        {
            "run_id": "run-old",
            "workload_id": "sum_reduce_1m",
            "backend_id": "triton_source",
            "family": "reduction",
            "arch": "cuda:sm_120",
            "successful": True,
            "speedup": 1.8,
            "interventions": [
                {
                    "target": {
                        "kind": "source_replace",
                        "selector": "kernel_source",
                    },
                    "payload": {"module_source": "class ModelNew: ..."},
                    "rationale": "single-pass online softmax",
                }
            ],
        }
    )
    policy = ExperimentLogPolicy(tmp_path, experiment_log=log)
    hints = _consult(policy)

    prior = [h for h in hints if h.metadata.get("kind") == "prior_result"]
    assert len(prior) == 1
    assert "sum_reduce_1m" in prior[0].rationale
    assert "1.8" in prior[0].rationale
    iv = prior[0].suggested_interventions[0]
    assert iv.target.kind == "source_replace"
    assert iv.payload == {"module_source": "class ModelNew: ..."}
    assert iv.rationale == "single-pass online softmax"

    # Different family → no prior-result hint.
    other = policy.consult(
        workload=_spec(), analysis=Analysis(), family="matmul", arch="cuda:sm_120"
    )
    assert not any(h.metadata.get("kind") == "prior_result" for h in other)


def test_corrupt_rules_file_is_reseeded(tmp_path):
    policy = ExperimentLogPolicy(tmp_path)
    policy.rules_path.parent.mkdir(parents=True, exist_ok=True)
    policy.rules_path.write_text("{not json", encoding="utf-8")
    hints = _consult(policy)
    assert len(hints) == len(P0_SEED_RULES)


def test_policy_is_a_structural_candidate_policy():
    from compilagent.core.candidate_policy import CandidatePolicy

    assert isinstance(ExperimentLogPolicy(__import__("pathlib").Path(".")),
                      CandidatePolicy)
    assert ExperimentLogPolicy.name == "experiment_log"
