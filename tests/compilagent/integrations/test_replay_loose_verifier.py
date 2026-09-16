"""The loose-verifier replay (scripts/replay_loose_verifier.py) and the
``skip_gates`` payload flag it relies on.

The GPU test at the end runs the real runner in-process and needs CUDA;
pin it to a free device (GPU 0 is reserved on this machine):

    CUDA_VISIBLE_DEVICES=1 pytest tests/compilagent/integrations/test_replay_loose_verifier.py
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from compilagent.integrations.triton_source._internal import sandbox
from scripts import replay_loose_verifier as replay


def _row(key: str, harness: str, workload: str, run_id: str, candidates: list, **extra) -> dict:
    return {
        "key": key,
        "harness": harness,
        "workload": workload,
        "run_id": run_id,
        "seed": 13,
        "budget": 8,
        "model_id": "m",
        "completion_reason": "budget_met",
        "candidates": candidates,
        **extra,
    }


def _cand(cid: str, *, compile_ok: bool, correct: bool | None, failing: tuple[str, ...] = ()) -> dict:
    gates = {g: {"ok": True, "message": ""} for g in replay.ALL_GATES}
    for g in failing:
        gates[g] = {"ok": False, "message": "nope"}
    return {
        "candidate_id": cid,
        "compile_ok": compile_ok,
        "correctness_ok": correct,
        "gates": gates if compile_ok else {},
        "diagnostics": None if compile_ok else "CompilationError",
    }


def test_population_keeps_compiled_gate_rejected_candidates_only(tmp_path: Path) -> None:
    ledger = tmp_path / "t.jsonl"
    rows = [
        _row("a", "archetype_sr", "softmax_4096", "run-1", [
            _cand("c-pass", compile_ok=True, correct=True),
            _cand("c-g2", compile_ok=True, correct=False, failing=("g2_allclose",)),
            _cand("c-g2g5", compile_ok=True, correct=False, failing=("g5_determinism", "g2_allclose")),
            _cand("c-compile", compile_ok=False, correct=None),
        ]),
        # an error row and a harness-failed row are not good rows
        _row("b", "cascade", "softmax_4096", "run-2", [
            _cand("c-x", compile_ok=True, correct=False, failing=("g2_allclose",))], error="boom"),
        _row("c", "cascade", "softmax_4096", "run-3", [
            _cand("c-y", compile_ok=True, correct=False, failing=("g2_allclose",))],
            completion_reason="harness_failed"),
        # a superseded row: the later row with the same key wins
        _row("d", "archetype_bon", "gelu_8192x1024", "run-4", [
            _cand("c-old", compile_ok=True, correct=False, failing=("g3_no_alias_no_mutation",))]),
        _row("d", "archetype_bon", "gelu_8192x1024", "run-5", [
            _cand("c-new", compile_ok=True, correct=False, failing=("g3_no_alias_no_mutation",))]),
    ]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    items = replay.population("headline", replay.load_good_rows(ledger))
    keys = [it["key"] for it in items]
    assert keys == [
        "gelu_8192x1024|run-5|c-new",
        "softmax_4096|run-1|c-g2",
        "softmax_4096|run-1|c-g2g5",
    ]
    by_key = {it["key"]: it for it in items}
    assert by_key["softmax_4096|run-1|c-g2g5"]["original_failing_gates"] == [
        "g2_allclose", "g5_determinism",
    ]
    assert by_key["softmax_4096|run-1|c-g2"]["harness"] == "archetype_sr"


def test_summary_counts_admissions_and_apparent_speedups() -> None:
    rows = [
        {"key": "k1", "harness": "archetype_sr", "ledger": "headline",
         "original_failing_gates": ["g2_allclose"], "loose_admitted": True, "apparent_speedup": 2.5},
        {"key": "k2", "harness": "archetype_sr", "ledger": "headline",
         "original_failing_gates": ["g2_allclose"], "loose_admitted": True, "apparent_speedup": 0.8},
        {"key": "k3", "harness": "cascade", "ledger": "kb24",
         "original_failing_gates": ["g2_allclose", "g5_determinism"], "loose_admitted": False,
         "apparent_speedup": None},
        {"key": "k4", "harness": "cascade", "ledger": "kb24",
         "original_failing_gates": ["g5_determinism"], "loose_admitted": True, "apparent_speedup": 1.3},
        {"key": "k5", "harness": "cascade", "ledger": "kb24", "error": "timeout"},
    ]
    s = replay.summarize(rows, population_size=10)
    assert s["population_size"] == 10
    assert s["evaluated"] == 4 and s["errors"] == 1
    assert s["admitted"] == 3 and s["admitted_fraction"] == 0.75
    assert s["apparent_speedup"] == {
        "n": 3, "max": 2.5, "median": 1.3, "count_gt_1.0": 2, "count_gt_1.2": 2, "count_ge_2.0": 1,
    }
    assert s["stock_fast1_wins"] == 2
    assert s["by_original_failing_gates"]["g5_determinism"] == {"evaluated": 1, "admitted": 1, "gt1": 1}
    assert s["by_harness"]["archetype_sr"] == {"evaluated": 2, "admitted": 2, "gt1": 1}


def test_skip_gates_reaches_the_runner_payload(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["payload"] = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout=sandbox.MARKER + json.dumps({"compiled": True, "gates": {}}) + "\n", stderr="",
        )

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.delenv("COMPILAGENT_GPU_POOL", raising=False)

    sandbox.run_sandboxed_eval(
        reference_source="ref", candidate_source="cand", artifact_dir=tmp_path / "a",
    )
    assert "skip_gates" not in seen["payload"]  # ordinary runs: payload unchanged

    sandbox.run_sandboxed_eval(
        reference_source="ref", candidate_source="cand", artifact_dir=tmp_path / "b",
        atol=1e-2, rtol=1e-2, skip_gates=replay.SKIPPED_GATES,
    )
    assert seen["payload"]["skip_gates"] == list(replay.SKIPPED_GATES)
    assert seen["payload"]["atol"] == 1e-2


# ---------------------------------------------------------------------------
# GPU: the runner's gate policy under skip_gates
# ---------------------------------------------------------------------------

_REF = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class Model(nn.Module):
        def forward(self, x):
            return x * 2.0

    def get_inputs():
        return [torch.randn(64, 64, device="cuda")]
    """
)

# Right values, wrong manners: the output IS the (mutated) input buffer, so
# g1/g2 pass while g3 fails on both aliasing and mutation.
_MUTATING_CAND = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class ModelNew(nn.Module):
        def forward(self, x):
            x.mul_(2.0)
            return x
    """
)


def _evaluate(payload: dict, workdir: Path) -> dict:
    from compilagent.integrations.triton_source._internal.sandbox_runner import evaluate

    return evaluate({**payload, "workdir": str(workdir)})


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(), reason="CUDA required"
)
def test_runner_skip_gates_reports_but_does_not_withhold_timing(tmp_path: Path) -> None:
    base = {
        "reference_source": _REF, "candidate_source": _MUTATING_CAND,
        "atol": 1e-2, "rtol": 1e-2, "warmup": 2, "repetitions": 5,
        "trial_seeds": [100, 101],
    }
    strict = _evaluate(base, tmp_path / "strict")
    assert strict["gates"]["g3_no_alias_no_mutation"]["ok"] is False
    assert strict["cand_ms"] is None  # historical behaviour: no timing

    loose = _evaluate({**base, "skip_gates": list(replay.SKIPPED_GATES)}, tmp_path / "loose")
    assert loose["gates"]["g1_shape_dtype"]["ok"] is True
    assert loose["gates"]["g2_allclose"]["ok"] is True
    assert loose["gates"]["g3_no_alias_no_mutation"]["ok"] is False  # still reported
    assert loose["gates"]["g5_determinism"]["ok"] is False  # not measurable after mutation
    assert loose["cand_ms"] is not None and loose["ref_ms"] is not None
    assert loose["speedup_vs_ref"] is not None
