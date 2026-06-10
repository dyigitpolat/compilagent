"""GPU tests for the `triton_source` sandbox: timeout, aliasing gate, and a
hand-written CORRECT Triton softmax passing all E2a gates end-to-end.

These run the real subprocess sandbox and therefore need CUDA + triton.
Run pinned to a free GPU, e.g.:

    CUDA_VISIBLE_DEVICES=1 pytest tests/compilagent/integrations/test_triton_source_gpu.py
"""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for triton_source sandbox tests", allow_module_level=True)

from compilagent.core.backend import backend_registry
from compilagent.core.plan import Intervention, Plan, Target
from compilagent.core.workload import (
    BenchmarkBudget,
    ToleranceConfig,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.core.workload_registry import workload_registry

_TINY_REF = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.softmax(x, dim=1)

    def get_inputs():
        return [torch.randn(64, 64, device='cuda', dtype=torch.float32)]
    """
).strip()


def _tiny_spec(workload_id: str, **metadata_overrides) -> WorkloadSpec:
    metadata = {
        "reference_module_source": _TINY_REF,
        "banned_patterns": ["torch.softmax", "softmax"],
        "input_shapes": {"x": [64, 64]},
        "input_dtypes": ["fp32"],
        "op_signature": "softmax(x: f32[64,64], dim=1)",
        # Keep the GPU tier fast: tiny timing protocol.
        "sandbox_warmup": 3,
        "sandbox_repetitions": 10,
        **metadata_overrides,
    }
    return WorkloadSpec(
        id=workload_id,
        title="tiny softmax",
        description="tiny softmax for sandbox tests",
        kind=WorkloadKind.KERNEL,
        backend_id="triton_source",
        tolerance=ToleranceConfig(atol=1e-4, rtol=1e-3),
        budget=BenchmarkBudget(warmup=3, repetitions=10, max_seconds=120.0),
        metadata=metadata,
    )


def _backend():
    import compilagent.integrations.triton_source  # noqa: F401

    return backend_registry.get("triton_source")


def _plan(module_source: str) -> Plan:
    return Plan(
        interventions=(
            Intervention(
                target=Target("source_replace", "kernel_source"),
                payload={"module_source": module_source},
            ),
        )
    )


# ----------------------------------------------------------- sandbox timeout


def test_hanging_candidate_is_killed_not_the_session(tmp_path: Path):
    backend = _backend()
    spec = _tiny_spec("tiny_softmax_timeout", sandbox_timeout_seconds=45)
    hang = textwrap.dedent(
        """
        import torch, torch.nn as nn

        while True:
            pass

        class ModelNew(nn.Module):
            def forward(self, x):
                return x
        """
    )
    started = time.perf_counter()
    result = backend.compile(spec, _plan(hang), artifact_dir=tmp_path)
    elapsed = time.perf_counter() - started
    assert not result.ok
    assert result.metadata["timed_out"] is True
    assert "timeout" in (result.diagnostics or "").lower()
    # The parent survived and came back roughly at the deadline.
    assert elapsed < 120


# ----------------------------------------------------------- aliasing gate g3


def test_candidate_returning_its_input_fails_g3(tmp_path: Path):
    backend = _backend()
    spec = _tiny_spec("tiny_softmax_alias")
    alias = textwrap.dedent(
        """
        import torch, torch.nn as nn

        class ModelNew(nn.Module):
            def forward(self, x):
                return x
        """
    )
    result = backend.compile(spec, _plan(alias), artifact_dir=tmp_path)
    # It ran (no exec error), so compile is ok — the gates carry the verdict.
    assert result.ok
    gates = result.metadata["gates"]
    assert gates["g3_no_alias_no_mutation"]["ok"] is False
    assert "alias" in gates["g3_no_alias_no_mutation"]["message"]

    # Gate-failing candidates are never timed (budget-ledger rule) ...
    timing = backend.time_workload(spec, _plan(alias), warmup=3, repetitions=10)
    assert timing.median_ms is None
    assert "NOT" in (timing.diagnostics or "")
    # ... and correctness is served from the cached verdicts.
    correctness = backend.validate_correctness(
        spec, result, result, spec.tolerance
    )
    assert not correctness.ok
    assert correctness.failed_at == "g3_no_alias_no_mutation"


def test_candidate_mutating_its_input_fails_g3(tmp_path: Path):
    backend = _backend()
    spec = _tiny_spec("tiny_softmax_mutate")
    mutate = textwrap.dedent(
        """
        import torch, torch.nn as nn

        class ModelNew(nn.Module):
            def forward(self, x):
                x.zero_()
                return torch.empty_like(x)
        """
    )
    result = backend.compile(spec, _plan(mutate), artifact_dir=tmp_path)
    assert result.ok
    gates = result.metadata["gates"]
    assert gates["g3_no_alias_no_mutation"]["ok"] is False
    assert "mutated" in gates["g3_no_alias_no_mutation"]["message"]


# ----------------------------------- correct Triton softmax end-to-end (E2a)


_CORRECT_TRITON_SOFTMAX = textwrap.dedent(
    """
    import torch, torch.nn as nn
    import triton, triton.language as tl

    @triton.jit
    def softmax_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float('inf'))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        tl.store(out_ptr + row * stride + offs, num / den, mask=mask)

    class ModelNew(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            out = torch.empty_like(x)
            n_rows, n_cols = x.shape
            BLOCK = triton.next_power_of_2(n_cols)
            softmax_kernel[(n_rows,)](
                x, out, n_cols, x.stride(0), BLOCK=BLOCK, num_warps=8
            )
            return out
    """
).strip()


def test_correct_triton_softmax_passes_all_gates_end_to_end(tmp_path: Path):
    backend = _backend()
    import compilagent.integrations.triton_source  # noqa: F401

    spec = workload_registry.get_spec("softmax_4096")
    plan = _plan(_CORRECT_TRITON_SOFTMAX)

    # Baseline first (empty Plan times the reference module itself).
    baseline = backend.compile(spec, Plan(), artifact_dir=tmp_path / "baseline")
    assert baseline.ok, baseline.diagnostics
    baseline_timing = backend.time_workload(spec, Plan(), warmup=25, repetitions=100)
    assert baseline_timing.median_ms and baseline_timing.median_ms > 0

    candidate = backend.compile(spec, plan, artifact_dir=tmp_path / "cand")
    assert candidate.ok, candidate.diagnostics
    gates = candidate.metadata["gates"]
    for gate_name, verdict in gates.items():
        assert verdict["ok"], f"{gate_name} failed: {verdict['message']}"
    assert set(gates) == {
        "g1_shape_dtype",
        "g2_allclose",
        "g3_no_alias_no_mutation",
        "g4_banned_api",
        "g5_determinism",
    }

    timing = backend.time_workload(spec, plan, warmup=25, repetitions=100)
    assert timing.median_ms and timing.median_ms > 0
    assert len(timing.timings_ms) >= 50  # 100 reps minus 10% trim each tail

    correctness = backend.validate_correctness(spec, baseline, candidate, spec.tolerance)
    assert correctness.ok, correctness.diagnostics
    assert correctness.max_abs_diff is not None
    assert correctness.max_abs_diff < 1e-4 + 1e-3  # within the g2 envelope

    # The whole evaluation came from ONE subprocess run: the candidate dir
    # has exactly one sandbox payload + logs.
    assert (tmp_path / "cand" / "sandbox_stdout.log").exists()
    speedup = (baseline_timing.median_ms or 0.0) / timing.median_ms
    assert speedup > 0.1  # sanity: same order of magnitude as eager
