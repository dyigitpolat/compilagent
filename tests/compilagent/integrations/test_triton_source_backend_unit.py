"""Unit tests for the `triton_source` backend that need neither triton nor CUDA."""

from __future__ import annotations

import textwrap
from pathlib import Path

from compilagent.core.backend import Backend, backend_registry
from compilagent.core.plan import Intervention, Plan, Target
from compilagent.core.search_space import StructuredJsonRange
from compilagent.core.workload_registry import workload_registry
from compilagent.integrations.triton_source._internal.lint import lint_banned_apis


def _get_backend():
    import compilagent.integrations.triton_source  # noqa: F401

    return backend_registry.get("triton_source")


def _softmax_spec():
    import compilagent.integrations.triton_source  # noqa: F401

    return workload_registry.get_spec("softmax_4096")


# ------------------------------------------------------------- registration


def test_self_registration_installs_backend_and_workloads():
    import compilagent.integrations.triton_source  # noqa: F401
    from compilagent.integrations.triton_source.workloads import WORKLOAD_IDS

    assert "triton_source" in backend_registry.ids()
    b = backend_registry.get("triton_source")
    assert isinstance(b, Backend)
    assert b.id == "triton_source"
    for wid in WORKLOAD_IDS:
        spec = workload_registry.get_spec(wid)
        assert spec.backend_id == "triton_source"
        assert spec.metadata["reference_module_source"]
        assert spec.metadata["banned_patterns"]
        assert spec.metadata["input_shapes"]
    assert len(WORKLOAD_IDS) == 6


# --------------------------------------------------- lever derivation (E1)


def test_derived_lever_carries_reference_evidence():
    backend = _get_backend()
    spec = _softmax_spec()
    analysis = backend.analyze(spec, baseline_artifacts=())
    space = backend.derive_search_space(spec, analysis)

    assert len(space.levers) == 1
    lever = space.levers[0]
    assert lever.id == "kernel_source"
    assert lever.target_kind == "source_replace"
    assert isinstance(lever.range, StructuredJsonRange)
    # The evidence is REQUIRED at type level and must be derived, not
    # hand-coded: it cites the reference implementation and the shapes.
    assert lever.evidence.rule == "triton_source.reference_module"
    assert "reference implementation" in lever.evidence.signal
    assert "4096" in lever.evidence.signal


def test_derive_search_space_empty_without_reference():
    backend = _get_backend()
    spec = _softmax_spec()
    stripped = type(spec)(
        id="no_ref",
        title="t",
        description="d",
        kind=spec.kind,
        backend_id=spec.backend_id,
    )
    analysis = backend.analyze(stripped, baseline_artifacts=())
    assert backend.derive_search_space(stripped, analysis).levers == ()


# ------------------------------------------------------ validate_intervention


def test_validate_intervention_accepts_well_formed_source_replace():
    backend = _get_backend()
    iv = Intervention(
        target=Target("source_replace", "kernel_source"),
        payload={
            "module_source": "import torch.nn as nn\nclass ModelNew(nn.Module):\n    pass\n"
        },
    )
    assert backend.validate_intervention(iv).ok


def test_validate_intervention_rejects_unknown_kind_and_bad_payloads():
    backend = _get_backend()
    res = backend.validate_intervention(
        Intervention(target=Target("knob", "x"), payload={})
    )
    assert not res.ok and any("source_replace" in e for e in res.errors)

    res = backend.validate_intervention(
        Intervention(target=Target("source_replace", ""), payload={"module_source": ""})
    )
    assert not res.ok

    res = backend.validate_intervention(
        Intervention(
            target=Target("source_replace", ""),
            payload={"module_source": "def broken(:\n    pass"},
        )
    )
    assert not res.ok and any("syntax error" in e for e in res.errors)

    res = backend.validate_intervention(
        Intervention(
            target=Target("source_replace", ""),
            payload={"module_source": "class NotModelNew:\n    pass\n"},
        )
    )
    assert not res.ok and any("ModelNew" in e for e in res.errors)


# ------------------------------------------------------- banned-API lint (g4)


_CHEATING_SOFTMAX = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class ModelNew(nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=1)
    """
)

_HONEST_TRITON = textwrap.dedent(
    """
    import torch, torch.nn as nn
    import triton, triton.language as tl

    @triton.jit
    def softmax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n)
        tl.store(out_ptr + offs, tl.exp(x) / tl.sum(tl.exp(x), axis=0), mask=offs < n)

    class ModelNew(nn.Module):
        def forward(self, x):
            out = torch.empty_like(x)
            softmax_kernel[(x.shape[0],)](x, out, x.shape[1], BLOCK=1024)
            return out
    """
)


def test_lint_flags_reference_op_in_forward():
    violations = lint_banned_apis(_CHEATING_SOFTMAX, ["torch.softmax", "softmax"])
    assert violations
    assert "torch.softmax" in violations[0].detail
    assert "triton" in violations[0].message  # actionable hint for the agent


def test_lint_flags_method_form():
    src = "class ModelNew:\n    def forward(self, x):\n        return x.softmax(dim=1)\n"
    assert lint_banned_apis(src, ["softmax"])


def test_lint_clean_on_honest_triton_module():
    assert lint_banned_apis(_HONEST_TRITON, ["torch.softmax", "softmax", "Softmax"]) == []


def test_lint_exempts_init_but_flags_forward_call_through_alias():
    src = textwrap.dedent(
        """
        import torch.nn as nn

        class ModelNew(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(1024)   # allowed: parameter container

            def forward(self, x):
                return self.ln(x)              # banned: calls the op anyway
        """
    )
    violations = lint_banned_apis(src, ["LayerNorm", "layer_norm"])
    assert len(violations) == 1
    assert "self.ln" in violations[0].detail


def test_lint_init_only_instantiation_is_clean():
    src = textwrap.dedent(
        """
        import torch, torch.nn as nn

        class ModelNew(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(1024)

            def forward(self, x):
                return x * self.ln.weight + self.ln.bias
        """
    )
    assert lint_banned_apis(src, ["LayerNorm", "layer_norm"]) == []


def test_lint_flags_matmul_operator_but_not_decorators():
    src = textwrap.dedent(
        """
        import triton, triton.language as tl

        @triton.jit
        def k(a_ptr):
            pass

        class ModelNew:
            def forward(self, a, b):
                return a @ b
        """
    )
    violations = lint_banned_apis(src, ["@", "matmul"])
    assert len(violations) == 1
    assert violations[0].pattern == "@"


def test_lint_exempts_tl_calls_inside_jit_kernels():
    src = textwrap.dedent(
        """
        import triton, triton.language as tl

        @triton.jit
        def swish_kernel(x_ptr):
            v = tl.sigmoid(tl.load(x_ptr))

        class ModelNew:
            def forward(self, x):
                return x
        """
    )
    assert lint_banned_apis(src, ["sigmoid", "silu"]) == []


def test_lint_reports_syntax_error_as_violation():
    violations = lint_banned_apis("def broken(:", ["softmax"])
    assert violations and violations[0].pattern == "<syntax>"


# ----------------------------------------- compile-time g4 gate (no GPU work)


def test_compile_rejects_banned_api_before_spawning_sandbox(tmp_path: Path):
    backend = _get_backend()
    spec = _softmax_spec()
    plan = Plan(
        interventions=(
            Intervention(
                target=Target("source_replace", "kernel_source"),
                payload={"module_source": _CHEATING_SOFTMAX},
            ),
        )
    )
    result = backend.compile(spec, plan, artifact_dir=tmp_path)
    assert not result.ok
    assert "banned-API lint failed" in (result.diagnostics or "")
    gates = result.metadata["gates"]
    assert gates["g4_banned_api"]["ok"] is False
    # No sandbox was spawned: no stdout/stderr logs were written.
    assert not (tmp_path / "sandbox_stdout.log").exists()
    # The cached evaluation also serves time_workload without re-running.
    timing = backend.time_workload(spec, plan, warmup=1, repetitions=1)
    assert timing.median_ms is None
    assert "lint" in (timing.diagnostics or "")


def test_compile_requires_source_replace_intervention(tmp_path: Path):
    backend = _get_backend()
    spec = _softmax_spec()
    plan = Plan(
        interventions=(
            Intervention(target=Target("source_replace", ""), payload={"module_source": "  "}),
        )
    )
    result = backend.compile(spec, plan, artifact_dir=tmp_path)
    assert not result.ok
    assert "source_replace" in (result.diagnostics or "")


def test_time_workload_without_compile_reports_cache_miss():
    backend = _get_backend()
    spec = _softmax_spec()
    timing = backend.time_workload(spec, Plan(), warmup=1, repetitions=1)
    assert timing.median_ms is None
    assert "compile()" in (timing.diagnostics or "")


def test_validate_correctness_served_from_compile_metadata():
    from compilagent.core.analysis import CompileResult
    from compilagent.core.workload import ToleranceConfig

    backend = _get_backend()
    spec = _softmax_spec()
    evaluation = {
        "gates": {
            "g1_shape_dtype": {"ok": True, "message": "ok"},
            "g2_allclose": {"ok": False, "message": "drifted"},
        },
        "max_abs_diff": 0.5,
        "max_rel_diff": 1.0,
    }
    candidate = CompileResult(ok=True, metadata={"evaluation": evaluation})
    baseline = CompileResult(ok=True)
    res = backend.validate_correctness(spec, baseline, candidate, ToleranceConfig())
    assert not res.ok
    assert res.failed_at == "g2_allclose"
    assert res.max_abs_diff == 0.5


# -------------------------------------------------------------- introspection


def test_read_reference_source_tool_serves_reference_and_banned_patterns():
    import json

    backend = _get_backend()
    tools = backend.list_introspection_tools()
    assert [t.name for t in tools] == ["read_reference_source"]
    decl = tools[0]
    assert decl.read_only
    payload = json.loads(decl.handler(workload_id="softmax_4096"))
    assert payload["workload_id"] == "softmax_4096"
    assert "torch.softmax" in payload["reference_module_source"]
    assert "softmax" in payload["banned_patterns"]
