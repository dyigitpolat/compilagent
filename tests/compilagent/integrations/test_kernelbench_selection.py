"""CPU tests for the KernelBench 24-task selection logic (ticket D8).

Everything GPU-flavored is mocked or pure: contamination classification runs
on hand-built probe stats, conversion is exercised by exec-ing the adapted
module on CPU (construction only — `get_inputs` is never called), and the
committed manifest is checked through the registered workload specs.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kernelbench_select import (  # noqa: E402
    KbTask,
    TemplateError,
    adapt_reference_source,
    contamination_reasons,
    conv_exclusion,
    derive_banned_patterns,
    inefficient_exclusion,
    op_family,
    parse_probe_lines,
    parse_shape_template,
    stratified_select,
)

_GEMM_SOURCE = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, in_features, out_features):
            super(Model, self).__init__()
            self.gemm = nn.Linear(in_features, out_features)

        def forward(self, x):
            return torch.relu(self.gemm(x))

    batch_size = 1024
    in_features = 8192
    out_features = 4096

    def get_inputs():
        return [torch.rand(batch_size, in_features)]

    def get_init_inputs():
        return [in_features, out_features]
    """
).strip()


def _task(level: int, index: int, name: str, source: str) -> KbTask:
    return KbTask(
        level=level,
        index=index,
        name=name,
        path=Path(f"/kb/level{level}/{index}_{name}.py"),
        source=source,
    )


# ------------------------------------------------- contamination classification


def _clean_stats(**overrides):
    stats = {
        "out_abs_max": 3.5,
        "out_std": 1.0,
        "input_impact": 2.0,
        "near_identity": False,
    }
    stats.update(overrides)
    return stats


def test_clean_probe_stats_yield_no_contamination_reasons():
    assert contamination_reasons(_clean_stats()) == []


def test_outputs_inside_pm_001_band_are_contaminated():
    reasons = contamination_reasons(_clean_stats(out_abs_max=0.009))
    assert len(reasons) == 1
    assert "[-0.01, 0.01]" in reasons[0]


def test_low_output_std_is_contaminated():
    reasons = contamination_reasons(_clean_stats(out_std=0.001))
    assert any("std" in r for r in reasons)


def test_low_input_impact_is_contaminated():
    reasons = contamination_reasons(_clean_stats(input_impact=0.0001))
    assert any("input-impact" in r for r in reasons)


def test_near_identity_reference_is_contaminated():
    reasons = contamination_reasons(_clean_stats(near_identity=True))
    assert any("near-identity" in r for r in reasons)


def test_multiple_criteria_all_reported():
    reasons = contamination_reasons(
        _clean_stats(out_abs_max=0.001, out_std=0.0001, input_impact=0.0)
    )
    assert len(reasons) == 3


# --------------------------------------------------------------- static checks


def test_conv_tasks_are_excluded_by_name_and_source():
    assert conv_exclusion(_task(2, 1, "Conv2D_ReLU_BiasAdd", "x")) is not None
    by_source = _task(1, 50, "mystery", "self.conv = nn.Conv2d(3, 8, 3)")
    assert conv_exclusion(by_source) is not None
    assert conv_exclusion(_task(2, 12, "Gemm_Multiply_LeakyReLU", _GEMM_SOURCE)) is None


def test_diag_tril_triu_baselines_are_inherently_inefficient():
    for snippet in ("torch.diag(A) @ B", "torch.tril(torch.matmul(A, B))", "A.triu()"):
        task = _task(1, 12, "Matmul_structured", f"def f(A, B):\n    return {snippet}\n")
        reason = inefficient_exclusion(task)
        assert reason is not None and "inherently_inefficient" in reason
    assert inefficient_exclusion(_task(1, 1, "Square", _GEMM_SOURCE)) is None


# --------------------------------------------------------- shape-template parse


def test_template_marks_init_bound_axes_fixed_and_batch_free():
    template = parse_shape_template(_GEMM_SOURCE)
    assert [e["name"] for e in template.inputs] == ["x"]
    assert template.inputs[0]["factory"] == "rand"
    assert template.inputs[0]["dims"] == ["batch_size", "in_features"]
    assert template.vars["batch_size"] == {"size": 1024, "free": True}
    assert template.vars["in_features"] == {"size": 8192, "free": False}
    assert template.input_shapes() == {"x": [1024, 8192]}


def test_template_ties_shared_axes_through_one_var():
    source = textwrap.dedent(
        """
        import torch, torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, A, B):
                return torch.matmul(A, B)

        M = 1024
        K = 4096 * 2
        N = 512

        def get_inputs():
            A = torch.randn(M, K)
            B = torch.randn(K, N)
            return [A, B]

        def get_init_inputs():
            return []
        """
    ).strip()
    template = parse_shape_template(source)
    assert template.inputs[0]["dims"] == ["M", "K"]
    assert template.inputs[1]["dims"] == ["K", "N"]
    assert template.vars["K"] == {"size": 8192, "free": True}  # constant folded
    assert all(v["free"] for v in template.vars.values())


def test_template_expands_starred_tuple_shapes():
    source = textwrap.dedent(
        """
        import torch, torch.nn as nn

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.cumsum(x, dim=self.dim)

        batch_size = 32768
        input_shape = (4096,)
        dim = 1

        def get_inputs():
            return [torch.rand(batch_size, *input_shape)]

        def get_init_inputs():
            return [dim]
        """
    ).strip()
    template = parse_shape_template(source)
    assert template.inputs[0]["dims"] == ["batch_size", "input_shape_0"]
    assert template.vars["input_shape_0"] == {"size": 4096, "free": True}


def test_unparameterizable_inputs_raise_template_error():
    scalar = _GEMM_SOURCE.replace(
        "return [torch.rand(batch_size, in_features)]",
        "return [torch.rand(batch_size, in_features), 3.14]",
    )
    mask = _GEMM_SOURCE.replace(
        "return [torch.rand(batch_size, in_features)]",
        "return [torch.randint(0, 2, (batch_size, in_features)).bool()]",
    )
    symmetrized = _GEMM_SOURCE.replace(
        "return [torch.rand(batch_size, in_features)]",
        "A = torch.rand(batch_size, batch_size)\n"
        "    A = (A + A.T) / 2\n"
        "    return [A]",
    )
    for source in (scalar, mask, symmetrized):
        try:
            parse_shape_template(source)
        except TemplateError:
            continue
        raise AssertionError("expected TemplateError")


# -------------------------------------------------------------------- conversion


def test_adapted_reference_builds_a_no_arg_model_with_unprefixed_params():
    import pytest

    torch = pytest.importorskip("torch")

    adapted = adapt_reference_source(_GEMM_SOURCE, header="test")
    namespace: dict = {}
    exec(compile(adapted, "<adapted>", "exec"), namespace)  # noqa: S102
    torch.manual_seed(0)
    # `super(Model, self).__init__()` in the original class must not recurse
    # after the rename — construction succeeding proves it.
    model = namespace["Model"]()
    assert sorted(model.state_dict()) == ["gemm.bias", "gemm.weight"]
    assert tuple(model.state_dict()["gemm.weight"].shape) == (4096, 8192)
    assert "_KBModel" in adapted
    assert "def get_inputs():" in adapted  # CUDA/fp32 wrapper shim


def test_adapted_reference_is_valid_python_with_shadowing_definitions():
    adapted = adapt_reference_source(_GEMM_SOURCE)
    tree = ast.parse(adapted)
    class_defs = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert class_defs == ["_KBModel", "Model"]
    fn_defs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert fn_defs.index("_kb_get_inputs") < fn_defs.index("get_inputs")


# --------------------------------------------------------------- banned patterns


def test_banned_patterns_cover_every_known_op_token():
    gemm = derive_banned_patterns(_task(2, 12, "Gemm_Multiply_LeakyReLU", ""))
    assert {"matmul", "mm", "@", "linear", "Linear", "leaky_relu"} <= set(gemm)
    softmax = derive_banned_patterns(_task(1, 23, "Softmax", ""))
    assert "softmax" in softmax and "log_softmax" in softmax
    gpt_gelu = derive_banned_patterns(_task(1, 88, "MinGPTNewGelu", ""))
    assert "gelu" in gpt_gelu and "tanh" in gpt_gelu
    assert "matmul" in derive_banned_patterns(
        _task(1, 9, "Tall_skinny_matrix_multiplication", "")
    )


# ---------------------------------------------------------- stratified selection


def test_stratified_select_round_robins_families_deterministically():
    tasks = [
        _task(1, i, name, "")
        for i, name in [
            (1, "Square_matrix_multiplication"),
            (2, "Standard_matrix_multiplication"),
            (3, "Batched_matrix_multiplication"),
            (19, "ReLU"),
            (21, "Sigmoid"),
            (36, "RMSNorm"),
            (40, "LayerNorm"),
            (89, "cumsum"),
        ]
    ]
    selected = stratified_select(tasks, 6)
    assert len(selected) == 6
    families = [op_family(t) for t in selected]
    # Every family is covered before any family gets its second pick.
    assert set(families) == {"matmul", "activation", "norm", "scan"}
    assert selected == stratified_select(list(reversed(tasks)), 6)  # deterministic
    # Within a family, ascending KB index wins the first slot.
    matmul_picks = [t.index for t in selected if op_family(t) == "matmul"]
    assert matmul_picks == sorted(matmul_picks)
    assert matmul_picks[0] == 1


def test_stratified_select_respects_quota():
    tasks = [_task(1, i, "ReLU", "") for i in range(1, 30)]
    assert len(stratified_select(tasks, 12)) == 12


# ------------------------------------------------------------- probe-line parse


def test_parse_probe_lines_recovers_marker_rows_and_ignores_noise():
    stdout = "\n".join(
        [
            "some torch warning",
            'KB_PROBE_JSON:{"id": "kb1_19_relu", "ok": true, "out_std": 0.5}',
            "KB_PROBE_JSON:not json",
            'KB_PROBE_JSON:{"id": "kb1_21_sigmoid", "ok": false, "error": "boom"}',
        ]
    )
    parsed = parse_probe_lines(stdout)
    assert set(parsed) == {"kb1_19_relu", "kb1_21_sigmoid"}
    assert parsed["kb1_19_relu"]["ok"] is True
    assert parsed["kb1_21_sigmoid"]["error"] == "boom"


# -------------------------------------------------- manifest-backed registration


def test_manifest_registers_24_kernelbench_workloads():
    import compilagent.integrations.triton_source  # noqa: F401
    from compilagent.core.workload_registry import workload_registry
    from compilagent.integrations.triton_source.kernelbench_workloads import (
        KB_WORKLOAD_IDS,
    )

    assert len(KB_WORKLOAD_IDS) == 24
    level1 = [i for i in KB_WORKLOAD_IDS if i.startswith("kb1_")]
    level2 = [i for i in KB_WORKLOAD_IDS if i.startswith("kb2_")]
    assert len(level1) == 12 and len(level2) == 12

    registered = workload_registry.ids()
    for workload_id in KB_WORKLOAD_IDS:
        assert workload_id in registered
        spec = workload_registry.get_spec(workload_id)
        assert spec.backend_id == "triton_source"
        assert spec.dtype_policy.activation_dtype == "fp32"
        source = spec.metadata["reference_module_source"]
        assert "class Model(_KBModel):" in source
        ast.parse(source)  # adapted module must be valid python
        assert spec.metadata["banned_patterns"], workload_id
        assert spec.metadata["input_shapes"], workload_id
        assert spec.metadata["holdout"]["inputs"], workload_id
        assert spec.metadata["kernelbench"]["kb_file"].startswith("level")
