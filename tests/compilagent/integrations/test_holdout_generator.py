"""CPU tests for the D10 unseen-config holdout generator.

The generator is pure shape algebra over `spec.metadata["holdout"]` — no
torch, no GPU. Coverage is asserted across EVERY registered triton_source
workload (the 6 originals + the 24 KernelBench adaptations), plus targeted
checks on a synthetic tied-axes matmul spec.
"""

from __future__ import annotations

import ast

import pytest

from compilagent.core.workload import (
    ToleranceConfig,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.integrations.triton_source.holdout import (
    CATEGORIES,
    apply_to_reference,
    generate_holdout_configs,
)


def _spec(metadata: dict) -> WorkloadSpec:
    return WorkloadSpec(
        id="synthetic",
        title="synthetic",
        description="synthetic",
        kind=WorkloadKind.KERNEL,
        backend_id="triton_source",
        tolerance=ToleranceConfig(atol=1e-4, rtol=1e-3),
        metadata=metadata,
    )


_MATMUL_HOLDOUT = {
    "inputs": [
        {"name": "a", "factory": "randn", "dims": ["m", "k"]},
        {"name": "b", "factory": "randn", "dims": ["k", "n"]},
    ],
    "vars": {
        "m": {"size": 1024, "free": True},
        "k": {"size": 1024, "free": True},
        "n": {"size": 1024, "free": True},
    },
}

_LAYERNORM_HOLDOUT = {
    "inputs": [{"name": "x", "factory": "randn", "dims": ["batch", "hidden"]}],
    "vars": {
        "batch": {"size": 2048, "free": True},
        "hidden": {"size": 1024, "free": False},
    },
}


def _is_prime(x: int) -> bool:
    if x < 2:
        return False
    return all(x % f for f in range(2, int(x**0.5) + 1))


# ------------------------------------------------------------ category algebra


def test_generator_emits_one_config_per_category():
    configs = generate_holdout_configs(_spec({"holdout": _MATMUL_HOLDOUT}))
    assert [c.category for c in configs] == list(CATEGORIES)
    assert len(configs) == 6


def test_tied_axes_stay_tied_in_every_config():
    configs = generate_holdout_configs(_spec({"holdout": _MATMUL_HOLDOUT}))
    for config in configs:
        a, b = config.input_shapes["a"], config.input_shapes["b"]
        assert a[1] == b[0], f"{config.category}: K untied ({a} vs {b})"
        assert all(d >= 1 for d in a + b)


def test_fixed_axes_never_move():
    configs = generate_holdout_configs(_spec({"holdout": _LAYERNORM_HOLDOUT}))
    for config in configs:
        assert config.input_shapes["x"][1] == 1024, config.category
        assert config.sizes["hidden"] == 1024


def test_every_config_differs_from_the_seen_shape():
    for holdout in (_MATMUL_HOLDOUT, _LAYERNORM_HOLDOUT):
        seen = {
            entry["name"]: [holdout["vars"][d]["size"] for d in entry["dims"]]
            for entry in holdout["inputs"]
        }
        for config in generate_holdout_configs(_spec({"holdout": holdout})):
            assert config.input_shapes != seen, config.category


def test_edge_boundary_drives_leading_axis_to_one():
    configs = {
        c.category: c
        for c in generate_holdout_configs(_spec({"holdout": _LAYERNORM_HOLDOUT}))
    }
    assert configs["edge_boundary"].input_shapes["x"][0] == 1


def test_scale_directions_and_alignment_stress():
    configs = {
        c.category: c
        for c in generate_holdout_configs(_spec({"holdout": _MATMUL_HOLDOUT}))
    }
    assert configs["scale_up"].sizes["m"] == 2048
    assert configs["scale_down"].sizes["m"] == 256
    for var in ("m", "k", "n"):
        value = configs["alignment_stress"].sizes[var]
        assert _is_prime(value) and value > 1024
    asym = configs["asymmetric_aspect"].sizes
    assert asym["m"] > _MATMUL_HOLDOUT["vars"]["m"]["size"]
    assert asym["k"] < _MATMUL_HOLDOUT["vars"]["k"]["size"]
    assert configs["production_realistic"].sizes["m"] == 1000


def test_scale_up_respects_the_element_budget():
    huge = {
        "inputs": [{"name": "x", "factory": "rand", "dims": ["rows", "cols"]}],
        "vars": {
            "rows": {"size": 4096, "free": True},
            "cols": {"size": 393216, "free": False},
        },
    }
    configs = {
        c.category: c for c in generate_holdout_configs(_spec({"holdout": huge}))
    }
    seen_elements = 4096 * 393216
    up = configs["scale_up"].sizes["rows"] * 393216
    assert seen_elements < up <= 2 * seen_elements


def test_missing_or_degenerate_template_raises():
    with pytest.raises(ValueError, match="no holdout shape template"):
        generate_holdout_configs(_spec({}))
    all_fixed = {
        "inputs": [{"name": "x", "factory": "rand", "dims": ["d"]}],
        "vars": {"d": {"size": 8, "free": False}},
    }
    with pytest.raises(ValueError, match="no free axes"):
        generate_holdout_configs(_spec({"holdout": all_fixed}))


# ------------------------------------------------------------ source emission


def test_get_inputs_override_is_valid_python_with_cuda_fp32_tensors():
    configs = generate_holdout_configs(_spec({"holdout": _MATMUL_HOLDOUT}))
    for config in configs:
        tree = ast.parse(config.get_inputs_source)
        assert isinstance(tree.body[0], ast.FunctionDef)
        assert tree.body[0].name == "get_inputs"
        assert config.get_inputs_source.count("torch.randn(") == 2
        assert "device='cuda'" in config.get_inputs_source
        assert "dtype=torch.float32" in config.get_inputs_source


def test_apply_to_reference_appends_a_shadowing_get_inputs():
    reference = "import torch\n\ndef get_inputs():\n    return [torch.randn(2, 2)]\n"
    config = generate_holdout_configs(_spec({"holdout": _LAYERNORM_HOLDOUT}))[0]
    combined = apply_to_reference(reference, config)
    tree = ast.parse(combined)
    get_inputs_defs = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_inputs"
    ]
    assert len(get_inputs_defs) == 2  # original + override; the LAST one wins
    assert config.category in combined


# ------------------------------------------------- coverage over real workloads


def test_every_registered_triton_source_workload_generates_six_configs():
    import compilagent.integrations.triton_source  # noqa: F401
    from compilagent.core.workload_registry import workload_registry
    from compilagent.integrations.triton_source.kernelbench_workloads import (
        KB_WORKLOAD_IDS,
    )
    from compilagent.integrations.triton_source.workloads import WORKLOAD_IDS

    all_ids = [*WORKLOAD_IDS, *KB_WORKLOAD_IDS]
    assert len(all_ids) == 30
    for workload_id in all_ids:
        spec = workload_registry.get_spec(workload_id)
        configs = generate_holdout_configs(spec)
        assert [c.category for c in configs] == list(CATEGORIES), workload_id
        template = spec.metadata["holdout"]
        for config in configs:
            for name, value in config.sizes.items():
                var = template["vars"][name]
                assert value >= 1
                if not var.get("free"):
                    assert value == var["size"], (workload_id, config.category)
            ast.parse(config.get_inputs_source)
            combined = apply_to_reference(
                spec.metadata["reference_module_source"], config
            )
            ast.parse(combined)
