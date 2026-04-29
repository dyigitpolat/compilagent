"""Verify each integration's bundled example workloads register cleanly.

The tests assert that *spec registration* survives even on a CPU-only box —
the build functions lazy-import their heavy deps, so the import-time side
effects must succeed without `torch`/`triton`/`torchvision` being usable.
"""

from __future__ import annotations

from compilagent.core.workload_registry import workload_registry


def test_triton_integration_registers_example_workloads():
    import compilagent.integrations.triton  # noqa: F401

    ids = workload_registry.ids()
    assert "vector_add" in ids
    assert "vector_copy" in ids

    spec = workload_registry.get_spec("vector_add")
    assert spec.backend_id == "triton"
    assert spec.kind.value == "kernel"
    assert spec.metadata.get("kernel_symbol") == "vector_add_kernel"


def test_torch_inductor_integration_registers_example_workloads():
    import compilagent.integrations.torch_inductor  # noqa: F401

    assert "vit_block" in workload_registry.ids()
    spec = workload_registry.get_spec("vit_block")
    assert spec.backend_id == "torch_inductor"
    assert spec.kind.value == "full_model"
    assert spec.shape_policy.batch_size == 32


def test_register_workload_safely_is_idempotent():
    """Re-importing the integration must not raise on duplicate ids."""

    import compilagent.integrations.triton  # noqa: F401

    pre = workload_registry.ids()
    # Re-trigger the side effects.
    import importlib

    importlib.reload(__import__("compilagent.integrations.triton.examples", fromlist=["x"]))
    post = workload_registry.ids()
    assert pre == post  # no churn, no exceptions


def test_workload_source_endpoint_works_for_example_workload(tmp_path):
    """The observation UI's `/api/workloads/{id}/source` reads the
    integration's example module."""

    from fastapi.testclient import TestClient

    import compilagent.integrations.triton  # noqa: F401
    from compilagent.integrations.observation_ui.app import create_app
    from compilagent.settings import CompilagentSettings

    settings = CompilagentSettings(model_name="test", harness="nop")
    app = create_app(workspace_root=tmp_path, settings=settings)
    with TestClient(app) as client:
        r = client.get("/api/workloads/vector_add/source")
        assert r.status_code == 200
        body = r.json()
        assert body["language"] == "python"
        assert "vector_add_kernel" in body["source"]
