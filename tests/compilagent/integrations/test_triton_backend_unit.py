"""Unit tests for the Triton backend that don't require triton or CUDA."""

from __future__ import annotations

from pathlib import Path

import pytest

from compilagent.core.backend import Backend, backend_registry
from compilagent.core.plan import Intervention, Target
from compilagent.core.workload import WorkloadKind, WorkloadSpec


def test_self_registration_installs_triton_backend():
    import compilagent.integrations.triton  # noqa

    assert "triton" in backend_registry.ids()
    b = backend_registry.get("triton")
    assert isinstance(b, Backend)
    assert b.id == "triton"
    assert b.artifact_stages == ("ttir", "ttgir", "llir", "ptx")


def test_validate_intervention_accepts_pass_and_launch():
    import compilagent.integrations.triton  # noqa

    b = backend_registry.get("triton")
    assert b.validate_intervention(
        Intervention(target=Target("pass", "ttir:foo"), payload={"action": "skip"})
    ).ok
    assert b.validate_intervention(
        Intervention(target=Target("launch", "k"), payload={"BLOCK_SIZE": 128})
    ).ok


def test_validate_intervention_rejects_unknown_kind():
    import compilagent.integrations.triton  # noqa

    b = backend_registry.get("triton")
    result = b.validate_intervention(
        Intervention(target=Target("lowering", "x"), payload={})
    )
    assert not result.ok
    assert any("lowering" in e for e in result.errors)


def test_validate_intervention_rejects_bad_pass_action():
    import compilagent.integrations.triton  # noqa

    b = backend_registry.get("triton")
    result = b.validate_intervention(
        Intervention(target=Target("pass", "ttir:foo"), payload={"action": "explode"})
    )
    assert not result.ok


def test_renderers_advertise_expected_suffixes():
    from compilagent.integrations.triton.renderers import MlirRenderer, PtxRenderer

    assert ".ttgir" in MlirRenderer().suffixes
    assert ".ttir" in MlirRenderer().suffixes
    assert ".ptx" in PtxRenderer().suffixes


def test_renderers_render_files(tmp_path: Path):
    from compilagent.integrations.triton.renderers import MlirRenderer, PtxRenderer

    f = tmp_path / "demo.ttgir"
    f.write_text("module attributes {} { tt.func @kernel() { tt.return } }", encoding="utf-8")
    out = MlirRenderer().render(f)
    assert out.kind == "code"
    assert out.language == "mlir"
    assert "tt.func" in out.text

    f2 = tmp_path / "demo.ptx"
    f2.write_text(".version 7.5\n.target sm_90", encoding="utf-8")
    out2 = PtxRenderer().render(f2)
    assert out2.language == "ptx"


def test_introspection_tool_handlers_runnable():
    pytest.importorskip("triton", reason="triton import needed for the pass catalog")

    import compilagent.integrations.triton  # noqa
    from compilagent.integrations.triton.tools import list_triton_introspection_tools

    tools = list_triton_introspection_tools()
    names = [t.name for t in tools]
    assert "list_compiler_passes" in names
    assert "describe_compiler_pass" in names
    listing = next(t for t in tools if t.name == "list_compiler_passes").handler({})
    assert "passes" in listing


def test_infer_workload_family():
    import compilagent.integrations.triton  # noqa

    b = backend_registry.get("triton")
    spec = WorkloadSpec(
        id="x", title="matmul vit", description="", kind=WorkloadKind.KERNEL, backend_id="triton"
    )
    assert b.infer_workload_family(spec) == "matmul"
