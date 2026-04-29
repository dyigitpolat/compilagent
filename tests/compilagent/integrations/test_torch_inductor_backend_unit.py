"""Unit tests for the TorchInductor backend (no real torch.compile call)."""

from __future__ import annotations

from pathlib import Path

from compilagent.core.backend import Backend, backend_registry
from compilagent.core.plan import Intervention, Target
from compilagent.core.workload import WorkloadKind, WorkloadSpec


def test_self_registration_installs_inductor_backend():
    import compilagent.integrations.torch_inductor  # noqa

    assert "torch_inductor" in backend_registry.ids()
    b = backend_registry.get("torch_inductor")
    assert isinstance(b, Backend)
    assert b.id == "torch_inductor"


def test_validate_intervention_accepts_supported_kinds():
    import compilagent.integrations.torch_inductor  # noqa

    b = backend_registry.get("torch_inductor")
    for kind, sel, payload in (
        ("knob", "inductor.max_autotune", True),
        ("knob", "dynamo.suppress_errors", True),
        ("scheduler", "pre_fusion", lambda x: x),
        ("scheduler", "post_fusion", lambda x: x),
        ("lowering", "torch.ops.aten.add", "module:fn"),
        ("fx_node", "node_target", {"rewrite": "..."}),
        ("choices", "", {}),
    ):
        result = b.validate_intervention(
            Intervention(target=Target(kind, sel), payload=payload)
        )
        assert result.ok, (kind, sel, result.errors)


def test_validate_intervention_rejects_bad_scheduler_selector():
    import compilagent.integrations.torch_inductor  # noqa

    b = backend_registry.get("torch_inductor")
    result = b.validate_intervention(
        Intervention(target=Target("scheduler", "bogus"), payload=None)
    )
    assert not result.ok


def test_renderers_render_files(tmp_path: Path):
    from compilagent.integrations.torch_inductor.renderers import (
        FxGraphRenderer,
        OutputCodeRenderer,
        SchedulerLogRenderer,
    )

    f = tmp_path / "demo.fx_graph"
    f.write_text("def forward(x): return x\n", encoding="utf-8")
    p = FxGraphRenderer().render(f)
    assert p.language == "python"

    f = tmp_path / "demo.output_code"
    f.write_text("# generated\n", encoding="utf-8")
    p = OutputCodeRenderer().render(f)
    assert p.language == "python"

    f = tmp_path / "demo.fusion_log"
    f.write_text("scheduler step", encoding="utf-8")
    p = SchedulerLogRenderer().render(f)
    assert p.kind == "text"


def test_infer_workload_family():
    import compilagent.integrations.torch_inductor  # noqa

    b = backend_registry.get("torch_inductor")
    spec = WorkloadSpec(
        id="vit", title="ViT", description="", kind=WorkloadKind.FULL_MODEL,
        backend_id="torch_inductor",
    )
    assert b.infer_workload_family(spec) == "transformer"
