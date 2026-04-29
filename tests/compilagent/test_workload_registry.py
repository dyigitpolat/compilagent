from __future__ import annotations

import pytest

from compilagent.core.workload import WorkloadInstance, WorkloadKind, WorkloadSpec
from compilagent.core.workload_registry import (
    WorkloadRegistry,
    register_workload,
    workload_registry,
)


def _make_spec(workload_id: str = "demo") -> WorkloadSpec:
    return WorkloadSpec(
        id=workload_id,
        title="Demo workload",
        description="Test fixture",
        kind=WorkloadKind.KERNEL,
        backend_id="fake",
    )


def test_register_via_decorator_round_trip():
    spec = _make_spec()

    @register_workload(spec)
    def _builder(s: WorkloadSpec) -> WorkloadInstance:
        return WorkloadInstance(spec=s, forward=lambda: None, example_inputs=())

    assert workload_registry.ids() == ["demo"]
    assert workload_registry.get_spec("demo") is spec
    instance = workload_registry.build("demo")
    assert isinstance(instance, WorkloadInstance)
    assert instance.spec.id == "demo"


def test_register_duplicate_raises():
    registry = WorkloadRegistry()
    spec = _make_spec()
    registry.register(spec, lambda s: WorkloadInstance(spec=s, forward=lambda: None))
    with pytest.raises(ValueError):
        registry.register(spec, lambda s: WorkloadInstance(spec=s, forward=lambda: None))


def test_get_unknown_raises():
    registry = WorkloadRegistry()
    with pytest.raises(KeyError):
        registry.get_spec("missing")
