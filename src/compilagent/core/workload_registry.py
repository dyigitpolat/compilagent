"""Workload registry — explicit registration via decorator.

Workloads register themselves at integration import time:

    from compilagent.core.workload_registry import register_workload

    @register_workload(_VECTOR_ADD_SPEC)
    def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
        ...

Bringing an integration online is therefore `import
compilagent.integrations.<name>` — Python's import machinery is the
registration mechanism. The core itself does not walk filesystem trees.
"""

from __future__ import annotations

from collections.abc import Callable

from .workload import WorkloadBuilder, WorkloadInstance, WorkloadSpec


class WorkloadRegistry:
    """Process-wide map from workload id to (spec, builder)."""

    def __init__(self) -> None:
        self._specs: dict[str, WorkloadSpec] = {}
        self._builders: dict[str, WorkloadBuilder] = {}

    def register(self, spec: WorkloadSpec, builder: WorkloadBuilder) -> None:
        if spec.id in self._specs:
            raise ValueError(f"Workload `{spec.id}` is already registered.")
        self._specs[spec.id] = spec
        self._builders[spec.id] = builder

    def get_spec(self, workload_id: str) -> WorkloadSpec:
        if workload_id not in self._specs:
            known = sorted(self._specs.keys())
            raise KeyError(
                f"Unknown workload `{workload_id}`. Registered: {known or '(none)'}."
            )
        return self._specs[workload_id]

    def build(self, workload_id: str) -> WorkloadInstance:
        spec = self.get_spec(workload_id)
        return self._builders[workload_id](spec)

    def ids(self) -> list[str]:
        return sorted(self._specs.keys())

    def specs(self) -> list[WorkloadSpec]:
        return [self._specs[wid] for wid in self.ids()]

    def clear(self) -> None:
        self._specs.clear()
        self._builders.clear()


workload_registry = WorkloadRegistry()


def register_workload(
    spec: WorkloadSpec,
) -> Callable[[WorkloadBuilder], WorkloadBuilder]:
    """Decorator: register `spec` against the decorated builder."""

    def _decorator(builder: WorkloadBuilder) -> WorkloadBuilder:
        workload_registry.register(spec, builder)
        return builder

    return _decorator
