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

import inspect
from collections.abc import Callable
from typing import Any

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

    def get_builder(self, workload_id: str) -> WorkloadBuilder:
        """Return the registered builder callable (does not call it)."""

        self.get_spec(workload_id)
        return self._builders[workload_id]

    def get_builder_source(self, workload_id: str) -> dict[str, Any]:
        """Return `{language, source_path, source}` for the workload's builder.

        Reads the entire module the builder was defined in so the UI can
        show the workload context (imports, helpers, the spec literal),
        not just the `def build_workload(...)` body. Returns `source=""`
        and `source_path=None` if the builder's source can't be located
        (e.g. defined in a REPL or a closure with no `__file__`).
        """

        builder = self.get_builder(workload_id)
        try:
            source_path = inspect.getsourcefile(builder)
        except TypeError:
            source_path = None
        source = ""
        if source_path:
            try:
                source = inspect.getsource(inspect.getmodule(builder))
            except (OSError, TypeError):
                source = ""
        return {
            "workload_id": workload_id,
            "language": "python",
            "source_path": source_path,
            "source": source,
            "line_count": source.count("\n") + 1 if source else 0,
        }

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


def register_workload_safely(
    spec: WorkloadSpec,
) -> Callable[[WorkloadBuilder], WorkloadBuilder]:
    """Idempotent variant of `register_workload`.

    Designed for example workloads shipped by integrations: a duplicate
    registration (e.g. when the test harness reloads the integration
    module to reset registry state) is silently ignored instead of raising.
    Use this when a workload registration should not fight module-reimport
    semantics. Behaviour is otherwise identical to `register_workload`.
    """

    def _decorator(builder: WorkloadBuilder) -> WorkloadBuilder:
        if spec.id not in workload_registry.ids():
            workload_registry.register(spec, builder)
        return builder

    return _decorator
