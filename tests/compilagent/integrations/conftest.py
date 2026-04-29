"""Shared fixtures for integration tests.

After clearing the registries, re-import each loaded `compilagent.integrations.<x>`
module so its self-registration side effect runs again. This keeps each test
isolated without making integrations responsible for re-registering on every
import.
"""

from __future__ import annotations

import contextlib
import importlib
import sys

import pytest

from compilagent.core.backend import backend_registry
from compilagent.core.workload_registry import workload_registry
from compilagent.harness.registry import harness_registry


def _reload_loaded_integrations() -> None:
    """Re-execute every loaded `compilagent.integrations.*` module.

    Reloads the leaves first (depth-3 modules like
    `compilagent.integrations.triton.examples.vector_add`) so a parent
    package's top-level `from . import examples` re-runs the registration
    side effects of its submodules. Best-effort: a failing reload (e.g.
    because optional deps are missing) is tolerated and doesn't break
    tests for unrelated integrations.
    """

    targets = [
        name
        for name in list(sys.modules)
        if name.startswith("compilagent.integrations.") and name.count(".") >= 2
    ]
    targets.sort(key=lambda n: -n.count("."))  # leaves before parents
    for name in targets:
        mod = sys.modules.get(name)
        if mod is not None:
            with contextlib.suppress(Exception):
                importlib.reload(mod)


@pytest.fixture(autouse=True)
def _reset_registries():
    backend_registry.clear()
    workload_registry.clear()
    harness_registry.clear()
    _reload_loaded_integrations()
    yield
    backend_registry.clear()
    workload_registry.clear()
    harness_registry.clear()
