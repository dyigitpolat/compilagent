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
    for name in list(sys.modules):
        if name.startswith("compilagent.integrations.") and name.count(".") == 2:
            mod = sys.modules.get(name)
            if mod is not None:
                # Best-effort: an integration that fails to reload (e.g.
                # because optional deps are missing) shouldn't break tests
                # for unrelated integrations.
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
