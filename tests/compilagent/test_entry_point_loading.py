"""Verify that out-of-tree integrations advertised via entry points are loaded.

Uses `unittest.mock.patch` on `importlib.metadata.entry_points` so we don't
need to actually `pip install` a fake distribution.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from compilagent import bootstrap
from compilagent.core.backend import backend_registry


@dataclass(frozen=True)
class _FakeEntryPoint:
    name: str
    group: str
    value: str


def _install_fake_module(module_name: str) -> None:
    """Install a synthetic module that registers a fake backend on import."""

    class _Backend:
        id = "fake_ep"
        artifact_stages: tuple[str, ...] = ()

    module = types.ModuleType(module_name)

    def _on_import() -> None:
        if "fake_ep" not in backend_registry.ids():
            backend_registry.register("fake_ep", _Backend)

    _on_import()
    sys.modules[module_name] = module


@pytest.fixture
def _fake_entry_points():
    fake = _FakeEntryPoint(
        name="fake_ep", group="compilagent.integrations", value="fake_ep_module"
    )

    def _entry_points(*, group: str):
        return [fake] if group == "compilagent.integrations" else []

    if "fake_ep_module" in sys.modules:
        del sys.modules["fake_ep_module"]
    bootstrap._reset_entry_point_cache()

    # `entry_points` is imported lazily inside the function — patch the
    # importlib.metadata symbol the function actually looks up.
    with (
        patch("compilagent.bootstrap.entry_points", _entry_points, create=True),
        patch("importlib.metadata.entry_points", _entry_points),
    ):
        yield fake

    bootstrap._reset_entry_point_cache()
    sys.modules.pop("fake_ep_module", None)


def test_load_entry_point_integrations_imports_advertised_modules(_fake_entry_points):
    # Pre-arm: the module isn't loaded yet, and the fake module's import
    # side effect is what registers the backend.
    sys.modules.pop("fake_ep_module", None)

    # Patch importlib.import_module so importing "fake_ep_module" triggers
    # the fake registration without actually finding a distribution.
    real_import = bootstrap.importlib.import_module

    def _import_module(name: str):
        if name == "fake_ep_module":
            _install_fake_module(name)
            return sys.modules[name]
        return real_import(name)

    with patch.object(bootstrap.importlib, "import_module", _import_module):
        loaded = bootstrap.load_entry_point_integrations()

    assert "fake_ep_module" in loaded
    assert "fake_ep" in backend_registry.ids()


def test_load_entry_point_integrations_is_idempotent(_fake_entry_points):
    real_import = bootstrap.importlib.import_module

    def _import_module(name: str):
        if name == "fake_ep_module":
            _install_fake_module(name)
            return sys.modules[name]
        return real_import(name)

    with patch.object(bootstrap.importlib, "import_module", _import_module):
        first = bootstrap.load_entry_point_integrations()
        second = bootstrap.load_entry_point_integrations()

    assert first  # first call imported something
    assert second == []  # second call is a no-op


def test_load_entry_point_integrations_swallows_failed_imports(_fake_entry_points):
    def _import_module(name: str):
        raise ImportError(f"simulated failure importing {name}")

    with patch.object(bootstrap.importlib, "import_module", _import_module):
        loaded = bootstrap.load_entry_point_integrations()

    assert loaded == []  # nothing imported, but no exception escaped
