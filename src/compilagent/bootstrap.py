"""Integration bootstrap.

Two ways to bring an integration online:

1. **Direct import.** `import compilagent.integrations.triton` (or any external
   package, e.g. `import compilagent_acme`). The package's `__init__.py` calls
   the appropriate registry's `.register(...)` at import time. Python's import
   machinery is the registration mechanism.

2. **Entry points.** A pip-installed third-party integration declares one of
   the supported entry-point groups in its `pyproject.toml`:

       [project.entry-points."compilagent.integrations"]
       acme = "compilagent_acme"

   `load_entry_point_integrations()` then imports every advertised module the
   first time anyone calls it. `OptimizationSession.__init__` calls it once
   per process, so a user who runs `pip install compilagent-acme` and then
   `compilagent.OptimizationSession(workload_id="my_workload", ...)` gets
   their out-of-tree backend wired in with zero glue code.

Recognized entry-point groups (any of these works, all do the same thing —
import the named module so its registration side effects run):

  - `compilagent.integrations`
  - `compilagent.backends`
  - `compilagent.harnesses`
  - `compilagent.workloads`

The core itself never imports `compilagent.integrations.*`; entry points and
explicit `import` are the only paths.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence

ENTRY_POINT_GROUPS: tuple[str, ...] = (
    "compilagent.integrations",
    "compilagent.backends",
    "compilagent.harnesses",
    "compilagent.workloads",
)

_entry_points_loaded: bool = False


def import_modules(names: Sequence[str]) -> None:
    """Import each dotted module path. Triggers self-registration side effects."""

    for name in names:
        if not name:
            continue
        importlib.import_module(name)


def load_entry_point_integrations(
    *,
    groups: Sequence[str] = ENTRY_POINT_GROUPS,
    force: bool = False,
) -> list[str]:
    """Import every module advertised under the given entry-point groups.

    Idempotent: subsequent calls return the empty list unless `force=True`.
    Failures importing one entry point do not abort the others; the failing
    module name is silently skipped (the user can debug by importing
    explicitly).

    Returns the list of module names that were imported on this call.
    """

    global _entry_points_loaded
    if _entry_points_loaded and not force:
        return []

    try:
        from importlib.metadata import entry_points
    except ImportError:
        _entry_points_loaded = True
        return []

    imported: list[str] = []
    for group in groups:
        try:
            eps = entry_points(group=group)
        except TypeError:
            # Python <3.10 returned a different shape; ignore on those.
            continue
        for ep in eps:
            module_name = ep.value.split(":")[0].strip()
            if not module_name:
                continue
            try:
                importlib.import_module(module_name)
                imported.append(module_name)
            except Exception:
                # An integration that fails to import should not break the
                # session. Users who care can import the module explicitly to
                # see the traceback.
                continue

    _entry_points_loaded = True
    return imported


def _reset_entry_point_cache() -> None:
    """Test helper: re-arm `load_entry_point_integrations`."""

    global _entry_points_loaded
    _entry_points_loaded = False
