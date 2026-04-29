"""Example workloads shipped with the torch_inductor integration.

Each example registers itself via `register_workload_safely` so duplicate
imports don't raise. A failure to import any one example does not abort
the others — the user gets the rest in the UI plus a diagnostic surfaced
through `/api/workloads/diagnostics`.
"""

from __future__ import annotations

import contextlib
import importlib
import sys

for _example in ("vit_block",):
    _full = f"{__name__}.{_example}"
    with contextlib.suppress(Exception):
        if _full in sys.modules:
            importlib.reload(sys.modules[_full])
        else:
            importlib.import_module(_full)

del _example, _full  # noqa: F821 — defined once the loop ran
