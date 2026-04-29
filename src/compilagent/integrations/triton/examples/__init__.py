"""Example workloads shipped with the Triton integration.

Each example registers itself via `register_workload_safely` so duplicate
imports (e.g. by the test harness reloading the integration) don't raise.
A failure to import any one example does not abort the others — the user
gets the rest in the UI plus a diagnostic surfaced through
`/api/workloads/diagnostics`.
"""

from __future__ import annotations

import contextlib
import importlib
import sys

# Each module body registers its own workload at import time via
# `register_workload_safely`. The reload branch makes the registration
# deterministic when the test harness reloads the parent integration
# package — without it, a stale-module entry in `sys.modules` would skip
# re-running the decorator side effects.
for _example in ("vector_add", "vector_copy"):
    _full = f"{__name__}.{_example}"
    with contextlib.suppress(Exception):
        if _full in sys.modules:
            importlib.reload(sys.modules[_full])
        else:
            importlib.import_module(_full)

del _example, _full  # noqa: F821 — defined once the loop ran
