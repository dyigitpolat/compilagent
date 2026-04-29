"""Triton compiler backend integration.

Self-registers a `TritonBackend` instance under the id `"triton"` at import
time. Importing this module *does not* import `triton` or `torch` itself —
those imports happen lazily inside the backend's methods so the package
loads on a CPU-only box.
"""

from __future__ import annotations

from compilagent.core.backend import backend_registry

from .backend import TritonBackend

if "triton" not in backend_registry.ids():
    backend_registry.register("triton", TritonBackend)

__all__ = ["TritonBackend"]
