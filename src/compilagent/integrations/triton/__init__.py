"""Triton compiler backend integration.

Self-registers a `TritonBackend` instance under the id `"triton"` at import
time, and registers a small set of example workloads (`vector_add`,
`vector_copy`) for the observation UI to surface as drop-in demos.

Importing this module *does not* import `triton` or `torch` itself — those
imports happen lazily inside the backend methods and the example builders,
so the package loads on a CPU-only box (the run will fail at start time
with a clear error if the user picks one of these workloads without CUDA).
"""

from __future__ import annotations

from compilagent.core.backend import backend_registry

from .backend import TritonBackend

if "triton" not in backend_registry.ids():
    backend_registry.register("triton", TritonBackend)

# Side-effect import: the examples module registers its workload specs into
# the global `workload_registry`. Failures inside individual examples are
# tolerated by the examples package itself.
from . import examples  # noqa: E402, F401

__all__ = ["TritonBackend"]
