"""TorchInductor compiler backend integration.

Self-registers a `TorchInductorBackend` instance under id `"torch_inductor"`,
and registers an example workload (`vit_block`) for the observation UI to
surface as a drop-in demo.

Lazy-imports `torch` / `torchvision` so the package loads on a CPU-only box
without those libraries installed; runs fail at start time with a clear
error if the user picks the example without CUDA.
"""

from __future__ import annotations

from compilagent.core.backend import backend_registry

from .backend import TorchInductorBackend

if "torch_inductor" not in backend_registry.ids():
    backend_registry.register("torch_inductor", TorchInductorBackend)

# Side-effect import: the examples module registers its workload specs into
# the global `workload_registry`. Failures inside individual examples are
# tolerated by the examples package itself.
from . import examples  # noqa: E402, F401

__all__ = ["TorchInductorBackend"]
