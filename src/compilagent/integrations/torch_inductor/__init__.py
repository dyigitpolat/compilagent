"""TorchInductor compiler backend integration.

Self-registers a `TorchInductorBackend` instance under id `"torch_inductor"`.
Lazy-imports `torch` so the package loads on a CPU-only box without torch
installed; only fails when `compile()` actually runs.
"""

from __future__ import annotations

from compilagent.core.backend import backend_registry

from .backend import TorchInductorBackend

if "torch_inductor" not in backend_registry.ids():
    backend_registry.register("torch_inductor", TorchInductorBackend)

__all__ = ["TorchInductorBackend"]
