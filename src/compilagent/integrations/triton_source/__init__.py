"""Triton kernel-SOURCE backend integration (ticket E1).

Self-registers a `TritonSourceBackend` under the id `"triton_source"` at
import time, and registers six KernelBench-L1-style example workloads
(`softmax_4096`, `layernorm_2048x1024`, ...) whose decision space is the
kernel source itself: the agent proposes a complete python module defining
`ModelNew` built on `@triton.jit` kernels, evaluated in a subprocess sandbox
with hardened correctness gates (E2a) and CUDA-event timing.

Importing this module does NOT import `torch` or `triton` — all GPU work
happens lazily inside the sandbox subprocess, so the package loads on
CPU-only boxes.
"""

from __future__ import annotations

from compilagent.core.backend import backend_registry

from .backend import TritonSourceBackend

if "triton_source" not in backend_registry.ids():
    backend_registry.register("triton_source", TritonSourceBackend)

# Side-effect import: registers the six example workload specs.
from . import workloads  # noqa: E402, F401

__all__ = ["TritonSourceBackend"]
