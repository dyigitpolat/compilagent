"""Six Triton kernel candidates with launch args for `optimize_kernel`.

Each builder returns a dict ready to splat into
`compilagent.integrations.python.optimize_kernel(**spec)` (minus
`max_candidates`/`harness`/`model_id` which the runner controls).

Sizes target sub-second baselines so the suite finishes in a reasonable
wall-clock window even with an LLM-driven search loop on top.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import torch
import triton

_DEVICE = "cuda"


def _import_kernel(module_name: str, attr: str) -> Any:
    mod = importlib.import_module(f"scripts.kernels.{module_name}")
    return getattr(mod, attr)


# ----------------------------------------------------------- 1. fused softmax

def build_fused_softmax() -> dict[str, Any]:
    kernel = _import_kernel("fused_softmax", "fused_softmax_kernel")
    torch.manual_seed(0)
    n_rows, n_cols = 4096, 2048
    BLOCK = triton.next_power_of_2(n_cols)
    x = torch.randn(n_rows, n_cols, device=_DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    return {
        "kernel": kernel,
        "args": (x, out, n_rows, n_cols, x.stride(0)),
        "grid": lambda meta: (n_rows,),
        "constexpr": {"BLOCK_SIZE": BLOCK},
    }


# ----------------------------------------------------------------- 2. RMSNorm

def build_rmsnorm_kernel() -> dict[str, Any]:
    kernel = _import_kernel("rmsnorm_fwd", "rmsnorm_fwd_kernel")
    torch.manual_seed(0)
    rows, cols = 8192, 4096
    BLOCK = triton.next_power_of_2(cols)
    x = torch.randn(rows, cols, device=_DEVICE, dtype=torch.float32)
    w = torch.ones(cols, device=_DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    return {
        "kernel": kernel,
        "args": (x, w, out, cols, x.stride(0), 1e-6),
        "grid": lambda meta: (rows,),
        "constexpr": {"BLOCK_SIZE": BLOCK},
    }


# --------------------------------------------------------------- 3. LayerNorm

def build_layernorm_kernel() -> dict[str, Any]:
    kernel = _import_kernel("layernorm_fwd", "layernorm_fwd_kernel")
    torch.manual_seed(0)
    rows, cols = 8192, 4096
    BLOCK = triton.next_power_of_2(cols)
    x = torch.randn(rows, cols, device=_DEVICE, dtype=torch.float32)
    w = torch.ones(cols, device=_DEVICE, dtype=torch.float32)
    b = torch.zeros(cols, device=_DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    return {
        "kernel": kernel,
        "args": (x, w, b, out, cols, x.stride(0), 1e-5),
        "grid": lambda meta: (rows,),
        "constexpr": {"BLOCK_SIZE": BLOCK},
    }


# -------------------------------------------------------------------- 4. GELU

def build_gelu_kernel() -> dict[str, Any]:
    kernel = _import_kernel("gelu_fwd", "gelu_fwd_kernel")
    torch.manual_seed(0)
    n = 1 << 24
    x = torch.randn(n, device=_DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    BLOCK = 1024
    return {
        "kernel": kernel,
        "args": (x, out, n),
        "grid": lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),),
        "constexpr": {"BLOCK_SIZE": BLOCK},
    }


# ---------------------------------------------------------------- 5. Dropout

def build_dropout_kernel() -> dict[str, Any]:
    kernel = _import_kernel("dropout_fwd", "dropout_fwd_kernel")
    torch.manual_seed(0)
    n = 1 << 24
    x = torch.randn(n, device=_DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    BLOCK = 1024
    return {
        "kernel": kernel,
        "args": (x, out, n, 0.1, 1234),
        "grid": lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),),
        "constexpr": {"BLOCK_SIZE": BLOCK},
    }


# ----------------------------------------------------------------- 6. matmul

def build_matmul_kernel() -> dict[str, Any]:
    kernel = _import_kernel("matmul_fwd", "matmul_fwd_kernel")
    torch.manual_seed(0)
    M = N = K = 1024
    a = torch.randn(M, K, device=_DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=_DEVICE, dtype=torch.float16)
    c = torch.empty((M, N), device=_DEVICE, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    return {
        "kernel": kernel,
        "args": (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1)),
        "grid": lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"])),
        "constexpr": {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
    }


KERNEL_BUILDERS: dict[str, Callable[[], dict[str, Any]]] = {
    "fused_softmax": build_fused_softmax,
    "rmsnorm_kernel": build_rmsnorm_kernel,
    "layernorm_kernel": build_layernorm_kernel,
    "gelu_kernel": build_gelu_kernel,
    "dropout_kernel": build_dropout_kernel,
    "matmul_kernel": build_matmul_kernel,
}
