"""Six KernelBench-L1-style workloads for the `triton_source` backend (E3).

Each workload is a self-contained reference module (a `Model` nn.Module with
a no-arg `__init__` plus a `get_inputs()` factory returning CUDA tensors),
a banned-API pattern list for lint gate g4, and the input shapes/dtypes the
backend's cheap `analyze()` records. Registered with
`register_workload_safely` (the integration-example idiom) so test-harness
module reloads never trip duplicate-id errors.

These are hand-written L1-style tasks, NOT imports from the KernelBench
repo — the full KernelBench adapter is ticket E3-full / a later ticket.

The reference sources only touch torch/CUDA inside the sandbox subprocess,
so this module imports cleanly on CPU-only boxes.
"""

from __future__ import annotations

import textwrap

from compilagent.core.workload import (
    BenchmarkBudget,
    DtypePolicy,
    ShapePolicy,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.core.workload_registry import register_workload_safely

_BUDGET = BenchmarkBudget(warmup=25, repetitions=100, max_seconds=240.0)
_TOLERANCE = ToleranceConfig(atol=1e-4, rtol=1e-3, notes="fp32 E2a gate g2")


def _spec(
    *,
    workload_id: str,
    title: str,
    description: str,
    reference_source: str,
    banned_patterns: list[str],
    input_shapes: dict[str, list[int]],
    op_signature: str,
) -> WorkloadSpec:
    first_shape = next(iter(input_shapes.values()), [1])
    return WorkloadSpec(
        id=workload_id,
        title=title,
        description=description,
        kind=WorkloadKind.KERNEL,
        backend_id="triton_source",
        dtype_policy=DtypePolicy(activation_dtype="fp32", param_dtype="fp32"),
        shape_policy=ShapePolicy(
            batch_size=int(first_shape[0]),
            extra={"input_shapes": dict(input_shapes)},
        ),
        tolerance=_TOLERANCE,
        budget=_BUDGET,
        metadata={
            "reference_module_source": reference_source,
            "banned_patterns": list(banned_patterns),
            "input_shapes": dict(input_shapes),
            "input_dtypes": ["fp32"] * len(input_shapes),
            "op_signature": op_signature,
        },
    )


def _register(spec: WorkloadSpec) -> None:
    @register_workload_safely(spec)
    def _build(s: WorkloadSpec) -> WorkloadInstance:
        # The triton_source backend evaluates everything inside its
        # subprocess sandbox; the in-process forward is never invoked.
        return WorkloadInstance(
            spec=s,
            forward=lambda: None,
            example_inputs=(),
            metadata={"source_path": __file__},
        )


# ------------------------------------------------------------ 1. softmax_4096

_SOFTMAX_REF = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.softmax(x, dim=1)

    def get_inputs():
        return [torch.randn(4096, 4096, device='cuda', dtype=torch.float32)]
    """
).strip()

_register(
    _spec(
        workload_id="softmax_4096",
        title="Row-wise softmax 4096x4096",
        description=(
            "Row-wise softmax over dim=1 of a (4096, 4096) fp32 tensor."
        ),
        reference_source=_SOFTMAX_REF,
        banned_patterns=["torch.softmax", "softmax", "log_softmax", "Softmax"],
        input_shapes={"x": [4096, 4096]},
        op_signature="softmax(x: f32[4096,4096], dim=1) -> f32[4096,4096]",
    )
)

# ------------------------------------------------------ 2. layernorm_2048x1024

_LAYERNORM_REF = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(1024)

        def forward(self, x):
            return self.ln(x)

    def get_inputs():
        return [torch.randn(2048, 1024, device='cuda', dtype=torch.float32)]
    """
).strip()

_register(
    _spec(
        workload_id="layernorm_2048x1024",
        title="LayerNorm 2048x1024",
        description=(
            "LayerNorm over the last dim (1024) of a (2048, 1024) fp32 "
            "tensor, with learnable weight/bias (parameter names `ln.weight` "
            "/ `ln.bias`)."
        ),
        reference_source=_LAYERNORM_REF,
        banned_patterns=["layer_norm", "LayerNorm", "native_layer_norm"],
        input_shapes={"x": [2048, 1024]},
        op_signature="layer_norm(x: f32[2048,1024], normalized_shape=(1024,))",
    )
)

# -------------------------------------------------------- 3. matmul_relu_1024

_MATMUL_RELU_REF = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.relu(a @ b)

    def get_inputs():
        return [torch.randn(1024, 1024, device='cuda', dtype=torch.float32),
                torch.randn(1024, 1024, device='cuda', dtype=torch.float32)]
    """
).strip()

_register(
    _spec(
        workload_id="matmul_relu_1024",
        title="Fused matmul + ReLU 1024^3",
        description=(
            "Fused matrix multiply (1024x1024 @ 1024x1024, fp32) followed "
            "by ReLU."
        ),
        reference_source=_MATMUL_RELU_REF,
        banned_patterns=["matmul", "mm", "bmm", "einsum", "@"],
        input_shapes={"a": [1024, 1024], "b": [1024, 1024]},
        op_signature="relu(a: f32[1024,1024] @ b: f32[1024,1024])",
    )
)

# ---------------------------------------------------------- 4. gelu_8192x1024

_GELU_REF = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.gelu(x, approximate='tanh')

    def get_inputs():
        return [torch.randn(8192, 1024, device='cuda', dtype=torch.float32)]
    """
).strip()

_register(
    _spec(
        workload_id="gelu_8192x1024",
        title="GELU (tanh approx) 8192x1024",
        description=(
            "Elementwise GELU with the tanh approximation over a "
            "(8192, 1024) fp32 tensor."
        ),
        reference_source=_GELU_REF,
        banned_patterns=["gelu", "GELU"],
        input_shapes={"x": [8192, 1024]},
        op_signature="gelu(x: f32[8192,1024], approximate='tanh')",
    )
)

# ---------------------------------------------------- 5. l2norm_rows_4096x1024

_L2NORM_REF = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x / torch.linalg.vector_norm(x, dim=1, keepdim=True)

    def get_inputs():
        return [torch.randn(4096, 1024, device='cuda', dtype=torch.float32)]
    """
).strip()

_register(
    _spec(
        workload_id="l2norm_rows_4096x1024",
        title="Row-wise L2 normalisation 4096x1024",
        description=(
            "Row-wise L2 normalisation x / ||x||_2 (per row, dim=1) of a "
            "(4096, 1024) fp32 tensor."
        ),
        reference_source=_L2NORM_REF,
        banned_patterns=["norm", "vector_norm", "normalize"],
        input_shapes={"x": [4096, 1024]},
        op_signature="x / l2_norm_rows(x: f32[4096,1024])",
    )
)

# --------------------------------------------- 6. fused_bias_swish_4096x1024

_BIAS_SWISH_REF = textwrap.dedent(
    """
    import torch, torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.randn(1024) * 0.02)

        def forward(self, x):
            return x * torch.sigmoid(x + self.bias)

    def get_inputs():
        return [torch.randn(4096, 1024, device='cuda', dtype=torch.float32)]
    """
).strip()

_register(
    _spec(
        workload_id="fused_bias_swish_4096x1024",
        title="Fused bias + swish 4096x1024",
        description=(
            "Fused x * sigmoid(x + bias) over a (4096, 1024) fp32 tensor "
            "with a learnable bias of shape (1024,) (parameter name `bias`)."
        ),
        reference_source=_BIAS_SWISH_REF,
        banned_patterns=["sigmoid", "silu", "SiLU", "Sigmoid"],
        input_shapes={"x": [4096, 1024]},
        op_signature="x * sigmoid(x + bias[1024]) over f32[4096,1024]",
    )
)

WORKLOAD_IDS: tuple[str, ...] = (
    "softmax_4096",
    "layernorm_2048x1024",
    "matmul_relu_1024",
    "gelu_8192x1024",
    "l2norm_rows_4096x1024",
    "fused_bias_swish_4096x1024",
)
