"""ResNet-50 bottleneck residual block.

The canonical 1×1 → 3×3 → 1×1 bottleneck with BN + ReLU + skip add. Inductor
compiles this into a chain of fused conv-bn-relu kernels; the workload exposes
levers around `conv_layout`, `epilogue_fusion`, `aggressive_fusion`, and
matmul-template choice for the 1×1 convs (which Inductor can lower as
`addmm`-style ops on Blackwell).
"""

from __future__ import annotations

from ...core.workload import (
    BenchmarkBudget,
    DtypePolicy,
    ShapePolicy,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from ..registry import register_workload


_SPEC = WorkloadSpec(
    id="resnet50_bottleneck",
    title="ResNet-50 bottleneck residual block",
    description=(
        "Standard ResNet bottleneck (1x1 conv → 3x3 conv → 1x1 conv, BN+ReLU "
        "after each, residual add). Inputs `[B=64, 256, 56, 56]` bf16; expansion "
        "factor 4. Exercises Inductor conv lowering + BN fusion + skip-add."
    ),
    kind=WorkloadKind.FULL_MODEL,
    backend_id="torch_inductor",
    entrypoint="compilagent_triton.workloads.pytorch_models.resnet50_bottleneck:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=64,
        image_size=(56, 56),
        extra={"in_planes": 256, "planes": 64, "expansion": 4},
    ),
    tolerance=ToleranceConfig(atol=5e-4, rtol=5e-3),
    budget=BenchmarkBudget(warmup=3, repetitions=20, max_seconds=60.0),
    metadata={"seed": 0},
)


@register_workload(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(int(spec.metadata.get("seed", 0)))

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        spec.dtype_policy.activation_dtype
    ]
    in_planes = int(spec.shape_policy.extra.get("in_planes", 256))
    planes = int(spec.shape_policy.extra.get("planes", 64))
    expansion = int(spec.shape_policy.extra.get("expansion", 4))
    out_planes = planes * expansion
    H, W = spec.shape_policy.image_size or (56, 56)
    B = spec.shape_policy.batch_size or 1

    class Bottleneck(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, out_planes, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes)
            self.skip = (
                nn.Identity() if in_planes == out_planes else
                nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=False),
                              nn.BatchNorm2d(out_planes))
            )

        def forward(self, x):
            identity = self.skip(x)
            out = torch.relu(self.bn1(self.conv1(x)))
            out = torch.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            return torch.relu(out + identity)

    block = Bottleneck().to(device="cuda", dtype=dtype).eval()
    inputs = (torch.randn(B, in_planes, H, W, device="cuda", dtype=dtype),)

    def forward():
        with torch.no_grad():
            return block(inputs[0])

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=inputs,
        metadata={"module": block,
                  "param_count": sum(p.numel() for p in block.parameters())},
    )
