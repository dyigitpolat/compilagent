"""Mamba (S6) selective state-space block — pure PyTorch reference forward.

Mamba uses a *selective* SSM (state-space model) where the dynamics matrices
A, B, Δ are conditioned on the input. The reference implementation in
`mamba-ssm` uses a custom CUDA kernel; here we write the pure-PyTorch
recurrence so Inductor sees the full op graph and can fuse the elementwise
discretisation, the cumulative scan, and the final projection.

The PyTorch reference is sequential (the inner loop is a Python for-range),
which gives Inductor a real challenge — the agent's wins typically come from
forcing the discretisation matmul + segment-sum patterns into a single
generated kernel via `epilogue_fusion` and FX rewrites.
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
    id="mamba_block",
    title="Mamba S6 selective-SSM block",
    description=(
        "PyTorch-reference S6 block: Conv1D depthwise + selective SSM "
        "(B/C/Δ projected from input) + gated SiLU + output projection. "
        "Inputs `[B=4, 512, 2048]` bf16. The PyTorch fallback is what "
        "torch.compile sees when mamba-ssm's custom kernel isn't available."
    ),
    kind=WorkloadKind.FULL_MODEL,
    backend_id="torch_inductor",
    entrypoint="compilagent_triton.workloads.pytorch_models.mamba_block:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=4,
        sequence_length=512,
        extra={"d_model": 2048, "d_state": 16, "d_conv": 4, "expand": 2},
    ),
    tolerance=ToleranceConfig(atol=5e-3, rtol=5e-2,
                              notes="SSM recurrence accumulates fp error; tolerance is loose."),
    budget=BenchmarkBudget(warmup=3, repetitions=10, max_seconds=120.0),
    metadata={"seed": 0},
)


@register_workload(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(int(spec.metadata.get("seed", 0)))

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        spec.dtype_policy.activation_dtype
    ]
    B = spec.shape_policy.batch_size or 4
    L = spec.shape_policy.sequence_length or 512
    d_model = int(spec.shape_policy.extra.get("d_model", 2048))
    d_state = int(spec.shape_policy.extra.get("d_state", 16))
    d_conv = int(spec.shape_policy.extra.get("d_conv", 4))
    expand = int(spec.shape_policy.extra.get("expand", 2))
    d_inner = d_model * expand

    class MambaBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, kernel_size=d_conv,
                padding=d_conv - 1, groups=d_inner, bias=True,
            )
            self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
            self.dt_proj = nn.Linear(1, d_inner, bias=True)
            # Continuous A initialised to a known stable spectrum.
            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(d_inner))
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        def forward(self, x):
            B_, L_, _ = x.shape
            xz = self.in_proj(x)                            # [B, L, 2*d_inner]
            x_, z = xz.chunk(2, dim=-1)
            # Depthwise conv along the sequence axis.
            x_ = self.conv1d(x_.transpose(1, 2))[:, :, :L_].transpose(1, 2)
            x_ = F.silu(x_)
            # Project to (B-coef, C-coef, Δ) per timestep, then run discretised SSM.
            xp = self.x_proj(x_)
            B_c, C_c, dt = xp.split([d_state, d_state, 1], dim=-1)
            dt = F.softplus(self.dt_proj(dt))               # [B, L, d_inner]
            A = -torch.exp(self.A_log.float())              # [d_inner, d_state]
            # Discretise: ΔA = exp(Δ ⊗ A), ΔB = Δ ⊗ B (zero-order hold).
            dA = torch.exp(dt.float().unsqueeze(-1) * A)    # [B, L, d_inner, d_state]
            dB = dt.float().unsqueeze(-1) * B_c.float().unsqueeze(2)  # broadcast
            # Sequential recurrence — no parallel scan available without a custom kernel.
            h = torch.zeros(B_, d_inner, d_state, device=x.device, dtype=torch.float32)
            ys = []
            x_f = x_.float()
            C_f = C_c.float()
            for t in range(L_):
                h = dA[:, t] * h + dB[:, t] * x_f[:, t].unsqueeze(-1)
                y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)   # [B, d_inner]
                ys.append(y_t)
            y = torch.stack(ys, dim=1).to(x.dtype) + x_ * self.D
            y = y * F.silu(z)
            return self.out_proj(y)

    block = MambaBlock().to(device="cuda", dtype=dtype).eval()
    inputs = (torch.randn(B, L, d_model, device="cuda", dtype=dtype),)

    def forward():
        with torch.no_grad():
            return block(inputs[0])

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=inputs,
        metadata={"module": block,
                  "param_count": sum(p.numel() for p in block.parameters())},
    )
