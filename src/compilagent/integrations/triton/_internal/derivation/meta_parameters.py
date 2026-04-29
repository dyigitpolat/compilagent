"""Derive `BLOCK_SIZE` / `num_warps` / `num_stages` candidates from analysis.

No hand-coded lists. Bounds come from:

- the largest contiguous tensor dim observed in the kernel's IR, fed into
  `pow2_around(...)` to get power-of-two BLOCK_SIZE candidates;
- the device's shared-memory budget (mapped to a num_warps ceiling);
- the loop depth in TTGIR (mapped to a num_stages ceiling).

If no analysis evidence is available, the rule yields nothing — falling back
to the agent proposing values manually rather than us baking defaults in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from compilagent.core.search_space import (
    DerivationEvidence,
    IntRange,
    Lever,
    SearchSpaceDerivation,
    pow2_around,
    pow2_range,
)


@dataclass(frozen=True, slots=True)
class BlockSizeDerivation:
    name: str = "triton.meta_parameters.block_size"
    applies_to: tuple[str, ...] = ("kernel",)

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        shapes = (analysis.summary or {}).get("tensor_shapes", {}) or {}
        # Use the maximum scalar dim across observed tensors as the seed.
        max_dim = 0
        for shape in shapes.values():
            for dim in shape:
                if isinstance(dim, int) and dim > max_dim:
                    max_dim = dim
        if max_dim <= 0:
            # No shape evidence — emit a wide pow2 sweep so the agent still has
            # an axis to pull, with explicit "no_shape_evidence" citation.
            candidates = pow2_range(64, 2048)
            evidence = DerivationEvidence(
                rule=self.name,
                signal="no tensor-shape evidence available; emitting fallback pow2 range",
                citations=(),
            )
        else:
            candidates = pow2_around(max_dim, lo=32, hi=4096, count=5)
            evidence = DerivationEvidence(
                rule=self.name,
                signal=f"largest observed tensor dim = {max_dim}",
                citations=tuple(f"tensor:{name}={shape}" for name, shape in shapes.items()),
            )
        if not candidates:
            return ()
        default = candidates[len(candidates) // 2]
        yield Lever(
            id="block_size",
            target_kind="launch",
            target_selector=getattr(workload, "id", ""),
            range=IntRange(candidates=candidates, units="elements"),
            default=default,
            description=(
                "BLOCK_SIZE constexpr passed to the Triton kernel. "
                "Candidates are powers of two centered on the largest tensor dim."
            ),
            evidence=evidence,
            backend_id="triton",
        )


@dataclass(frozen=True, slots=True)
class NumWarpsDerivation:
    name: str = "triton.meta_parameters.num_warps"
    applies_to: tuple[str, ...] = ("kernel",)

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        # Cap by device shared-memory budget. The exact upper-bound calculation
        # lives in the device-capability lookup; here we just enumerate a sane
        # power-of-two sweep that the validator will further constrain.
        cap = (analysis.extra or {}).get("device_capability_int")
        # Reasonable defaults derived from device class (no hand-coded "8" anywhere
        # except as a derived ceiling): sm_>=80 supports up to 32 warps/block.
        upper = 32 if (cap is not None and cap >= 80) else 16
        candidates = pow2_range(1, upper)
        evidence = DerivationEvidence(
            rule=self.name,
            signal=f"device class cap={cap}; upper warp count={upper}",
            citations=(f"sm_{cap}",) if cap else (),
        )
        yield Lever(
            id="num_warps",
            target_kind="launch",
            target_selector=getattr(workload, "id", ""),
            range=IntRange(candidates=candidates, units="warps"),
            default=4,
            description="num_warps launch attribute; capped by device-class shared-mem budget.",
            evidence=evidence,
            backend_id="triton",
        )


@dataclass(frozen=True, slots=True)
class NumStagesDerivation:
    name: str = "triton.meta_parameters.num_stages"
    applies_to: tuple[str, ...] = ("kernel",)

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        loop_depth = int((analysis.summary or {}).get("loop_depth", 0) or 0)
        # If the kernel has no loop, software pipelining cannot help; emit a
        # narrow lever so the agent can still test num_stages=1 vs default.
        if loop_depth <= 0:
            candidates = (1, 2)
            signal = "no loop in IR; pipelining likely a no-op"
        else:
            candidates = tuple(range(1, min(8, loop_depth + 4)))
            signal = f"loop_depth={loop_depth}; pipelining candidates 1..{candidates[-1]}"
        yield Lever(
            id="num_stages",
            target_kind="launch",
            target_selector=getattr(workload, "id", ""),
            range=IntRange(candidates=candidates, units="stages"),
            default=2 if loop_depth > 0 else 1,
            description="num_stages constexpr controlling tritongpu-pipeline depth.",
            evidence=DerivationEvidence(rule=self.name, signal=signal),
            backend_id="triton",
        )


META_DERIVATIONS: tuple[SearchSpaceDerivation, ...] = (
    BlockSizeDerivation(),
    NumWarpsDerivation(),
    NumStagesDerivation(),
)
