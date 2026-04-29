"""One lever per inductor-generated Triton kernel discovered in baseline output.

For each `triton_*` kernel found in the captured `output_code.py`, emit a
StructuredJsonRange lever the agent can fill in with `{BLOCK_SIZE, num_warps,
num_stages}`. The override is plumbed through a `choices` intervention that
biases inductor's `InductorChoices` for that specific kernel.

The set of kernels — and therefore the set of levers — comes entirely from
the captured artifacts; no kernel name list is hand-coded here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from compilagent.core.search_space import (
    DerivationEvidence,
    Lever,
    SearchSpaceDerivation,
    StructuredJsonRange,
)


@dataclass(frozen=True, slots=True)
class PerKernelTuningDerivation:
    name: str = "torch_inductor.per_kernel_tuning"
    applies_to: tuple[str, ...] = ("full_model", "fused_subgraph")
    max_kernels: int = 32

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        kernels = (analysis.extra or {}).get("kernels", []) or []
        for entry in kernels[: self.max_kernels]:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not name:
                continue
            size_hints = entry.get("size_hints") if isinstance(entry, dict) else {}
            evidence = DerivationEvidence(
                rule=self.name,
                signal=f"inductor-generated kernel {name} discovered in baseline output_code",
                citations=tuple(f"size_hint:{k}={v}" for k, v in (size_hints or {}).items()),
            )
            yield Lever(
                id=f"per_kernel:{name}",
                target_kind="choices",
                target_selector=name,
                range=StructuredJsonRange(
                    examples=(
                        {"kernel": name, "BLOCK_SIZE": 64, "num_warps": 4, "num_stages": 2},
                        {"kernel": name, "BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 3},
                        {"kernel": name, "BLOCK_SIZE": 256, "num_warps": 8, "num_stages": 4},
                    ),
                    schema_hint=(
                        '{"kernel": "<inductor kernel name>", '
                        '"BLOCK_SIZE": <int>, "num_warps": <int>, "num_stages": <int>}'
                    ),
                ),
                default={"kernel": name, "BLOCK_SIZE": None, "num_warps": None, "num_stages": None},
                description=(
                    f"Per-kernel autotune bias for inductor-generated {name}. "
                    "Applied via a custom InductorChoices handler scoped to this candidate."
                ),
                evidence=evidence,
                backend_id="torch_inductor",
            )


PER_KERNEL_DERIVATIONS: tuple[SearchSpaceDerivation, ...] = (PerKernelTuningDerivation(),)
