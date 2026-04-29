"""Lever per `register_lowering`-able aten op observed in the FX graph.

Inductor's per-op algorithm choice is settable via the
`torch._inductor.lowering.lowerings[op]` dict. For each high-impact aten op
seen in the captured FX graph, emit a `StructuredJsonRange` lever the agent
can fill with a dotted reference (`module:fn`) to a replacement lowering.

The intervention's payload is resolved at runtime by the agent runner's tool
wrapper — this module only declares which ops have a swap surface.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from compilagent.core.search_space import (
    DerivationEvidence,
    Lever,
    SearchSpaceDerivation,
    StructuredJsonRange,
)

_OP_RE = re.compile(r"\btorch\.ops\.aten\.(?P<op>[a-zA-Z_][a-zA-Z0-9_]*)")

# Curated set of ops we know are productive to swap for ML workloads.
_HIGH_IMPACT = {
    "softmax", "_softmax", "_softmax_backward_data",
    "layer_norm", "native_layer_norm", "native_layer_norm_backward",
    "gelu", "gelu_backward",
    "matmul", "mm", "addmm", "bmm", "baddbmm",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
}


@dataclass(frozen=True, slots=True)
class LoweringOverrideDerivation:
    name: str = "torch_inductor.lowering_overrides"
    applies_to: tuple[str, ...] = ("full_model", "fused_subgraph")

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        fx_text: str = (analysis.extra or {}).get("fx_text", "") or ""
        observed: set[str] = {m.group("op") for m in _OP_RE.finditer(fx_text)}
        targets = sorted(observed & _HIGH_IMPACT)
        if not targets and fx_text:
            # Substring-match for partial names like `_softmax`
            for op in _HIGH_IMPACT:
                if op in fx_text:
                    targets.append(op)
        for op_name in targets:
            evidence = DerivationEvidence(
                rule=self.name,
                signal=f"aten op `{op_name}` observed in FX graph; lowering swap is available",
                citations=(f"aten:{op_name}",),
            )
            yield Lever(
                id=f"lowering:{op_name}",
                target_kind="lowering",
                target_selector=f"aten.{op_name}",
                range=StructuredJsonRange(
                    examples=(
                        {"replacement": "module.path:my_fn",
                         "doc": "dotted reference to the replacement lowering callable"},
                    ),
                    schema_hint='{"replacement": "<module>:<fn>"}',
                ),
                default={"replacement": None},
                description=(
                    f"Replace inductor's lowering rule for aten.{op_name}. "
                    "The replacement callable is hot-loaded for the candidate's "
                    "compile and restored on exit."
                ),
                evidence=evidence,
                backend_id="torch_inductor",
            )


LOWERING_DERIVATIONS: tuple[SearchSpaceDerivation, ...] = (LoweringOverrideDerivation(),)
