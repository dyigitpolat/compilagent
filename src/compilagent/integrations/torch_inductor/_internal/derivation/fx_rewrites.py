"""FX-graph rewrite candidates derived from the captured FX module.

For each high-impact aten op observed in the captured FX graph (SDPA, softmax,
layer_norm, gelu, …), emit a `StructuredJsonRange` lever describing the rewrite
options known to be meaningful for that op. The set of levers comes from the
graph, not a hand-coded op list — though we keep a curated map of "interesting"
ops to their available rewrite recipes.

Rewrite payloads are passed to the backend as Python callables in the
intervention's payload (resolved by the agent runner's tool wrapper from a
dotted path); this module only declares the available levers.
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

_OP_REWRITES = {
    # SDPA: choose the dispatch backend or force decomposition for fusion.
    "scaled_dot_product_attention": [
        {"recipe": "force_math_decomposition",
         "doc": "Decompose SDPA into matmul/softmax/matmul for inductor fusion."},
        {"recipe": "force_flash_attention",
         "doc": "Pin SDPA to FlashAttention via torch.nn.attention.sdpa_kernel."},
        {"recipe": "force_efficient_attention",
         "doc": "Pin SDPA to memory-efficient attention."},
    ],
    "softmax": [
        {"recipe": "online_softmax_on", "doc": "Force inductor's online_softmax fast path."},
        {"recipe": "online_softmax_off", "doc": "Disable online_softmax to expose larger fused range."},
    ],
    "layer_norm": [
        {"recipe": "fuse_with_residual", "doc": "Fuse LN with the prior residual add via FX rewrite."},
    ],
    "gelu": [
        {"recipe": "approximate_tanh", "doc": "Use the tanh approximation of GELU."},
        {"recipe": "exact_erf", "doc": "Use the exact erf-based GELU."},
    ],
}


_OP_RE = re.compile(r"\btorch\.ops\.aten\.(?P<op>[a-zA-Z_][a-zA-Z0-9_]*)")


@dataclass(frozen=True, slots=True)
class FxRewriteDerivation:
    name: str = "torch_inductor.fx_rewrites"
    applies_to: tuple[str, ...] = ("full_model", "fused_subgraph")

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        # The analysis carries a captured FX graph size; if zero we can't read
        # ops. The Phase-6 inductor analysis pipeline also stores `fx_text` in
        # `analysis.extra["fx_text"]` if available — fall back gracefully.
        fx_text: str = (analysis.extra or {}).get("fx_text", "") or ""
        observed_ops: set[str] = set()
        for m in _OP_RE.finditer(fx_text):
            observed_ops.add(m.group("op"))
        # Surface any rewrite recipes whose target op was actually observed.
        for op_name, recipes in _OP_REWRITES.items():
            if not any(op_name in seen for seen in observed_ops):
                # The aten op qualname can be e.g. `_scaled_dot_product_flash_attention`;
                # do a substring match against observed_ops too.
                if not any(op_name in seen for seen in observed_ops):
                    if not (fx_text and op_name in fx_text):
                        continue
            evidence = DerivationEvidence(
                rule=self.name,
                signal=f"aten op `{op_name}` observed in FX graph",
                citations=(f"aten:{op_name}",),
            )
            yield Lever(
                id=f"fx_rewrite:{op_name}",
                target_kind="fx_node",
                target_selector=op_name,
                range=StructuredJsonRange(
                    examples=tuple(recipes),
                    schema_hint=(
                        '{"recipe": "<recipe name>", '
                        '"args"?: {...}}'
                    ),
                ),
                default={"recipe": "noop"},
                description=(
                    f"FX-graph rewrite candidates for {op_name}. "
                    "Each recipe is applied inside our custom Dynamo backend "
                    "before `compile_fx` runs."
                ),
                evidence=evidence,
                backend_id="torch_inductor",
            )


FX_REWRITE_DERIVATIONS: tuple[SearchSpaceDerivation, ...] = (FxRewriteDerivation(),)
