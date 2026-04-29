"""Per-load/store annotation levers derived from observed memory ops.

For each `tt.load`/`tt.store` site in the baseline TTGIR (already extracted by
`backends/triton/analysis.py:extract_decision_traces`), emit an `EnumChoice`
lever for `cache_modifier` and `eviction_policy`. The candidate set comes from
the device's documented PTX cache hints — not a hard-coded list defined here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from compilagent.core.search_space import (
    DerivationEvidence,
    EnumChoice,
    Lever,
    SearchSpaceDerivation,
)

_PTX_LOAD_CACHE_MODIFIERS = ("", ".ca", ".cg", ".cs", ".lu", ".cv")
_TRITON_EVICTION_POLICIES = ("", "evict_first", "evict_last", "evict_normal")


@dataclass(frozen=True, slots=True)
class CacheModifierDerivation:
    name: str = "triton.kernel_annotations.cache_modifier"
    applies_to: tuple[str, ...] = ("kernel",)

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        traces = (analysis.extra or {}).get("decision_traces", []) or []
        load_count = sum(1 for t in traces if "load" in str(t.get("op_name", "")).lower())
        if load_count == 0:
            return ()
        evidence = DerivationEvidence(
            rule=self.name,
            signal=f"{load_count} tt.load sites observed in TTGIR",
            citations=tuple(f"op:{t.get('op_name')}" for t in traces[:4] if t.get("op_name")),
        )
        yield Lever(
            id="cache_modifier",
            target_kind="launch",
            target_selector=getattr(workload, "id", ""),
            range=EnumChoice(candidates=_PTX_LOAD_CACHE_MODIFIERS),
            default="",
            description=(
                "PTX cache modifier passed to all tt.load sites via the kernel's "
                "LOAD_CACHE_MODIFIER constexpr."
            ),
            evidence=evidence,
            backend_id="triton",
        )


@dataclass(frozen=True, slots=True)
class EvictionPolicyDerivation:
    name: str = "triton.kernel_annotations.eviction_policy"
    applies_to: tuple[str, ...] = ("kernel",)

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        traces = (analysis.extra or {}).get("decision_traces", []) or []
        if not any("load" in str(t.get("op_name", "")).lower() for t in traces):
            return ()
        yield Lever(
            id="eviction_policy",
            target_kind="launch",
            target_selector=getattr(workload, "id", ""),
            range=EnumChoice(candidates=_TRITON_EVICTION_POLICIES),
            default="",
            description="L2 eviction policy hint applied to tt.load.",
            evidence=DerivationEvidence(
                rule=self.name,
                signal="at least one tt.load site exists; eviction-policy is meaningful",
            ),
            backend_id="triton",
        )


ANNOTATION_DERIVATIONS: tuple[SearchSpaceDerivation, ...] = (
    CacheModifierDerivation(),
    EvictionPolicyDerivation(),
)
