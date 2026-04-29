"""Lever-per-knob derivation against the introspected `KnobCatalog`.

We don't enumerate knobs by hand. Instead we walk the catalog produced by
`backends/torch_inductor/knobs.py:build_knob_catalog`, classify each knob by
the runtime type of its default, and emit a typed lever:

  - bool       → BooleanFlag
  - int        → IntRange     (candidates produced by KnobDescriptor)
  - float      → FloatRange
  - Literal[…] → EnumChoice   (from the annotation walk)
  - other      → StructuredJsonRange with the default as the example

Knobs whose default is `None`/callable/dict are emitted as
`StructuredJsonRange` so the agent sees them but cannot fire-and-forget.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from compilagent.core.search_space import (
    BooleanFlag,
    DerivationEvidence,
    EnumChoice,
    FloatRange,
    IntRange,
    Lever,
    SearchSpaceDerivation,
    StructuredJsonRange,
)

from ..knobs import KnobDescriptor, build_knob_catalog

_INTERESTING_INDUCTOR_KNOBS = {
    "epilogue_fusion",
    "max_autotune",
    "max_autotune_pointwise",
    "max_autotune_gemm",
    "max_autotune_gemm_backends",
    "max_autotune_gemm_search_space",
    "max_autotune_conv_backends",
    "coordinate_descent_tuning",
    "coordinate_descent_check_all_directions",
    "shape_padding",
    "comprehensive_padding",
    "layout_optimization",
    "force_layout_optimization",
    "loop_ordering_after_fusion",
    "loop_index_inversion_in_fusion",
    "online_softmax",
    "reorder_for_locality",
    "reorder_for_peak_memory",
    "split_reductions",
    "pattern_matcher",
    "prologue_fusion",
    "aggressive_fusion",
    "force_pointwise_cat",
    "fallback_random",
    "use_static_cuda_launcher",
    "score_fusion_memory_threshold",
    "max_pointwise_cat_inputs",
    "unroll_reductions_threshold",
    "realize_reads_threshold",
    "realize_opcount_threshold",
    "realize_acc_reads_threshold",
}


def _knob_to_lever(knob: KnobDescriptor, workload_id: str) -> Lever | None:
    """Build a typed lever for a knob if the runtime type is supported."""

    default = knob.default
    selector = knob.name  # "inductor.<leaf>" or "dynamo.<leaf>"
    description = (
        f"{knob.namespace}.{knob.leaf} (type={knob.py_type}). "
        "Override is scoped to the candidate's compile via context manager."
    )
    evidence = DerivationEvidence(
        rule="torch_inductor.knobs",
        signal=f"introspected from torch._{knob.namespace}.config",
        citations=(f"knob:{selector}",),
    )
    if isinstance(default, bool):
        rng: Any = BooleanFlag()
    elif knob.candidates and all(isinstance(v, int) and not isinstance(v, bool) for v in knob.candidates):
        rng = IntRange(candidates=tuple(knob.candidates))
    elif knob.candidates and all(isinstance(v, float) for v in knob.candidates):
        rng = FloatRange(candidates=tuple(knob.candidates))
    elif knob.candidates and all(isinstance(v, str) for v in knob.candidates):
        rng = EnumChoice(candidates=tuple(knob.candidates))
    else:
        # Unsupported shape — surface as a structured-json lever so the agent
        # can still propose a literal value but understands it's an open type.
        rng = StructuredJsonRange(
            examples=tuple({selector: v} for v in (knob.candidates or (default,))),
            schema_hint=f'{{"{selector}": <{knob.py_type}>}}',
        )
    return Lever(
        id=f"knob:{selector}",
        target_kind="knob",
        target_selector=selector,
        range=rng,
        default=default,
        description=description,
        evidence=evidence,
        backend_id="torch_inductor",
    )


@dataclass(frozen=True, slots=True)
class InductorKnobDerivation:
    name: str = "torch_inductor.knobs"
    applies_to: tuple[str, ...] = ("full_model", "fused_subgraph")
    only_interesting: bool = True

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        catalog = build_knob_catalog()
        workload_id = getattr(workload, "id", "") or ""
        for knob in catalog.knobs:
            if self.only_interesting and knob.namespace == "inductor" and knob.leaf not in _INTERESTING_INDUCTOR_KNOBS:
                continue
            if knob.namespace == "dynamo" and knob.leaf not in {
                "suppress_errors", "specialize_int", "specialize_float",
                "automatic_dynamic_shapes", "assume_static_by_default",
            }:
                continue
            lever = _knob_to_lever(knob, workload_id)
            if lever is not None:
                yield lever


KNOB_DERIVATIONS: tuple[SearchSpaceDerivation, ...] = (InductorKnobDerivation(),)
