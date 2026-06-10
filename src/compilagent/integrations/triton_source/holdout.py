"""Unseen-config holdout generator for `triton_source` workloads (D10).

Winning candidates are tuned against ONE seen input configuration; this
module derives ~6 held-out configurations per workload — one per category —
by transforming the workload's symbolic shape template
(``spec.metadata["holdout"]``: per-input factory + dim variable names, var →
{size, free}). Tied axes stay tied because they share one variable; fixed
axes (bound to parameter shapes, e.g. a LayerNorm hidden dim) never move, so
the candidate's `ModelNew` still state_dict-copies against the reference.

Categories:

  - ``edge_boundary``         — leading free axis → 1, others tiny/odd (3):
    masking and zero-guard bugs.
  - ``scale_up``              — leading free axis × 2 (element-budget capped).
  - ``scale_down``            — every free axis → max(1, seen // 4).
  - ``alignment_stress``      — every free axis → the next prime > seen:
    non-power-of-2, non-multiple-of-warp sizes stress BLOCK masking.
  - ``asymmetric_aspect``     — leading free axis × 8, next free axis ÷ 8.
  - ``production_realistic``  — leading free axis → 1000 (a deployment-ish,
    non-power-of-2 round batch).

Application is purely textual and backend-idiomatic: `apply_to_reference`
appends a shadowing ``get_inputs`` (explicit dims, ``device='cuda'``,
``dtype=torch.float32``) to the workload's reference module source, which the
sandbox then evaluates exactly like any other reference. Everything here is
CPU-only; the GPU work happens in `scripts/eval_holdout.py` through the
sandbox + lease layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compilagent.core.workload import WorkloadSpec

CATEGORIES: tuple[str, ...] = (
    "edge_boundary",
    "scale_up",
    "scale_down",
    "alignment_stress",
    "asymmetric_aspect",
    "production_realistic",
)

#: Per-tensor element budget for transformed configs. Never below 2x the
#: largest seen input so scale-up stays meaningful for already-huge tasks.
_MIN_ELEMENT_BUDGET = 2**31


@dataclass(frozen=True, slots=True)
class HoldoutConfig:
    """One held-out input configuration."""

    category: str
    sizes: dict[str, int]  # var name → value (fixed vars keep seen size)
    input_shapes: dict[str, list[int]]  # input name → concrete shape
    get_inputs_source: str  # shadowing get_inputs() definition


def _next_prime(n: int) -> int:
    """Smallest prime strictly greater than `n`."""

    def _is_prime(x: int) -> bool:
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        f = 3
        while f * f <= x:
            if x % f == 0:
                return False
            f += 2
        return True

    candidate = n + 1
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def _template_of(spec: WorkloadSpec) -> dict[str, Any]:
    template = spec.metadata.get("holdout")
    if not isinstance(template, dict) or not template.get("inputs"):
        raise ValueError(
            f"workload `{spec.id}` has no holdout shape template "
            '(metadata["holdout"]) — cannot generate unseen configs.'
        )
    return template


def _shapes_for(template: dict[str, Any], sizes: dict[str, int]) -> dict[str, list[int]]:
    return {
        entry["name"]: [int(sizes[d]) for d in entry["dims"]]
        for entry in template["inputs"]
    }


def _max_elements(template: dict[str, Any], sizes: dict[str, int]) -> int:
    worst = 1
    for shape in _shapes_for(template, sizes).values():
        elements = 1
        for dim in shape:
            elements *= dim
        worst = max(worst, elements)
    return worst


def _capped_scale(
    template: dict[str, Any],
    sizes: dict[str, int],
    var: str,
    factor: int,
    budget: int,
) -> int:
    """Largest `var` value ≤ seen*factor that keeps every input tensor
    within the element budget (but never below the seen size)."""

    seen = sizes[var]
    value = seen * factor
    while value > seen:
        trial = dict(sizes, **{var: value})
        if _max_elements(template, trial) <= budget:
            return value
        value = max(seen, int(value * 3 / 4))
    return seen


def _get_inputs_source(template: dict[str, Any], sizes: dict[str, int]) -> str:
    lines = ["def get_inputs():", "    return ["]
    for entry in template["inputs"]:
        dims = ", ".join(str(int(sizes[d])) for d in entry["dims"])
        lines.append(
            f"        torch.{entry['factory']}"
            f"({dims}, device='cuda', dtype=torch.float32),"
        )
    lines.append("    ]")
    return "\n".join(lines)


def generate_holdout_configs(spec: WorkloadSpec) -> tuple[HoldoutConfig, ...]:
    """Six held-out configs (one per category) for one workload spec."""

    template = _template_of(spec)
    vars_: dict[str, dict[str, Any]] = template["vars"]
    seen = {name: int(v["size"]) for name, v in vars_.items()}
    free = [name for name, v in vars_.items() if v.get("free")]
    if not free:
        raise ValueError(
            f"workload `{spec.id}` has no free axes in its holdout template."
        )
    leading = free[0]
    budget = max(_MIN_ELEMENT_BUDGET, 2 * _max_elements(template, seen))

    def _config(category: str, sizes: dict[str, int]) -> HoldoutConfig:
        return HoldoutConfig(
            category=category,
            sizes=sizes,
            input_shapes=_shapes_for(template, sizes),
            get_inputs_source=_get_inputs_source(template, sizes),
        )

    configs: list[HoldoutConfig] = []

    sizes = dict(seen)
    for name in free:
        sizes[name] = 1 if name == leading else max(1, min(3, seen[name]))
    configs.append(_config("edge_boundary", sizes))

    sizes = dict(seen)
    sizes[leading] = _capped_scale(template, seen, leading, 2, budget)
    configs.append(_config("scale_up", sizes))

    sizes = dict(seen)
    for name in free:
        sizes[name] = max(1, seen[name] // 4)
    configs.append(_config("scale_down", sizes))

    sizes = dict(seen)
    for name in free:
        sizes[name] = _next_prime(seen[name])
    if _max_elements(template, sizes) > budget:  # prime bump may tip huge tasks
        sizes = dict(seen, **{leading: _next_prime(seen[leading])})
    configs.append(_config("alignment_stress", sizes))

    sizes = dict(seen)
    if len(free) > 1:  # shrink the second axis first, then stretch the lead
        sizes[free[1]] = max(1, seen[free[1]] // 8)
    sizes[leading] = _capped_scale(template, sizes, leading, 8, budget)
    configs.append(_config("asymmetric_aspect", sizes))

    sizes = dict(seen)
    target = 1000 if seen[leading] != 1000 else 1296
    while target > 1 and _max_elements(template, dict(sizes, **{leading: target})) > budget:
        target = max(1, int(target * 3 / 4))
    sizes[leading] = target
    configs.append(_config("production_realistic", sizes))

    # De-duplicate against the seen config and each other by nudging the
    # leading axis (rare: tiny seen sizes can make categories collide).
    taken = {tuple(map(tuple, sorted(seen.items())))}
    unique: list[HoldoutConfig] = []
    for config in configs:
        sizes = dict(config.sizes)
        while tuple(map(tuple, sorted(sizes.items()))) in taken:
            sizes[leading] = sizes[leading] + 1
        if sizes != config.sizes:
            config = _config(config.category, sizes)
        taken.add(tuple(map(tuple, sorted(sizes.items()))))
        unique.append(config)
    return tuple(unique)


def apply_to_reference(reference_source: str, config: HoldoutConfig) -> str:
    """Reference module + shadowing `get_inputs` for one holdout config.

    Python module semantics make the LAST definition win, so appending a
    fresh `get_inputs` re-parameterizes the inputs without touching the
    model (whose parameter shapes the candidate mirrors).
    """

    return (
        f"{reference_source.rstrip()}\n\n"
        f"# --- D10 holdout override: {config.category} "
        f"{config.input_shapes} ---\n"
        f"{config.get_inputs_source}\n"
    )
