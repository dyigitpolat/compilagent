"""Per-pass intervention levers derived from baseline IR-diff signal.

When the runtime compiles a baseline pipeline pass-by-pass (`pipeline.execute_plan`),
each pass records its `ir_after_size` and elapsed time. Passes that materially
changed the IR are the ones worth letting the agent skip / parameterize; passes
that produced zero diff are deprioritized.

This rule reads `analysis.extra["pass_impact"]` (a list of dicts emitted by the
pipeline executor), and emits one `Lever` per impactful pass with a
`StructuredJsonRange` payload describing the available actions.
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
class PassInterventionDerivation:
    name: str = "triton.pass_impact"
    applies_to: tuple[str, ...] = ("kernel",)
    min_duration_ms: float = 0.05      # ignore noise

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]:
        impacts = (analysis.extra or {}).get("pass_impact", []) or []
        if impacts:
            yield from self._from_impact_log(impacts)
            return
        # No impact log available — fall back to the active default pipeline.
        yield from self._from_pipeline_catalog(analysis)

    def _from_impact_log(self, impacts: Any) -> Iterable[Lever]:
        for record in impacts:
            if not isinstance(record, dict):
                continue
            stage = record.get("stage", "ttgir")
            pass_name = record.get("name") or record.get("pass")
            if not pass_name:
                continue
            duration_ms = float(record.get("duration_ms", 0.0) or 0.0)
            ir_changed = bool(record.get("ir_changed", False))
            if not ir_changed and duration_ms < self.min_duration_ms:
                continue
            yield self._make_lever(
                stage=stage,
                pass_name=pass_name,
                signal=(
                    f"{pass_name} ran for {duration_ms:.2f}ms"
                    + (" and modified IR" if ir_changed else " (no IR diff)")
                ),
                citations=(f"baseline_pass:{stage}:{pass_name}",),
            )

    def _from_pipeline_catalog(self, analysis: Any) -> Iterable[Lever]:
        """Emit a lever for every pass in the default pipeline at this capability.

        When the baseline did not run pass-by-pass (no IR-diff log), we still
        give the agent a lever per actual MLIR pass in the active pipeline, so
        it can propose `skip` / parameter overrides on real compiler decisions.
        """

        try:
            from ..pipeline import build_ttgir_plan, build_ttir_plan
        except Exception:  # noqa: BLE001
            return
        cap = int((analysis.extra or {}).get("device_capability_int", 80) or 80)
        try:
            ttir_plan = build_ttir_plan(
                capability=cap, num_warps=4, num_stages=3, num_ctas=1,
            )
            ttgir_plan = build_ttgir_plan(
                capability=cap, num_warps=4, num_stages=3, num_ctas=1,
            )
        except Exception:  # noqa: BLE001
            return
        for plan, stage in ((ttir_plan, "ttir"), (ttgir_plan, "ttgir")):
            seen: set[str] = set()
            for step in plan.steps:
                pname = step.descriptor.name
                if pname in seen:
                    continue
                seen.add(pname)
                yield self._make_lever(
                    stage=stage,
                    pass_name=pname,
                    signal=(
                        f"{pname} runs at the {stage} stage on the active "
                        f"compute capability (sm_{cap}); a real compiler heuristic"
                    ),
                    citations=(f"pipeline:{stage}", f"sm_{cap}"),
                )

    def _make_lever(
        self, *, stage: str, pass_name: str, signal: str, citations: tuple[str, ...],
    ) -> Lever:
        return Lever(
            id=f"pass:{stage}:{pass_name}",
            target_kind="pass",
            target_selector=f"{stage}:{pass_name}",
            range=StructuredJsonRange(
                examples=(
                    {"action": "skip"},
                    {"action": "run"},
                ),
                schema_hint='{"action": "run"|"skip"|"replace", "args"?: {...}}',
            ),
            default={"action": "run"},
            description=(
                f"Override the {pass_name} MLIR pass at the {stage} stage. "
                "`skip` drops the pass entirely; `replace` substitutes an "
                "alternate; `args` overrides numeric pass parameters such as "
                "`num_stages` for `tritongpu-pipeline`."
            ),
            evidence=DerivationEvidence(
                rule=self.name, signal=signal, citations=citations,
            ),
            backend_id="triton",
        )


PASS_DERIVATIONS: tuple[SearchSpaceDerivation, ...] = (PassInterventionDerivation(),)
