"""`ExperimentLogPolicy` — CASCADE's C6 cross-task skill memory (ticket E6).

A `CandidatePolicy` that closes the loop opened by ticket E9:

  - ``consult()`` injects constraint rules as `PolicyHint`s from a JSON
    rules file: the P0-probe failure-taxonomy SEED rules (always injected)
    plus rules distilled from earlier runs whose frequency has reached
    ``min_frequency`` (default 2 — a lesson must repeat before it earns
    prompt space, the AdaExplore-style frequency filter). It also recalls
    the best prior validated result for the same (family, backend, arch)
    from the core `ExperimentLog` (the E9 reader) as one concrete hint.
  - ``observe()`` (the E9 update hook) distills every failed candidate
    into a one-line rule — failing gate + a gate-specific lesson + the
    error head — keyed so identical failures increment one frequency
    counter, persisted to the rules file after each observation (a
    crashed run still keeps its lessons).

The rules file lives next to the experiment log
(``<workspace>/memory/skill_rules.json``) and is created on first use,
seeded from the P0 failure taxonomy (08_p0_probe_findings.md §4).

Backend-generic (ticket D9): the store is keyed by decision space.
`triton_source` workloads use the kernel-source rules above; every other
backend (the `torch_inductor` knob space, the `triton` MLIR pass-pipeline
space) uses a sibling LEVER-MODE rules file
(``<workspace>/memory/skill_rules_lever.json``) seeded from
`LEVER_SEED_RULES` — the knob/pass material recorded in the existing
ExperimentLog (`.compilagent/memory/experiments.jsonl`) plus the canonical
`validate_intervention` rejection modes of both lever backends. Distilled
rules from observe() land in whichever file matches the failing workload's
backend, so source lessons never leak into lever prompts and vice versa.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from compilagent.core.analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    TimingResult,
)
from compilagent.core.candidate_policy import PolicyHint
from compilagent.core.plan import Intervention, Plan, Target
from compilagent.core.workload import WorkloadSpec
from compilagent.storage.experiment_log import ExperimentLog

#: Constraint rules seeded from the P0 capability-probe failure taxonomy.
#: These are always injected (no frequency gate) — they were each observed
#: repeatedly across the probe's 54 generations.
P0_SEED_RULES: tuple[tuple[str, str], ...] = (
    (
        "p0:tl_sum_mask",
        "tl.sum has NO mask keyword — mask at load time instead: "
        "tl.load(ptr, mask=..., other=<neutral element>).",
    ),
    (
        "p0:loop_carried_dtype",
        "Loop-carried variables must keep ONE dtype/shape across all "
        "iterations (e.g. a running max must stay a tensor of the same "
        "shape, not flip between scalar and tensor).",
    ),
    (
        "p0:constexpr_indexing",
        "Never index a tensor with a tl.constexpr loop variable — "
        "constexpr values are compile-time; use tl.arange offsets instead.",
    ),
    (
        "p0:branch_defined_vars",
        "Define every variable on ALL branches before use — a variable "
        "assigned only inside `if` is undefined on the other path.",
    ),
    (
        "p0:gemm_k_tail_mask",
        "GEMM kernels need explicit out-of-bounds masking on the K-tail "
        "(the last k-block when K % BLOCK_K != 0), or results are corrupt.",
    ),
    (
        "p0:single_pass_reductions",
        "Prefer single-pass (online) kernels for reductions: compute "
        "running max/sum in one sweep instead of multiple passes over "
        "global memory.",
    ),
)

#: Lever-mode constraint rules (ticket D9), always injected for non-source
#: backends. Seed material: the `triton` rows of the existing ExperimentLog
#: (`.compilagent/memory/experiments.jsonl` — validated pass-plan wins on
#: cuda:sm_120; the log records no knob/pass compile failures yet, so the
#: failure-shaped seeds encode the canonical `validate_intervention`
#: rejection modes of `torch_inductor` and `triton`, which surface as
#: retryable propose_candidate errors).
LEVER_SEED_RULES: tuple[tuple[str, str], ...] = (
    (
        "lever:typed_kinds",
        "Use only intervention kinds the backend accepts (torch_inductor: "
        "knob/lowering/fx_node/scheduler/choices; triton: pass/launch/knob)"
        " — any other target.kind is rejected before reaching the compiler.",
    ),
    (
        "lever:pass_selector_format",
        "Triton pass selectors must be `<stage>:<pass_name>` with stage "
        "ttir or ttgir, payload {\"action\": \"run\"|\"skip\"|\"replace\", "
        "\"args\": {...}} — a non-dict payload or unknown action is "
        "rejected.",
    ),
    (
        "lever:knob_payload_type",
        "Inductor knob payloads must match the knob's runtime type exactly "
        "(bool knobs take true/false, int knobs take ints, enum knobs take "
        "a listed member) — type mismatches fail the compile or silently "
        "no-op.",
    ),
    (
        "lever:respect_ranges",
        "Pick payloads from the lever's advertised range/candidates; "
        "off-catalog values may be rejected and out-of-range values waste "
        "a failed attempt.",
    ),
    (
        "lever:small_plans",
        "Change few levers per candidate (1-3) so the measured delta is "
        "attributable; sweep numeric levers in powers of two around the "
        "default.",
    ),
    (
        "lever:observed_wins",
        "Validated wins recorded on cuda:sm_120 (ExperimentLog): "
        "ttgir:tritongpu-pipeline with num_stages 1-2 (up to 1.06x), "
        "tritongpu-coalesce, tritongpu-optimize-thread-locality — "
        "pipeline-depth and memory-locality levers are good first probes.",
    ),
)

#: rule `source` tags that bypass the frequency gate (seed taxonomies).
_SEED_SOURCES = frozenset({"p0_seed", "lever_seed"})

#: Backends whose decision space is the kernel source itself; everything
#: else is a lever (knob/pass) space and uses the lever rule store.
_SOURCE_BACKENDS = frozenset({"triton_source"})

#: One-line lesson per failing E2a gate, used when distilling new rules.
_GATE_LESSONS: dict[str, str] = {
    "g1_shape_dtype": (
        "output must match the reference shape AND dtype exactly"
    ),
    "g2_allclose": (
        "numerical drift vs reference — check load masking "
        "(other=<neutral>), accumulator dtype, and reduction order"
    ),
    "g3_no_alias_no_mutation": (
        "never return a view/alias of an input buffer and never mutate "
        "inputs in place"
    ),
    "g4_banned_api": (
        "a banned torch API was used — the core computation must be a "
        "Triton kernel"
    ),
    "g5_determinism": (
        "the kernel must be deterministic for a fixed seed (avoid "
        "non-deterministic atomics)"
    ),
}


def _error_head(text: str | None, limit: int = 160) -> str:
    head = (text or "").strip().splitlines()
    return head[0][:limit] if head else ""


class ExperimentLogPolicy:
    """Skill-memory `CandidatePolicy` over ExperimentLog + a rules file."""

    name = "experiment_log"

    def __init__(
        self,
        root: Path,
        *,
        rules_path: Path | None = None,
        lever_rules_path: Path | None = None,
        min_frequency: int = 2,
        max_rules: int = 12,
        experiment_log: ExperimentLog | None = None,
    ) -> None:
        self.root = Path(root)
        self.rules_path = rules_path or (self.root / "memory" / "skill_rules.json")
        self.lever_rules_path = lever_rules_path or self.rules_path.with_name(
            "skill_rules_lever.json"
        )
        self.min_frequency = int(min_frequency)
        self.max_rules = int(max_rules)
        self.experiment_log = experiment_log or ExperimentLog(self.root)

    # ---- rules file -------------------------------------------------------

    def _rules_store(
        self, backend_id: str | None
    ) -> tuple[Path, tuple[tuple[str, str], ...], str]:
        """(path, seed rules, seed source-tag) for one decision space."""

        if backend_id and backend_id not in _SOURCE_BACKENDS:
            return self.lever_rules_path, LEVER_SEED_RULES, "lever_seed"
        return self.rules_path, P0_SEED_RULES, "p0_seed"

    def _load_rules(
        self, backend_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Read the backend's rules file, seeding its taxonomy if absent."""

        path, seeds, seed_source = self._rules_store(backend_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rules = data.get("rules")
                if isinstance(rules, list):
                    return rules
            except (OSError, json.JSONDecodeError):
                pass
        rules = [
            {"id": rid, "text": text, "frequency": 1, "source": seed_source}
            for rid, text in seeds
        ]
        self._save_rules(rules, backend_id)
        return rules

    def _save_rules(
        self, rules: list[dict[str, Any]], backend_id: str | None = None
    ) -> None:
        path, _, _ = self._rules_store(backend_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"rules": rules}, indent=2), encoding="utf-8"
            )
            tmp.replace(path)
        except OSError:
            return None

    # ---- consult (read path) ----------------------------------------------

    def consult(
        self,
        *,
        workload: WorkloadSpec,
        analysis: Analysis,
        family: str | None,
        arch: str,
    ) -> Sequence[PolicyHint]:
        hints: list[PolicyHint] = []

        # (a) constraint rules: seeds always, distilled rules only once
        # they have recurred (frequency gate). The rule store is selected
        # by the workload's decision space (source vs lever).
        injectable = [
            r
            for r in self._load_rules(workload.backend_id)
            if r.get("source") in _SEED_SOURCES
            or int(r.get("frequency", 0)) >= self.min_frequency
        ]
        injectable.sort(key=lambda r: -int(r.get("frequency", 0)))
        for rule in injectable[: self.max_rules]:
            freq = int(rule.get("frequency", 1))
            hints.append(
                PolicyHint(
                    suggested_interventions=(),
                    rationale=f"constraint rule: {rule.get('text', '')}",
                    confidence=min(1.0, freq / 4.0),
                    metadata={
                        "kind": "constraint_rule",
                        "rule_id": rule.get("id"),
                        "frequency": freq,
                        "source": rule.get("source"),
                    },
                )
            )

        # (b) best prior validated result for this context, via the E9
        # ExperimentLog reader.
        rows = self.experiment_log.recall(
            backend_id=workload.backend_id,
            family=family,
            arch=arch,
            successful_only=True,
            top_n=1,
        )
        for row in rows:
            interventions = tuple(
                Intervention(
                    target=Target(
                        kind=str((iv.get("target") or {}).get("kind", "")),
                        selector=str((iv.get("target") or {}).get("selector", "")),
                    ),
                    payload=iv.get("payload"),
                    rationale=str(iv.get("rationale", "")),
                )
                for iv in (row.get("interventions") or [])
                if isinstance(iv, dict)
            )
            speedup = row.get("speedup")
            hints.append(
                PolicyHint(
                    suggested_interventions=interventions,
                    rationale=(
                        "prior validated result on workload "
                        f"`{row.get('workload_id')}` (same family/arch): "
                        f"speedup {speedup}"
                    ),
                    confidence=0.5,
                    metadata={"kind": "prior_result", "run_id": row.get("run_id")},
                )
            )
        return tuple(hints)

    # ---- observe (E9 update path) -------------------------------------------

    def observe(
        self,
        *,
        workload: WorkloadSpec,
        candidate_id: str,
        plan: Plan,
        compile_result: CompileResult,
        timing: TimingResult | None,
        correctness: CorrectnessResult | None,
        speedup: float | None,
        successful: bool,
        family: str | None,
        arch: str,
    ) -> None:
        if successful:
            return
        distilled = self._distill(compile_result, correctness)
        if distilled is None:
            return
        rule_id, text = distilled
        rules = self._load_rules(workload.backend_id)
        for rule in rules:
            if rule.get("id") == rule_id:
                rule["frequency"] = int(rule.get("frequency", 1)) + 1
                break
        else:
            rules.append(
                {
                    "id": rule_id,
                    "text": text,
                    "frequency": 1,
                    "source": "distilled",
                    "workload_id": workload.id,
                }
            )
        self._save_rules(rules, workload.backend_id)

    @staticmethod
    def _distill(
        compile_result: CompileResult,
        correctness: CorrectnessResult | None,
    ) -> tuple[str, str] | None:
        """Failing gate + one-line lesson → (stable rule id, rule text)."""

        if correctness is not None and not correctness.ok:
            gate = correctness.failed_at or "correctness"
            lesson = _GATE_LESSONS.get(gate, "candidate failed a hardened gate")
            head = _error_head(correctness.diagnostics)
            digest = hashlib.sha256(f"{gate}|{head}".encode()).hexdigest()[:8]
            text = f"[{gate}] {lesson}" + (f" — seen as: {head}" if head else "")
            return f"gate:{gate}:{digest}", text
        if not compile_result.ok:
            head = _error_head(compile_result.diagnostics)
            if not head:
                return None
            digest = hashlib.sha256(head.encode()).hexdigest()[:8]
            return (
                f"compile:{digest}",
                f"[compile] avoid this failure mode: {head}",
            )
        return None
