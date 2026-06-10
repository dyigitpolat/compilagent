"""Archetype harness integration (tickets E4 + E5 + E6).

Self-registers six agent harnesses in the harness registry at import
time — five published-protocol archetypes plus the composite:

  - ``archetype_sr``   — serial refinement (KernelBench G+E protocol)
  - ``archetype_bon``  — parallel best-of-N at temperature 1.0
  - ``archetype_evo``  — population/archive evolution (EvoEngineer-style)
  - ``archetype_band`` — UCB1 bandit over strategy arms (KernelBand-style)
  - ``archetype_ma``   — profiler-in-the-loop Coder+Judge (CudaForge-style;
                          NCU metrics via triton_source ncu_profile)
  - ``cascade``        — CASCADE v0 composite (C1–C8, C10; per-ingredient
                          toggles for the T2 axis-bundle ablation)

All submit candidates through the canonical session tool protocol and are
model-agnostic via the pydantic_ai integration's model-string resolution.
`ExperimentLogPolicy` (CASCADE's C6 skill memory over the E9
CandidatePolicy observe hook + ExperimentLog reader) also lives here.
"""

from __future__ import annotations

from compilagent.harness.registry import harness_registry

from .bandit import ArchetypeBanditHarness
from .cascade import CascadeConfig, CascadeHarness
from .evolution import ArchetypeEvolutionHarness
from .harness import ArchetypeBestOfNHarness, ArchetypeSerialRefinementHarness
from .profiler_ma import ArchetypeProfilerMAHarness
from .skill_memory import ExperimentLogPolicy

if "archetype_sr" not in harness_registry.ids():
    harness_registry.register("archetype_sr", ArchetypeSerialRefinementHarness)
if "archetype_bon" not in harness_registry.ids():
    harness_registry.register("archetype_bon", ArchetypeBestOfNHarness)
if "archetype_evo" not in harness_registry.ids():
    harness_registry.register("archetype_evo", ArchetypeEvolutionHarness)
if "archetype_band" not in harness_registry.ids():
    harness_registry.register("archetype_band", ArchetypeBanditHarness)
if "cascade" not in harness_registry.ids():
    harness_registry.register("cascade", CascadeHarness)
if "archetype_ma" not in harness_registry.ids():
    harness_registry.register("archetype_ma", ArchetypeProfilerMAHarness)

__all__ = [
    "ArchetypeBanditHarness",
    "ArchetypeBestOfNHarness",
    "ArchetypeEvolutionHarness",
    "ArchetypeProfilerMAHarness",
    "ArchetypeSerialRefinementHarness",
    "CascadeConfig",
    "CascadeHarness",
    "ExperimentLogPolicy",
]
