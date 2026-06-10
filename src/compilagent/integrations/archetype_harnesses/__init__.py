"""Archetype harness integration (tickets E4 + E5).

Self-registers four published-protocol agent archetypes in the harness
registry at import time:

  - ``archetype_sr``   — serial refinement (KernelBench G+E protocol)
  - ``archetype_bon``  — parallel best-of-N at temperature 1.0
  - ``archetype_evo``  — population/archive evolution (EvoEngineer-style)
  - ``archetype_band`` — UCB1 bandit over strategy arms (KernelBand-style)

All submit candidates through the canonical session tool protocol and are
model-agnostic via the pydantic_ai integration's model-string resolution.
"""

from __future__ import annotations

from compilagent.harness.registry import harness_registry

from .bandit import ArchetypeBanditHarness
from .evolution import ArchetypeEvolutionHarness
from .harness import ArchetypeBestOfNHarness, ArchetypeSerialRefinementHarness

if "archetype_sr" not in harness_registry.ids():
    harness_registry.register("archetype_sr", ArchetypeSerialRefinementHarness)
if "archetype_bon" not in harness_registry.ids():
    harness_registry.register("archetype_bon", ArchetypeBestOfNHarness)
if "archetype_evo" not in harness_registry.ids():
    harness_registry.register("archetype_evo", ArchetypeEvolutionHarness)
if "archetype_band" not in harness_registry.ids():
    harness_registry.register("archetype_band", ArchetypeBanditHarness)

__all__ = [
    "ArchetypeBanditHarness",
    "ArchetypeBestOfNHarness",
    "ArchetypeEvolutionHarness",
    "ArchetypeSerialRefinementHarness",
]
