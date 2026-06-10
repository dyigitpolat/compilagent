"""Archetype harness integration (ticket E4).

Self-registers two published-protocol agent archetypes in the harness
registry at import time:

  - ``archetype_sr``  — serial refinement (KernelBench G+E protocol)
  - ``archetype_bon`` — parallel best-of-N at temperature 1.0

Both submit candidates through the canonical session tool protocol and are
model-agnostic via the pydantic_ai integration's model-string resolution.
"""

from __future__ import annotations

from compilagent.harness.registry import harness_registry

from .harness import ArchetypeBestOfNHarness, ArchetypeSerialRefinementHarness

if "archetype_sr" not in harness_registry.ids():
    harness_registry.register("archetype_sr", ArchetypeSerialRefinementHarness)
if "archetype_bon" not in harness_registry.ids():
    harness_registry.register("archetype_bon", ArchetypeBestOfNHarness)

__all__ = [
    "ArchetypeBestOfNHarness",
    "ArchetypeSerialRefinementHarness",
]
