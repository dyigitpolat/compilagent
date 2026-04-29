"""`Toolset` — an immutable collection of `ToolDecl` records.

Sessions build a single `Toolset` once and pass it to a harness adapter.
Harness adapters never construct `ToolDecl`s themselves; they only consume
them. This is the seam that prevents tool definitions from drifting between
pydantic-ai and the Claude Agent SDK.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from compilagent.core.tool_decl import ToolDecl


@dataclass(frozen=True, slots=True)
class Toolset:
    tools: tuple[ToolDecl, ...] = ()

    def by_name(self, name: str) -> ToolDecl:
        for tool in self.tools:
            if tool.name == name:
                return tool
        known = sorted(t.name for t in self.tools)
        raise KeyError(f"Unknown tool `{name}`. Registered: {known or '(none)'}.")

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def read_only_subset(self) -> Toolset:
        return Toolset(tools=tuple(t for t in self.tools if t.read_only))

    def with_extra(self, extra: Iterable[ToolDecl]) -> Toolset:
        existing = {t.name for t in self.tools}
        merged: list[ToolDecl] = list(self.tools)
        for tool in extra:
            if tool.name in existing:
                raise ValueError(
                    f"Cannot extend toolset: tool `{tool.name}` is already registered."
                )
            existing.add(tool.name)
            merged.append(tool)
        return Toolset(tools=tuple(merged))
