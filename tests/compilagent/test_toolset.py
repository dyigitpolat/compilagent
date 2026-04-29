from __future__ import annotations

import pytest

from compilagent.core.tool_decl import ToolDecl
from compilagent.toolset import Toolset


def _decl(name: str, *, read_only: bool = True) -> ToolDecl:
    return ToolDecl(
        name=name,
        description=f"description for {name}",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _args: f"called {name}",
        read_only=read_only,
    )


def test_by_name_returns_decl():
    toolset = Toolset(tools=(_decl("a"), _decl("b")))
    assert toolset.by_name("a").name == "a"


def test_by_name_unknown_raises():
    toolset = Toolset(tools=(_decl("a"),))
    with pytest.raises(KeyError):
        toolset.by_name("missing")


def test_read_only_subset_filters():
    toolset = Toolset(tools=(_decl("a"), _decl("b", read_only=False)))
    subset = toolset.read_only_subset()
    assert subset.names() == ["a"]


def test_with_extra_appends_and_rejects_duplicates():
    base = Toolset(tools=(_decl("a"),))
    extended = base.with_extra((_decl("c"),))
    assert extended.names() == ["a", "c"]
    with pytest.raises(ValueError):
        base.with_extra((_decl("a"),))


def test_handler_dispatch():
    toolset = Toolset(tools=(_decl("a"),))
    assert toolset.by_name("a").handler({}) == "called a"
