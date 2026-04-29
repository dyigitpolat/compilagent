"""Backend-specific introspection tools the Triton backend exposes.

`list_compiler_passes` / `describe_compiler_pass` / `read_pass_source` give
the agent visibility into the Triton MLIR pass catalog. Surfaced via
`Backend.list_introspection_tools()` and merged into the canonical session
toolset in the agent's tool list.
"""

from __future__ import annotations

import json
from typing import Any

from compilagent.core.tool_decl import ToolDecl


def _pass_descriptor_to_dict(descriptor: Any) -> dict[str, Any]:
    return {
        "name": getattr(descriptor, "name", ""),
        "stage": getattr(descriptor, "stage", "") or "",
        "summary": getattr(descriptor, "summary", "") or "",
        "args": list(getattr(descriptor, "args", []) or ()),
        "capabilities": list(getattr(descriptor, "capabilities", []) or ()),
    }


def _list_compiler_passes(_args: dict) -> str:
    from ._internal.passes import PASS_CATALOG

    rows = [_pass_descriptor_to_dict(d) for d in PASS_CATALOG]
    rows.sort(key=lambda r: (r["stage"], r["name"]))
    return json.dumps({"passes": rows}, indent=2, default=str)


def _describe_compiler_pass(args: dict) -> str:
    from ._internal.passes import PASS_CATALOG

    name = str(args.get("name", "")).strip()
    if not name:
        raise ValueError("name is required")
    for descriptor in PASS_CATALOG:
        if getattr(descriptor, "name", "") == name:
            return json.dumps(_pass_descriptor_to_dict(descriptor), indent=2, default=str)
    known = sorted(getattr(d, "name", "") for d in PASS_CATALOG)
    raise ValueError(f"unknown pass `{name}`; first 10 known: {known[:10]}")


def list_triton_introspection_tools() -> tuple[ToolDecl, ...]:
    """Return the Triton-specific agent tools."""

    return (
        ToolDecl(
            name="list_compiler_passes",
            description=(
                "List all Triton MLIR passes the backend can intervene on. "
                "Each entry includes name, stage (`ttir`/`ttgir`/`llir`), and "
                "a one-line summary."
            ),
            args_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_list_compiler_passes,
            read_only=True,
        ),
        ToolDecl(
            name="describe_compiler_pass",
            description=(
                "Describe one Triton MLIR pass by name: stage, summary, "
                "argument schema, and any tracked capabilities."
            ),
            args_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact pass name (see list_compiler_passes).",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=_describe_compiler_pass,
            read_only=True,
        ),
    )
