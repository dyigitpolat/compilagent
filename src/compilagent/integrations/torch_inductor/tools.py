"""Backend-specific introspection tools the TorchInductor backend exposes."""

from __future__ import annotations

import json
from typing import Any

from compilagent.core.tool_decl import ToolDecl

_KNOB_CATALOG_CACHE: Any = None


def _catalog():
    global _KNOB_CATALOG_CACHE
    if _KNOB_CATALOG_CACHE is None:
        from ._internal.knobs import build_knob_catalog

        _KNOB_CATALOG_CACHE = build_knob_catalog()
    return _KNOB_CATALOG_CACHE


def _list_inductor_knobs(args: dict) -> str:
    namespace = str(args.get("namespace", "inductor")).strip() or "inductor"
    catalog = _catalog()
    knobs = catalog.in_namespace(namespace)
    return json.dumps(
        {
            "namespace": namespace,
            "count": len(knobs),
            "knobs": [k.serialize() for k in knobs],
        },
        indent=2,
        default=str,
    )


def _describe_inductor_knob(args: dict) -> str:
    name = str(args.get("name", "")).strip()
    if not name:
        raise ValueError("name is required")
    catalog = _catalog()
    knob = catalog.by_name(name)
    if knob is None:
        raise ValueError(f"unknown knob `{name}`")
    return json.dumps(knob.serialize(), indent=2, default=str)


def list_inductor_introspection_tools() -> tuple[ToolDecl, ...]:
    return (
        ToolDecl(
            name="list_inductor_knobs",
            description=(
                "List torch._inductor / torch._dynamo config knobs the agent "
                "can override per candidate. Each entry includes the default "
                "and heuristic candidate values."
            ),
            args_schema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Knob namespace to filter (default `inductor`).",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=_list_inductor_knobs,
            read_only=True,
        ),
        ToolDecl(
            name="describe_inductor_knob",
            description="Describe one Inductor / Dynamo knob by name (dotted or leaf).",
            args_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Knob name (e.g. `inductor.max_autotune`).",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=_describe_inductor_knob,
            read_only=True,
        ),
    )
