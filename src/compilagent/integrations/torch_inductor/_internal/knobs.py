"""Typed catalog of TorchInductor / TorchDynamo configuration knobs.

The catalog is built at import time by walking `torch._inductor.config` and
`torch._dynamo.config` — there is **no hand-coded knob list anywhere**.
For each public attribute we record:

  - canonical dotted path (`inductor.max_fusion_size`)
  - Python type
  - default value (the value found at import)
  - inferred range/candidate set:
      * `bool`  → `BooleanFlag`
      * `int`   → `IntRange` with a heuristic neighborhood around the default
      * `float` → `FloatRange` with same idea
      * `str`/`Literal[...]` → `EnumChoice` (string variants encountered in source)
      * other (callables, lists, dicts) → `StructuredJsonRange` example

Backends consume this catalog through `derivation/inductor_knobs.py` to emit
levers; the agent calls `list_inductor_knobs` / `describe_inductor_knob` to
inspect it.
"""

from __future__ import annotations

import inspect as _inspect
from dataclasses import dataclass, field
from typing import Any, Literal, get_args, get_origin


@dataclass(frozen=True, slots=True)
class KnobDescriptor:
    """One configuration knob exposed by a torch sub-config module."""

    name: str                       # canonical: "inductor.max_fusion_size"
    namespace: str                  # "inductor" | "dynamo"
    leaf: str                       # "max_fusion_size"
    py_type: str                    # str / int / bool / float / str(Literal[...])
    default: Any
    candidates: tuple[Any, ...] = ()    # plausible alternate values
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "leaf": self.leaf,
            "py_type": self.py_type,
            "default": self.default,
            "candidates": list(self.candidates),
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class KnobCatalog:
    """Process-wide collection of all introspected knobs."""

    knobs: tuple[KnobDescriptor, ...]

    def by_name(self, name: str) -> KnobDescriptor | None:
        for k in self.knobs:
            if k.name == name or k.leaf == name:
                return k
        return None

    def in_namespace(self, namespace: str) -> tuple[KnobDescriptor, ...]:
        return tuple(k for k in self.knobs if k.namespace == namespace)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _candidates_for(default: Any, py_type: type | None) -> tuple[Any, ...]:
    """Heuristic neighborhood for a knob's value.

    For ints around `n`, suggest `{n//4, n//2, n, n*2, n*4}` clamped to >=1.
    For bools, suggest the opposite. For strings, suggest no alternates here —
    enums are populated by the optional Literal-walk below.
    """

    if isinstance(default, bool):
        return (default, not default)
    if isinstance(default, int):
        n = max(1, default)
        candidates = sorted({max(1, n // 4), max(1, n // 2), n, n * 2, n * 4})
        return tuple(candidates)
    if isinstance(default, float):
        n = max(1e-9, abs(default))
        return (default * 0.5, default, default * 2.0)
    return (default,)


def _annotation_to_pytype(annotation: Any) -> str:
    if annotation is None:
        return "any"
    try:
        return str(annotation).replace("typing.", "")
    except Exception:  # noqa: BLE001
        return "any"


def _enum_candidates(annotation: Any) -> tuple[Any, ...] | None:
    """Extract Literal[...] members if the annotation is one."""

    if annotation is None:
        return None
    origin = get_origin(annotation)
    if origin is Literal:
        return tuple(get_args(annotation))
    # Union/Optional handling
    args = get_args(annotation)
    for arg in args:
        if get_origin(arg) is Literal:
            return tuple(get_args(arg))
    return None


def _walk_config_module(mod: Any, namespace: str) -> list[KnobDescriptor]:
    """Walk a torch sub-config module and emit one descriptor per public knob.

    Torch 2.4+ stores its configuration entries in a private `_config` dict on
    the module — keyed by leaf name, valued as `_ConfigEntry(default, value_type, ...)`.
    That dict is the canonical source of truth (≈540 inductor entries on
    torch 2.11), much more complete than `dir(mod)` or `__annotations__`.
    """

    descriptors: list[KnobDescriptor] = []
    config_dict = getattr(mod, "_config", None)
    annotations: dict[str, Any] = getattr(mod, "__annotations__", {}) or {}

    typing_globals = {
        "Any", "Callable", "Literal", "Optional", "Union",
        "List", "Dict", "Tuple", "Set", "Iterable", "Mapping",
        "TYPE_CHECKING", "cast",
    }
    skip = typing_globals | {"Config", "is_fbcode"}

    if isinstance(config_dict, dict):
        # Preferred path: walk the typed config dict.
        for attr_name, entry in sorted(config_dict.items()):
            if attr_name.startswith("_") or attr_name in skip:
                continue
            if "." in attr_name:
                # Nested namespace key (e.g. "cuda.use_fast_math") — skipped.
                continue
            if getattr(entry, "hide", False):
                continue
            default = getattr(entry, "default", None)
            value_type = getattr(entry, "value_type", None) or type(default)
            try:
                # Resolve the *current* runtime value (env / user_override may have
                # adjusted it). Falls back to default on any failure.
                value = getattr(mod, attr_name)
            except Exception:  # noqa: BLE001
                value = default
            py_type = getattr(value_type, "__name__", str(value_type))
            annotation = annotations.get(attr_name)
            enum_candidates = _enum_candidates(annotation) if annotation is not None else None
            if enum_candidates is not None:
                candidates: tuple[Any, ...] = tuple(enum_candidates)
            else:
                candidates = _candidates_for(value, value_type)
            descriptors.append(
                KnobDescriptor(
                    name=f"{namespace}.{attr_name}",
                    namespace=namespace,
                    leaf=attr_name,
                    py_type=py_type,
                    default=value,
                    candidates=candidates,
                    description="",
                )
            )
        return descriptors

    # Fallback for older torch / non-Config-module shapes.
    for attr_name in sorted(set(dir(mod)) | set(annotations)):
        if attr_name.startswith("_") or attr_name in skip:
            continue
        try:
            value = getattr(mod, attr_name)
        except Exception:  # noqa: BLE001
            continue
        if callable(value) or _inspect.ismodule(value):
            continue
        if type(value).__name__ in {"SubConfigProxy", "ConfigModule"}:
            continue
        annotation = annotations.get(attr_name)
        py_type = _annotation_to_pytype(annotation) if annotation is not None else type(value).__name__
        enum_candidates = _enum_candidates(annotation) if annotation is not None else None
        if enum_candidates is not None:
            candidates = tuple(enum_candidates)
        else:
            candidates = _candidates_for(value, type(value))
        descriptors.append(
            KnobDescriptor(
                name=f"{namespace}.{attr_name}",
                namespace=namespace,
                leaf=attr_name,
                py_type=py_type,
                default=value,
                candidates=candidates,
                description="",
            )
        )
    return descriptors


def build_knob_catalog() -> KnobCatalog:
    """Walk torch's config modules and produce a typed catalog.

    Imports are lazy + tolerant: if torch isn't installed, the catalog comes
    out empty — the runtime treats that as "no torch backend available."
    """

    knobs: list[KnobDescriptor] = []
    for mod_path, ns in (
        ("torch._inductor.config", "inductor"),
        ("torch._dynamo.config", "dynamo"),
    ):
        try:
            mod = __import__(mod_path, fromlist=["*"])
        except Exception:  # noqa: BLE001
            continue
        try:
            knobs.extend(_walk_config_module(mod, ns))
        except Exception:  # noqa: BLE001
            continue
    return KnobCatalog(knobs=tuple(knobs))
