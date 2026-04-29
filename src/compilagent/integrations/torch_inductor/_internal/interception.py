"""Per-compile context managers patching `torch._inductor.config`,
`torch._inductor.lowering.lowerings`, scheduler-pass hooks, and the choices
handler.

Each context manager snapshots the relevant state on enter, applies the plan's
overrides, restores on exit. They compose via `ExitStack` inside the custom
backend (`harness.py`).

Nothing in this module imports torch at import time; failures during compile
fall back to a no-op so the wider observation server doesn't crash on a
torch-less machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def patched_inductor_config(overrides: dict[str, Any]) -> Iterator[None]:
    """Apply `inductor.<knob> = value` for the duration of the block.

    `overrides` keys are dotted paths like `"inductor.max_fusion_size"` or
    bare leaf names like `"max_fusion_size"`; the leaf form is interpreted
    against `torch._inductor.config`.
    """

    if not overrides:
        yield
        return
    try:
        from torch._inductor import config as inductor_config  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        yield
        return
    snapshot: dict[str, Any] = {}
    for key, value in overrides.items():
        leaf = key.split(".", 1)[1] if key.startswith("inductor.") else key
        if not hasattr(inductor_config, leaf):
            continue
        snapshot[leaf] = getattr(inductor_config, leaf)
        try:
            setattr(inductor_config, leaf, value)
        except Exception:  # noqa: BLE001
            snapshot.pop(leaf, None)
    try:
        yield
    finally:
        for leaf, prev in snapshot.items():
            try:
                setattr(inductor_config, leaf, prev)
            except Exception:  # noqa: BLE001
                pass


@contextmanager
def patched_dynamo_config(overrides: dict[str, Any]) -> Iterator[None]:
    """Same shape as `patched_inductor_config` but against `torch._dynamo.config`."""

    if not overrides:
        yield
        return
    try:
        from torch._dynamo import config as dynamo_config  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        yield
        return
    snapshot: dict[str, Any] = {}
    for key, value in overrides.items():
        leaf = key.split(".", 1)[1] if key.startswith("dynamo.") else key
        if not hasattr(dynamo_config, leaf):
            continue
        snapshot[leaf] = getattr(dynamo_config, leaf)
        try:
            setattr(dynamo_config, leaf, value)
        except Exception:  # noqa: BLE001
            snapshot.pop(leaf, None)
    try:
        yield
    finally:
        for leaf, prev in snapshot.items():
            try:
                setattr(dynamo_config, leaf, prev)
            except Exception:  # noqa: BLE001
                pass


@contextmanager
def patched_lowering_registry(
    overrides: dict[str, Any],
) -> Iterator[None]:
    """Swap `torch._inductor.lowering.lowerings[op]` entries.

    Keys are aten op qualnames (`"aten.softmax"`); values are callables that
    will be installed for the duration of the block. Restores on exit.
    """

    if not overrides:
        yield
        return
    try:
        from torch._inductor import lowering as _lowering  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        yield
        return
    table = getattr(_lowering, "lowerings", None)
    if table is None:
        yield
        return
    snapshot: dict[Any, Any] = {}
    sentinel = object()
    for key, value in overrides.items():
        # Resolve "aten.softmax" -> torch.ops.aten.softmax (or skip if unknown)
        try:
            import torch  # type: ignore[import-not-found]
            if "." in key:
                ns, op = key.split(".", 1)
                target = getattr(getattr(torch.ops, ns), op)
            else:
                target = key
        except Exception:  # noqa: BLE001
            continue
        snapshot[target] = table.get(target, sentinel)
        try:
            table[target] = value
        except Exception:  # noqa: BLE001
            snapshot.pop(target, None)
    try:
        yield
    finally:
        for target, prev in snapshot.items():
            try:
                if prev is sentinel:
                    table.pop(target, None)
                else:
                    table[target] = prev
            except Exception:  # noqa: BLE001
                pass


@contextmanager
def patched_inductor_choices(choices_handler: Any | None) -> Iterator[None]:
    """Swap `V.choices_handler` (the autotune-config heuristic) for the block."""

    if choices_handler is None:
        yield
        return
    try:
        from torch._inductor.virtualized import V  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        yield
        return
    prev = getattr(V, "choices", None)
    try:
        V.set_choices_handler(choices_handler)
    except Exception:  # noqa: BLE001
        yield
        return
    try:
        yield
    finally:
        try:
            if prev is not None:
                V.set_choices_handler(prev)
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def scheduler_passes(
    *,
    pre_fusion: Any | None = None,
    post_fusion: Any | None = None,
) -> Iterator[None]:
    """Install `_pre_fusion_custom_pass` / `_post_fusion_custom_pass` for the block."""

    if pre_fusion is None and post_fusion is None:
        yield
        return
    try:
        from torch._inductor import config as inductor_config  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        yield
        return
    prev_pre = getattr(inductor_config, "_pre_fusion_custom_pass", None)
    prev_post = getattr(inductor_config, "_post_fusion_custom_pass", None)
    if pre_fusion is not None:
        inductor_config._pre_fusion_custom_pass = pre_fusion
    if post_fusion is not None:
        inductor_config._post_fusion_custom_pass = post_fusion
    try:
        yield
    finally:
        try:
            inductor_config._pre_fusion_custom_pass = prev_pre
            inductor_config._post_fusion_custom_pass = prev_post
        except Exception:  # noqa: BLE001
            pass
