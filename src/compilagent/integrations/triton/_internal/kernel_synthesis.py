"""Kernel-source synthesizer.

Lets the agent rewrite a kernel's `.py` file (e.g. add `tl.multiple_of`,
`cache_modifier`, `eviction_policy` annotations to specific loads/stores),
re-imports the kernel, and exposes the resulting Triton handle. The original
file is restored on exit so a candidate's edits never leak into the next
candidate's run.

The synthesizer is intentionally text-level — it doesn't try to AST-rewrite
Python. The agent supplies a list of (old_text, new_text) substitutions; we
apply them in order, abort on no-change, and re-AST-parse the file as a sanity
check before re-import. If the file fails to parse, the original is restored
and a `SynthesizeError` is raised with the parse diagnostic.
"""

from __future__ import annotations

import ast
import importlib
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class SynthesizeError(RuntimeError):
    """Raised when a kernel-source rewrite cannot be safely applied."""


@dataclass(slots=True)
class TextEdit:
    """One literal substitution to apply to the kernel source."""

    find: str
    replace: str
    expect_count: int | None = None       # if set, must match this many times exactly

    def apply(self, text: str) -> tuple[str, int]:
        if not self.find:
            raise SynthesizeError("TextEdit.find must not be empty")
        count = text.count(self.find)
        if self.expect_count is not None and count != self.expect_count:
            raise SynthesizeError(
                f"expected {self.expect_count} occurrence(s) of `{self.find!r}`, found {count}"
            )
        if count == 0:
            raise SynthesizeError(f"text `{self.find!r}` not found in kernel source")
        return text.replace(self.find, self.replace), count


@dataclass(slots=True)
class SynthesizeResult:
    source_path: Path
    edits: tuple[TextEdit, ...]
    diff_chars: int                         # rough size of the rewrite
    edits_applied: int


@contextmanager
def synthesize_kernel(
    source_path: str | Path,
    edits: list[TextEdit] | tuple[TextEdit, ...],
    *,
    module_name_hint: str | None = None,
) -> Iterator[tuple[SynthesizeResult, object]]:
    """Apply `edits` to `source_path`, re-import, yield (result, module).

    On exit the file is restored to its original contents and the cached
    Python module entry is invalidated so subsequent imports see the original
    source again. Use as:

        with synthesize_kernel(path, [TextEdit(find=..., replace=...)]) as (result, module):
            kernel = module.vector_add_kernel
            ...
    """

    path = Path(source_path).resolve()
    if not path.is_file():
        raise SynthesizeError(f"kernel source not found: {path}")

    original = path.read_text(encoding="utf-8")
    rewritten = original
    applied = 0
    for edit in edits:
        rewritten, n = edit.apply(rewritten)
        applied += n

    # Validate the rewritten source parses as Python.
    try:
        ast.parse(rewritten)
    except SyntaxError as exc:
        raise SynthesizeError(f"rewritten source does not parse: {exc}") from exc

    # Snapshot the file in a backup, write the rewrite, drop any cached module
    # entry so the next import re-reads the file.
    backup_dir = Path(tempfile.mkdtemp(prefix="compilagent-synth-"))
    backup_path = backup_dir / path.name
    shutil.copy2(path, backup_path)

    # Resolve the dotted import name from the package layout (mirrors
    # `compiler._resolve_package_module_name`).
    parts = [path.stem]
    parent = path.parent
    while (parent / "__init__.py").exists():
        parts.append(parent.name)
        parent = parent.parent
    qualname = (
        module_name_hint
        if module_name_hint is not None
        else (".".join(reversed(parts)) if len(parts) > 1 else None)
    )

    try:
        path.write_text(rewritten, encoding="utf-8")
        if qualname and qualname in sys.modules:
            del sys.modules[qualname]
        module = importlib.import_module(qualname) if qualname else None
        result = SynthesizeResult(
            source_path=path,
            edits=tuple(edits),
            diff_chars=abs(len(rewritten) - len(original)),
            edits_applied=applied,
        )
        yield result, module
    finally:
        # Restore the original file + cached module.
        try:
            shutil.copy2(backup_path, path)
        except Exception:  # noqa: BLE001
            pass
        if qualname and qualname in sys.modules:
            try:
                del sys.modules[qualname]
            except Exception:  # noqa: BLE001
                pass
        try:
            shutil.rmtree(backup_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
