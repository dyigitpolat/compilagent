"""Banned-API lint for `triton_source` candidates (correctness gate g4).

The decision space of the `triton_source` backend is full kernel source: the
agent submits a python module defining `ModelNew` whose forward path must
implement the reference computation with `@triton.jit` kernels. The cheapest
way to "pass" is to call the reference torch op (`torch.softmax`,
`F.layer_norm`, `a @ b`, ...) — this lint rejects that *before* the candidate
ever reaches the GPU sandbox.

Each workload carries its own banned-pattern list in
`WorkloadSpec.metadata["banned_patterns"]`. A pattern is either:

  - the literal string ``"@"`` — bans the matmul *operator* (`ast.MatMult`;
    decorators are unaffected, they are not `BinOp`s), or
  - a (possibly dotted) name such as ``"torch.softmax"`` or ``"layer_norm"``.
    Matching is by **terminal segment**: a call whose dotted function name
    ends in the pattern's last segment is flagged. This catches
    ``torch.softmax``, ``F.softmax``, ``torch.nn.functional.softmax`` and the
    method form ``x.softmax(dim=1)`` with one pattern.

Scoping rules (all AST-based, no execution):

  - ``__init__`` method bodies are EXEMPT — candidates legitimately
    instantiate e.g. ``nn.LayerNorm(1024)`` there to mirror the reference's
    parameter names so ``load_state_dict`` succeeds.
  - Functions decorated with ``@triton.jit`` are EXEMPT — ``tl.sigmoid`` /
    ``tl.dot`` inside a kernel are exactly what the agent should write.
  - Calls rooted at ``tl.`` / ``triton.`` are EXEMPT everywhere.
  - Banned modules stashed on ``self`` in ``__init__`` (e.g.
    ``self.ln = nn.LayerNorm(...)``) are tracked: calling ``self.ln(x)``
    outside ``__init__`` is flagged too.

Known limitation (documented for the RQ5 loose-verifier study): the lint is
a name-based heuristic, not a sandbox. Deeper laundering (importlib,
getattr-by-string) is out of scope for E2a and is caught — if at all — only
by the speed/correctness gates.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass

_EXEMPT_CALL_ROOTS = frozenset({"tl", "triton"})


@dataclass(frozen=True, slots=True)
class LintViolation:
    """One banned-API hit, with enough context for the agent to react."""

    pattern: str
    detail: str
    lineno: int

    @property
    def message(self) -> str:
        return (
            f"banned API `{self.detail}` (matches banned pattern "
            f"`{self.pattern}` for this workload) at line {self.lineno} — "
            "implement this computation with your own @triton.jit kernel "
            "instead of calling the reference torch op."
        )


def _dotted_name(node: ast.AST) -> str | None:
    """Reconstruct `a.b.c` from an Attribute/Name chain; None otherwise."""

    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _is_jit_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = _dotted_name(target) or ""
        if name == "jit" or name.endswith(".jit"):
            return True
    return False


def _banned_self_attrs(
    tree: ast.Module, terminals: frozenset[str]
) -> dict[str, str]:
    """Map `self.<attr>` names assigned a banned constructor in any __init__.

    Catches the classic evasion `self.ln = nn.LayerNorm(...)` (exempt in
    __init__) followed by `self.ln(x)` in forward.
    """

    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "__init__"):
            continue
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign):
                continue
            if not isinstance(stmt.value, ast.Call):
                continue
            ctor = _dotted_name(stmt.value.func) or ""
            terminal = ctor.rsplit(".", 1)[-1]
            if terminal not in terminals:
                continue
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    out[target.attr] = ctor
    return out


def lint_banned_apis(
    source: str, banned_patterns: Sequence[str]
) -> list[LintViolation]:
    """Return every banned-API violation in `source` (empty list == clean)."""

    patterns = [p for p in banned_patterns if p]
    if not patterns:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            LintViolation(
                pattern="<syntax>",
                detail=f"module does not parse: {exc.msg}",
                lineno=exc.lineno or 0,
            )
        ]

    ban_matmul_op = "@" in patterns
    terminal_to_pattern: dict[str, str] = {
        p.rsplit(".", 1)[-1]: p for p in patterns if p != "@"
    }
    terminals = frozenset(terminal_to_pattern)
    self_aliases = _banned_self_attrs(tree, terminals)

    violations: list[LintViolation] = []

    def _check_call(node: ast.Call) -> None:
        name = _dotted_name(node.func)
        if name is None:
            return
        root = name.split(".", 1)[0]
        if root in _EXEMPT_CALL_ROOTS:
            return
        terminal = name.rsplit(".", 1)[-1]
        if terminal in terminals:
            violations.append(
                LintViolation(
                    pattern=terminal_to_pattern[terminal],
                    detail=name,
                    lineno=node.lineno,
                )
            )
            return
        # `self.ln(x)` where __init__ did `self.ln = nn.LayerNorm(...)`.
        if (
            name.startswith("self.")
            and name.count(".") == 1
            and name.split(".")[1] in self_aliases
        ):
            attr = name.split(".")[1]
            violations.append(
                LintViolation(
                    pattern=terminal_to_pattern[
                        self_aliases[attr].rsplit(".", 1)[-1]
                    ],
                    detail=f"{name} (assigned {self_aliases[attr]} in __init__)",
                    lineno=node.lineno,
                )
            )

    # Iterative walk with per-node exemption state. A node is exempt when it
    # lives inside an `__init__` body or a `@triton.jit`-decorated function.
    stack: list[tuple[ast.AST, bool]] = [(tree, False)]
    while stack:
        node, exempt = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "__init__" or _is_jit_decorated(node):
                exempt = True
        if not exempt:
            if isinstance(node, ast.Call):
                _check_call(node)
            elif (
                ban_matmul_op
                and isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.MatMult)
            ):
                violations.append(
                    LintViolation(
                        pattern="@",
                        detail="`@` matmul operator",
                        lineno=node.lineno,
                    )
                )
        for child in ast.iter_child_nodes(node):
            stack.append((child, exempt))

    violations.sort(key=lambda v: v.lineno)
    return violations
