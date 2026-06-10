"""Prompt builders for the archetype harnesses (ticket E4).

The base prompt + in-context Triton exemplar MIRROR the P0 capability probe
(`papers/compilagent_iccd_2026/research_artifacts/p0_probe/probe.py`) word
for word, so archetype results stay comparable with the probe's one-shot /
serial numbers. The only addition is one rule line naming the workload's
lint-enforced banned patterns (gate g4), so the agent can react to the
enforcement the probe only stated informally.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

EXEMPLAR = textwrap.dedent('''
    Example of a complete Triton-kernel module (vector add) in the required format:
    ```python
    import torch, torch.nn as nn
    import triton, triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

    class ModelNew(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, x, y):
            out = torch.empty_like(x)
            n = x.numel()
            grid = (triton.cdiv(n, 1024),)
            add_kernel[grid](x, y, out, n, BLOCK=1024)
            return out
    ```
''')


def base_prompt(
    *,
    reference_source: str,
    task_description: str,
    banned_patterns: list[str],
) -> str:
    """The P0-probe prompt, parameterised by workload."""

    banned_line = (
        f"- Banned torch APIs for this task (enforced by an AST lint): "
        f"{', '.join(banned_patterns)}.\n"
        if banned_patterns
        else ""
    )
    return f"""You optimize PyTorch programs by writing custom Triton GPU kernels.

Reference implementation (PyTorch eager):
```python
{reference_source}
```
Task: {task_description}

Write a complete, self-contained Python module that defines `ModelNew`, a drop-in replacement
for `Model` with the same `__init__` and `forward` signatures, where the core computation is
performed by Triton kernel(s) you write (`@triton.jit`). Rules:
- Use Triton for the main computation; do NOT call the equivalent torch op (e.g. torch.softmax,
  nn.LayerNorm, torch.matmul) inside forward.
{banned_line}- Keep numerics in float32. Same output dtype/shape as the reference.
- If the module has parameters (e.g. LayerNorm weight/bias), ModelNew must define identical
  parameter names/shapes so state can be copied.
- Target: NVIDIA Blackwell GPU, Triton 3.6.
- Output exactly ONE fenced python code block with the full module, nothing else.
{EXEMPLAR}"""


RETRY_FORMAT_MESSAGE = (
    "Output exactly one fenced python code block with the full module."
)


#: Named optimization directions used for menu-dropout prompt variation
#: (archetype_evo / cascade seed rounds): each parallel sample sees a random
#: subset, pushing the seeds toward different regions of the design space.
OPTIMIZATION_MENU: tuple[str, ...] = (
    "Vectorize memory access: process many elements per program via "
    "tl.arange blocks and wide loads/stores.",
    "Tune tile/block sizes (BLOCK along each dim) and num_warps for "
    "occupancy on this shape.",
    "Fuse multiple passes over memory into a single-pass kernel "
    "(e.g. online max+sum for reductions).",
    "Minimize global-memory traffic: keep intermediates in registers, "
    "avoid materializing temporaries.",
    "Re-order loops / re-layout accesses so global loads are coalesced "
    "along the contiguous dimension.",
    "Hide latency with software pipelining (num_stages) and prefetching.",
)


def menu_dropout_prompt(
    *,
    reference_source: str,
    task_description: str,
    banned_patterns: list[str],
    rng: Any,
    keep_probability: float = 0.5,
) -> str:
    """`base_prompt` + a randomly-thinned optimization menu.

    `rng` is a `random.Random`-like object (seeded per run for
    reproducibility of the variation pattern; the LLM sampling itself is
    not seedable through the API).
    """

    kept = [item for item in OPTIMIZATION_MENU if rng.random() < keep_probability]
    if not kept:
        kept = [rng.choice(OPTIMIZATION_MENU)]
    menu = "\n".join(f"- {item}" for item in kept)
    return (
        base_prompt(
            reference_source=reference_source,
            task_description=task_description,
            banned_patterns=banned_patterns,
        )
        + f"\nFocus especially on these optimization directions:\n{menu}\n"
    )


def extract_code(text: str) -> str | None:
    """Largest fenced python block, exactly as the P0 probe parses."""

    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len).strip() if blocks else None


def rejection_feedback(error: str) -> str:
    """Feedback when propose_candidate rejects the submission outright."""

    return (
        f"Your submission was rejected before reaching the GPU: {error}\n"
        "Fix it and output the full corrected module in one python code block."
    )


def feedback_for_run_result(result: dict[str, Any]) -> str:
    """Structured verdict feedback per the KernelBench G+E protocol:
    compile error / failing gate + message / 'correct but slower'."""

    if not result.get("compile_ok"):
        return (
            "Your kernel failed to run. Error:\n"
            f"{result.get('compile_diagnostics')}\n"
            "Fix it and output the full corrected module in one python code block."
        )
    if result.get("correctness_ok") is False:
        gate_lines = [
            w
            for w in (result.get("compile_warnings") or [])
            if "FAILED" in str(w)
        ]
        detail = (
            " ".join(gate_lines)
            or f"max_abs_diff={result.get('max_abs_diff')}"
        )
        return (
            "Your kernel ran but FAILED a correctness gate vs the reference: "
            f"{detail} Fix correctness and output the full corrected module "
            "in one python code block."
        )
    cand_ms = result.get("median_ms")
    speedup = result.get("speedup_vs_baseline")
    ref_ms = (
        cand_ms * speedup
        if isinstance(cand_ms, (int, float)) and isinstance(speedup, (int, float))
        else None
    )
    if cand_ms is None or ref_ms is None:
        return (
            "Your kernel ran but produced no timing signal. Propose an "
            "improved module in one python code block."
        )
    return (
        f"Your kernel is CORRECT but its latency is {cand_ms:.4f} ms vs eager "
        f"{ref_ms:.4f} ms (speedup {speedup:.3f}x). Make it faster (better "
        "tiling/block sizes, vectorization, fewer passes over memory, "
        "num_warps/num_stages) and output the full improved module in one "
        "python code block."
    )
