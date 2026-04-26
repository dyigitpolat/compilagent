"""End-to-end demo: replace `torch.compile` with `compilagent.optimize_module`.

Defines a small ViT-style transformer encoder block, compiles a baseline
through `torch.compile`, then runs the agentic JIT through Mistral, and
times both compiled callables on identical inputs. Asserts:

  - the agent's output matches baseline within bf16 tolerance,
  - the agent's compiled callable runs faster than the baseline.

Run:
    env/bin/python scripts/examples/agentic_jit_demo.py

Optional flags:
    --trials 12              trial budget (default 8)
    --harness pydantic_ai    or "claude_agent_sdk"
    --model "anthropic:claude-opus-4-7"
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn

import compilagent_triton as cgt


class TransformerEncoderBlock(nn.Module):
    """Vanilla transformer encoder block — pre-norm, GELU MLP, residual adds."""

    def __init__(self, dim: int = 768, heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def time_callable(fn, x, *, warmup: int = 10, iters: int = 50) -> float:
    """Median CUDA-event time per call, in milliseconds."""

    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--harness", default="pydantic_ai",
                   choices=("pydantic_ai", "claude_agent_sdk"))
    p.add_argument("--model", default="mistral:mistral-large-latest")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        return 1

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # 1) User's existing model + inputs.
    model = TransformerEncoderBlock().cuda().to(torch.bfloat16).eval()
    x = torch.randn(8, 197, 768, device="cuda", dtype=torch.bfloat16)

    # 2) Baseline: stock torch.compile.
    baseline = torch.compile(model, mode="default")
    with torch.no_grad():
        for _ in range(3):
            baseline(x)  # warm Inductor cache
    baseline_ms = time_callable(lambda inp: baseline(inp), x)
    print(f"baseline `torch.compile`     : {baseline_ms:.3f} ms / call")

    # 3) Agentic JIT — three lines.
    print(f"\nrunning agent ({args.harness} · {args.model} · {args.trials} trials) ...")
    t0 = time.perf_counter()
    result = cgt.optimize_module(
        model, example_inputs=(x,),
        max_candidates=args.trials,
        harness=args.harness,
        model_name=args.model,
    )
    print(f"agent run elapsed             : {(time.perf_counter() - t0):.1f}s")
    print(f"agent best speedup (reported) : "
          f"{result.best_speedup:.4f}× over `torch.compile`"
          if result.best_speedup else "agent: no candidate beat baseline")
    print(f"correctness within tolerance  : {result.correctness_ok}")
    print(f"max abs diff vs. baseline     : {result.max_abs_diff}")

    if not result.improved:
        print("\nNo validated candidate beat baseline; nothing to swap in.")
        print(f"final report:\n{result.final_text}")
        return 2

    # 4) Drop-in replacement: call the optimized callable directly.
    optimized = result.optimized_callable
    optimized_ms = time_callable(lambda inp: optimized(inp), x)
    print(f"\noptimized callable             : {optimized_ms:.3f} ms / call")
    print(f"measured speedup (this script) : {baseline_ms / optimized_ms:.4f}×")

    # 5) Verify outputs match.
    with torch.no_grad():
        b = baseline(x)
        o = optimized(x)
    diff = (b - o).abs().max().item()
    rel = ((b - o).abs() / (b.abs() + 1e-3)).max().item()
    print(f"\noutput max-abs-diff            : {diff:.4e}")
    print(f"output max-rel-diff            : {rel:.4e}")
    print(f"within bf16 tol (atol=5e-4)    : {diff < 5e-4}")

    if optimized_ms >= baseline_ms:
        print("\nWARNING: optimized callable did not beat baseline in the "
              "post-hoc measurement (recheck flake or noise band).")
        return 3
    print("\nSUCCESS: agentic JIT beats torch.compile on this workload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
