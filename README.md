# Compilagent Triton

Agentic compiler optimization for PyTorch + Triton. Drop-in replacement for `torch.compile` / `@triton.jit` that runs an LLM-driven search over **real compiler heuristics** (MLIR pass pipeline, Inductor scheduler decisions, FX rewrites, lowering registry overrides) — not user-tunable knobs like `BLOCK_SIZE` or `num_warps`.

> **Building your own integration?** A new compiler, agent runtime, or workload can plug into the `compilagent` core without forking this repo. See [docs/integration_guide.md](docs/integration_guide.md).

## Quickstart — replace `torch.compile` with agentic JIT

```python
import torch
import compilagent_triton as cgt

model = MyTransformerBlock().cuda().eval()      # any nn.Module
x = torch.randn(8, 197, 768, device="cuda", dtype=torch.bfloat16)

# Before: stock torch.compile
baseline = torch.compile(model)
y_baseline = baseline(x)

# After: agentic JIT — three lines
result = cgt.optimize_module(model, example_inputs=(x,), max_candidates=8)
optimized = result.optimized_callable          # drop-in replacement
y_optimized = optimized(x)                     # same shape, same dtype, faster

print(f"{result.best_speedup:.3f}× speedup, correctness ok: {result.correctness_ok}")
```

That's it. The agent compiles a baseline through `torch.compile`, derives a search space from the FX graph + Inductor knob catalog, proposes multi-knob candidates (e.g. `shape_padding + max_autotune + coordinate_descent_tuning`), benchmarks each, validates correctness against the baseline output, and hands you back the **fastest validated callable**. If no candidate beat baseline, `result.optimized_callable` is `None` and `result.improved` is `False` — the caller falls back to their original code path.

End-to-end demo (baseline timing → agent → optimized timing → numerical check):

```bash
env/bin/python scripts/examples/agentic_jit_demo.py
```

## Quickstart — Triton kernel

```python
import torch, triton, triton.language as tl
import compilagent_triton as cgt

@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) +
                              tl.load(y_ptr + offs, mask=mask), mask=mask)

n = 1 << 20
x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
out = torch.empty_like(x)

result = cgt.optimize_kernel(
    my_kernel,
    args=(x, y, out, n),
    grid=lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),),
    constexpr={"BLOCK_SIZE": 1024},
    max_candidates=8,
)
print(f"{result.best_speedup:.3f}× over the Triton-default pass pipeline")
```

The agent operates on the MLIR pass pipeline (`tritongpu-coalesce`, `-accelerate-matmul`, `-pipeline`, `-optimize-thread-locality`, etc.) — skipping, reordering, parameterizing passes — never on `BLOCK_SIZE` or `num_warps` (those are user inputs, not compiler decisions).

## Configuration

`compilagent_triton` reads settings from `.env`. Minimum:

```bash
ANTHROPIC_API_KEY=...                # or MISTRAL_API_KEY / OPENAI_API_KEY
COMPILAGENT_MODEL=anthropic:claude-opus-4-7
```

Pass `model_name=` and `harness=` directly to override per-call:

```python
cgt.optimize_module(
    model, example_inputs,
    max_candidates=12,
    model_name="mistral:mistral-large-latest",
    harness="pydantic_ai",               # or "claude_agent_sdk"
)
```

**Harness × model compatibility:**
- `harness="pydantic_ai"` works with any provider — `anthropic:`, `mistral:`, `openai:`.
- `harness="claude_agent_sdk"` is the `claude` CLI under the hood and **only routes to Anthropic models** (e.g. `anthropic:claude-opus-4-7`); pass any other provider and it raises a clear error up front.

## How it works

1. **Baseline** — your callable is compiled through its native backend (`torch.compile` for `nn.Module`, `triton.JITFunction.__getitem__` for kernels) and timed with CUDA events.
2. **Analysis** — the backend introspects the produced artifacts: FX graph + Inductor `output_code.py` + scheduler / fusion logs (PyTorch path), or TTGIR + per-pass IR diffs (Triton path).
3. **Search-space derivation** — a backend-specific plugin set produces a list of *levers* with auto-bounded ranges and `evidence` tags linking each lever to a signal in the IR. No hand-coded knob list anywhere.
4. **Agent loop** — the agent inspects the workload + search space, forms named hypotheses, registers multi-knob candidates as a batch, runs them, reads correctness + speedup back, synthesizes across the batch, and proposes the next batch. Failed compiles don't count against the budget.
5. **Validation** — every reported speedup is verified by running the candidate's compiled output against the baseline output under per-dtype tolerances. Drift outside tolerance is treated as a failed candidate.

## Effectiveness study

A reproducible study driver lives in `scripts/experiments/`. It sweeps:

  - harnesses (`pydantic_ai`, `claude_agent_sdk`),
  - workloads (`vit_block`, `vector_add`),
  - trial budgets (`4`, `8`, `12`, `16`, `20`),
  - 3 RNG seeds each — for mean ± stddev error bars.

```bash
# Run the full grid (60 cells; takes a while)
env/bin/python scripts/experiments/run_study.py

# Or smoke-test the pipeline
env/bin/python scripts/experiments/run_study.py --quick

# Plot:
env/bin/python scripts/experiments/plot_study.py runs/study/<timestamp>/results.jsonl
```

The plotter writes four PNGs:

- `speedup_vs_trials.png` — best speedup vs trial budget
- `correctness_rate.png` — fraction of seeds where the *independent* recheck passed and the speedup beat 1.0
- `successful_per_trial.png` — successful_count / max_candidates (budget efficiency)
- `elapsed_vs_trials.png` — wall-clock per run

## Project layout

All integration code lives under `src/compilagent_triton/` (legacy) and `src/compilagent/` (new core + Phase-2 integrations). External dependencies — `triton`, `pydantic-acp`, etc. — are pulled from PyPI; the repo no longer carries upstream submodules. If you need to patch an upstream library, vendor only the changed files into the relevant integration package or open a PR upstream.

## Environment

Use the top-level virtual environment:

```bash
source env/bin/activate
python -m pip install -e ".[dev]"
```

For GPU benchmark experiments, install the optional GPU dependencies into the same `env`:

```bash
python -m pip install -e ".[gpu]"
```

Local runs load model configuration from `.env`. The API key is never printed or persisted by the package.

Useful settings:

- `ANTHROPIC_API_KEY`: Anthropic credential.
- `COMPILAGENT_MODEL`: defaults to `anthropic:claude-opus-4-7`.
- `COMPILAGENT_REASONING_EFFORT`: defaults to `extra_high`.
- `COMPILAGENT_MAX_TOKENS`: defaults to `8192`.
- `COMPILAGENT_TEMPERATURE`: defaults to `0.2`.
- `COMPILAGENT_MAX_CANDIDATES`: defaults to `4`.
- `COMPILAGENT_MAX_BENCHMARK_SECONDS`: defaults to `120`.
- `COMPILAGENT_HARNESS`: defaults to `pydantic_ai`. Set to `claude_agent_sdk` to start new ACP sessions on the Claude Agent SDK harness.
- `COMPILAGENT_CLAUDE_SDK_MAX_TURNS`: defaults to `24`.
- `COMPILAGENT_CLAUDE_SDK_MAX_BUDGET_USD`: optional Agent SDK cost cap.
- `COMPILAGENT_CLAUDE_SDK_PERMISSION_MODE`: defaults to `dontAsk`.

## ACP Server

After installing into `env`, launch the ACP server:

```bash
compilagent-triton
```

The server exposes optimizer agents through `pydantic-acp`. It creates per-session workspaces under `.compilagent-triton/`, records optimization episodes, captures Triton compile artifacts, and compares candidate configurations against baseline runs.

ACP clients receive a session-local `Harness` selector:

- `Current`: the existing `pydantic-ai` optimizer harness.
- `Claude Agent SDK`: routes prompts through the Claude Agent SDK while exposing the same Triton optimizer tools as an in-process MCP server.

The Claude Agent SDK harness uses the same `ANTHROPIC_API_KEY` and strips the `anthropic:` prefix from `COMPILAGENT_MODEL` before passing the model id to the SDK.

## Development Shape

Important modules:

- `settings.py`: typed environment and model settings.
- `agent.py`: `pydantic-ai` agent plus ACP adapter configuration.
- `workspace.py`: path-safe session workspace and artifact paths.
- `schemas.py`: Pydantic models for kernels, candidates, decisions, benchmarks, and episodes.
- `compiler.py`: controlled Triton compile harness.
- `triton_hooks/stages.py`: scoped `add_stages_inspection_hook` management.
- `decision_traces.py`: observe-only TTGIR decision extraction.
- `benchmarking.py`: correctness-checked benchmark helpers.
- `episodes.py`: file-backed episode state.

## Testing

Run top-level tests from the activated `env`:

```bash
python -m pytest
```

GPU-dependent Triton integration tests should be added separately and guarded so the core package remains testable on machines without CUDA.

## Benchmark Loop

Run the current vector-add candidate sweep on a selected GPU:

```bash
CUDA_VISIBLE_DEVICES=3 python -m compilagent_triton.gpu_benchmarks \
  --n-elements 8388608 \
  --block-sizes 256,512,1024,2048 \
  --num-warps 4,8,16 \
  --load-cache-modifiers none,.ca,.cg
```

Reports are written under `.compilagent-triton/reports/`, which is ignored by git.
