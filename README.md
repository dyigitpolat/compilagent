# Compilagent

A lean, plug-and-play optimization core: any compiler, any agentic harness,
any workload. The core (`src/compilagent/`) defines seven plug-in slots —
`Backend`, `Workload`, `Harness`, `Toolset`, `ObservationSink`,
`ArtifactRenderer`, `CandidatePolicy` — and ships zero coupling to any
specific compiler, LLM runtime, or UI. Integrations live under
`src/compilagent/integrations/`.

> **Building your own integration?** A new compiler, agent runtime, or
> workload can plug into the core without forking this repo. See
> [docs/integration_guide.md](docs/integration_guide.md).

## Installation

Clean install from a fresh clone (Python ≥ 3.11):

```bash
git clone https://github.com/<your-fork>/compilagent.git
cd compilagent

# 1. Create and activate a venv
python3 -m venv env
source env/bin/activate

# 2. Upgrade pip (recommended)
python -m pip install --upgrade pip

# 3. Install the package + every integration
python -m pip install -e ".[all]"

# 4. Configure provider credentials
cp .env.example .env 2>/dev/null || touch .env
echo 'ANTHROPIC_API_KEY=sk-ant-...'                >> .env
echo 'COMPILAGENT_MODEL=anthropic:claude-opus-4-7' >> .env

# 5. Sanity check
python -c "import compilagent; print(compilagent.__name__, 'ok')"

# 6. Launch the observation UI (http://127.0.0.1:8765)
compilagent-observe
```

Pick narrower extras on machines that don't need everything:

```bash
python -m pip install -e ".[ui]"                              # UI only
python -m pip install -e ".[inductor,pydantic-ai]"            # CPU-only dev
python -m pip install -e ".[triton,inductor,pydantic-ai,ui]"  # GPU box
python -m pip install -e ".[dev,all]"                         # everything + dev tools
```

Available extras: `triton`, `inductor`, `pydantic-ai`, `claude-sdk`, `acp`,
`ui`, `all`, `dev`, `gpu`.

## Quickstart — replace `torch.compile` with agentic JIT

```python
import torch
import compilagent.integrations.python as cgp     # registers entry points
import compilagent.integrations.torch_inductor    # registers backend
import compilagent.integrations.pydantic_ai       # registers harness

model = MyTransformerBlock().cuda().eval()
x = torch.randn(8, 197, 768, device="cuda", dtype=torch.bfloat16)

result = cgp.optimize_module(model, example_inputs=(x,), max_candidates=8)
optimized = result.optimized_callable          # drop-in replacement
y_optimized = optimized(x)
print(f"{result.best_speedup:.3f}× speedup, correctness ok: {result.correctness_ok}")
```

(Or `pip install compilagent` and let the entry-point loader bring the
integrations in automatically when `OptimizationSession` constructs.)

The agent compiles a baseline through `torch.compile`, derives a search
space from the FX graph + Inductor knob catalog, proposes multi-knob
candidates, benchmarks each, validates correctness against the baseline
output, and hands back the **fastest validated callable**. If no candidate
beat baseline, `result.optimized_callable` is `None` and `result.improved`
is `False` — the caller falls back to their original code path.

## Quickstart — Triton kernel

```python
import torch, triton, triton.language as tl
import compilagent.integrations.python as cgp
import compilagent.integrations.triton                # registers backend
import compilagent.integrations.pydantic_ai           # registers harness

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

result = cgp.optimize_kernel(
    my_kernel,
    args=(x, y, out, n),
    grid=lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),),
    constexpr={"BLOCK_SIZE": 1024},
    max_candidates=8,
)
print(f"{result.best_speedup:.3f}× over the Triton-default pass pipeline")
```

The Triton backend operates on the MLIR pass pipeline
(`tritongpu-coalesce`, `-accelerate-matmul`, `-pipeline`,
`-optimize-thread-locality`, …) — skipping, reordering, parameterising
passes — never on `BLOCK_SIZE` or `num_warps` (those are user inputs, not
compiler decisions).

## Configuration

`compilagent` reads settings from `.env`. Minimum:

```bash
ANTHROPIC_API_KEY=...                # or MISTRAL_API_KEY / OPENAI_API_KEY
COMPILAGENT_MODEL=anthropic:claude-opus-4-7
```

Pass `model_id=` and `harness=` directly to override per-call:

```python
cgp.optimize_module(
    model, example_inputs,
    max_candidates=12,
    model_id="mistral:mistral-large-latest",
    harness="pydantic_ai",                  # or "claude_agent_sdk"
)
```

Harness × model compatibility:

- `harness="pydantic_ai"` works with any provider — `anthropic:`,
  `mistral:`, `openai:`.
- `harness="claude_agent_sdk"` is the `claude` CLI under the hood and only
  routes to Anthropic models (e.g. `anthropic:claude-opus-4-7`).

## Architecture

The core defines seven protocols and a session loop:

```
src/compilagent/
├── core/             # Backend, WorkloadSpec, Plan, Intervention, ToolDecl,
│                     # SearchSpace, CandidatePolicy, ValidationResult,
│                     # CompileResult, TimingResult, CorrectnessResult, …
├── session/          # OptimizationSession + the canonical 8-tool toolset
├── harness/          # Harness protocol + StreamEvent + HarnessRegistry
├── toolset/          # Toolset dataclass
├── observation/      # ObservationSink, EventKind, ArtifactRenderer
├── storage/          # OptimizationWorkspace, TraceStore, EpisodeStore,
│                     # ExperimentLog
└── integrations/
    ├── triton/                    # Triton compiler backend
    ├── torch_inductor/            # torch.compile / Inductor backend
    ├── python/                    # optimize_module / optimize_kernel
    ├── pydantic_ai/               # default LLM harness
    ├── claude_agent_sdk/          # Anthropic-only SDK harness
    ├── pydantic_acp/              # ACP server shim
    └── observation_ui/            # FastAPI app + SPA
```

The core never imports from `integrations/`. Each integration self-registers
into the appropriate registry (`backend_registry`, `harness_registry`,
`workload_registry`, `artifact_renderer_registry`) at import time. Out-of-tree
integrations advertise themselves through the
`compilagent.integrations` setuptools entry-point group; the core picks them
up automatically the first time `OptimizationSession` constructs.

## Console scripts

- `compilagent-observe` — FastAPI observation UI + SPA.
- `compilagent-acp` — pydantic-acp server (mounts an `OptimizationSession`
  per ACP session, with a runtime harness selector).

## Useful settings

- `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY` — provider creds.
- `COMPILAGENT_MODEL` (default `anthropic:claude-opus-4-7`).
- `COMPILAGENT_HARNESS` (default `pydantic_ai`).
- `COMPILAGENT_REASONING_EFFORT` (default `high`).
- `COMPILAGENT_MAX_TOKENS` (default `8192`).
- `COMPILAGENT_TEMPERATURE` (default `0.2`).
- `COMPILAGENT_MAX_CANDIDATES` (default `4`).
- `COMPILAGENT_MAX_BENCHMARK_SECONDS` (default `120`).
- `COMPILAGENT_HARNESS_EXTRA_JSON` — JSON dict of harness-specific knobs
  (e.g. `{"max_turns": 24, "permission_mode": "dontAsk"}`).
- `COMPILAGENT_INTEGRATIONS` — comma-separated list of additional dotted
  module paths to import at startup (alongside the entry-point loader).

## Testing

```bash
python -m pytest
```

Tests live under `tests/compilagent/`. Backend smoke tests guard themselves
with `pytest.importorskip("triton")` / `pytest.importorskip("torch")` and
skip cleanly on machines without CUDA / those libraries.
