# Suite — 12 modern ML/NN primitives under pydantic-ai + Mistral Large

A benchmark suite that drives `compilagent` through 12 production-style
neural-net primitives — **6 PyTorch (Inductor) modules** and **6 Triton
kernels** — and reports the LLM-discovered speedup over each compiler's
default decisions. The agent runtime is the `pydantic_ai` harness backed
by `mistral:mistral-large-latest`.

## Candidates

| # | Name              | Backend          | Primitive                                                      |
|---|-------------------|------------------|----------------------------------------------------------------|
| 1 | `rmsnorm`         | torch_inductor   | Llama-style RMSNorm                                            |
| 2 | `swiglu_mlp`      | torch_inductor   | Llama FFN: down(silu(gate(x)) * up(x))                         |
| 3 | `mha`             | torch_inductor   | Multi-head self-attention (fused QKV + SDPA)                   |
| 4 | `gqa`             | torch_inductor   | Llama-3 grouped-query attention (Q-heads ≠ KV-heads)           |
| 5 | `rotary_attn`     | torch_inductor   | Multi-head attention with rotary position embeddings           |
| 6 | `moe_ffn`         | torch_inductor   | Mixtral-style top-k mixture of SwiGLU experts                  |
| 7 | `fused_softmax`   | triton           | Row-wise fused softmax (attention scores)                      |
| 8 | `rmsnorm_kernel`  | triton           | Forward RMSNorm (Llama-family normalization)                   |
| 9 | `layernorm_kernel`| triton           | Forward LayerNorm (classic transformer norm)                   |
|10 | `gelu_kernel`     | triton           | Forward GELU (tanh approximation)                              |
|11 | `dropout_kernel`  | triton           | Seeded Philox dropout                                          |
|12 | `matmul_kernel`   | triton           | Tiled GEMM (the canonical Triton primitive)                    |

PyTorch primitives live in `scripts/modules.py`, Triton kernels in
`scripts/kernels/*.py` (one kernel per file so the harness can import
each by file path).

## Run

Requires CUDA + the `triton`, `inductor`, and `pydantic-ai` extras
installed, plus `MISTRAL_API_KEY` in the environment / `.env`.

```bash
# all 12 candidates
python -m scripts.run_suite

# subset
python -m scripts.run_suite rmsnorm gelu_kernel matmul_kernel

# tune the per-workload LLM proposal budget (default 3)
SUITE_MAX_CANDIDATES=4 python -m scripts.run_suite
```

Per-workload results are written to
`scripts/results/suite_results.json` and the bar chart is rendered with:

```bash
python -m scripts.plot_results
# → scripts/results/suite_speedups.png
```

## What gets reported

For each candidate the runner records:

- `baseline_median_ms` — compiler default (Inductor or Triton MLIR pass
  pipeline)
- `best_median_ms` — fastest validated candidate the LLM-driven search
  produced
- `best_speedup` — `baseline / best`
- `correctness_ok` — output matches baseline within `ToleranceConfig`
- `improved` — true iff a validated candidate beat baseline

Failures (CUDA OOM, harness errors, etc.) are caught per-candidate and
land in the JSON as `{"error": "..."}` rows so one bad workload does not
sink the suite.
