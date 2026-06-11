#!/bin/bash
# Next-round discharge runs D1-D4 (all mistral via OpenRouter, pool mode).
set -a; source /home/yigit/repos/research_stuff/.env; set +a
cd /home/yigit/repos/research_stuff/compilagent
W=softmax_4096,layernorm_2048x1024,matmul_relu_1024,gelu_8192x1024,l2norm_rows_4096x1024,fused_bias_swish_4096x1024
M=openrouter:mistralai/mistral-large-2512
COMMON="--workloads $W --model $M --gpu-pool 1,2,3 --llm-min-interval 0.6 --llm-max-concurrent 2"

# D1: axis-bundle ablation, 5 bundles x 6 tasks x 3 seeds (cascade-disable per bundle)
for B in proposal feedback budget memory c9; do
  nohup env/bin/python -m scripts.run_pilot $COMMON \
    --harnesses cascade --budgets 8 --seeds 13,42,77 \
    --cascade-disable "$B" --episode-workers 4 \
    --out scripts/results/ablation_bundles.jsonl > "scripts/results/abl_$B.log" 2>&1 &
  sleep 1
done

# D2: +2 seeds for the headline grid (5 harnesses x 6 tasks x seeds {7,21})
nohup env/bin/python -m scripts.run_pilot $COMMON \
  --harnesses archetype_sr,archetype_bon,archetype_evo,archetype_band,cascade \
  --budgets 8 --seeds 7,21 --episode-workers 10 \
  --out scripts/results/t1lite_or.jsonl > scripts/results/seeds_7_21.log 2>&1 &

# D3: E=40 cells for budget-parity exactness (H-EVO, H-SR x 6 tasks, seed 13)
nohup env/bin/python -m scripts.run_pilot $COMMON \
  --harnesses archetype_evo,archetype_sr --budgets 40 --seeds 13 \
  --max-turns 50 --episode-workers 4 \
  --out scripts/results/depth_sweep.jsonl > scripts/results/e40.log 2>&1 &

# D4: open-model stability grid (qwen3-coder via OpenRouter)
nohup env/bin/python -m scripts.run_pilot $COMMON \
  --harnesses archetype_sr,archetype_bon,archetype_evo,archetype_band,cascade \
  --budgets 8 --seeds 13,42 --episode-workers 8 \
  --model openrouter:qwen/qwen3-coder \
  --out scripts/results/stability_qwen.jsonl > scripts/results/stability_qwen.log 2>&1 &

echo "discharge runs launched: $(date)"
