#!/bin/bash
# F1 launcher: env is exported INSIDE this script so every child inherits it
# (the earlier inline launch backgrounded the env setup with the first driver,
# starving the second of ANTHROPIC_API_KEY).
set -a; source /home/yigit/repos/research_stuff/.env; set +a
cd /home/yigit/repos/research_stuff/compilagent

WORKLOADS=softmax_4096,layernorm_2048x1024,matmul_relu_1024,gelu_8192x1024,l2norm_rows_4096x1024,fused_bias_swish_4096x1024
SETTINGS='{"anthropic_effort": "xhigh", "temperature": null}'

nohup env/bin/python -m scripts.run_pilot \
  --harnesses cascade,archetype_bon,archetype_band \
  --workloads "$WORKLOADS" --budgets 8 --seeds 13,42 \
  --model anthropic:claude-fable-5 --model-settings "$SETTINGS" \
  --price-in 15 --price-out 75 --gpu-pool 1,2,3 --episode-workers 8 \
  --llm-min-interval 0.6 --llm-max-concurrent 2 \
  --out scripts/results/fable5_grid.jsonl > scripts/results/fable5_f1a.log 2>&1 &

sleep 2

nohup env/bin/python -m scripts.run_pilot \
  --harnesses archetype_sr \
  --workloads "$WORKLOADS" --budgets 8 --seeds 13 \
  --model anthropic:claude-fable-5 --model-settings "$SETTINGS" \
  --price-in 15 --price-out 75 --gpu-pool 1,2,3 --episode-workers 3 \
  --llm-min-interval 0.6 --llm-max-concurrent 2 \
  --out scripts/results/fable5_grid.jsonl > scripts/results/fable5_f1b.log 2>&1 &

echo "F1 launched: $(date)"
