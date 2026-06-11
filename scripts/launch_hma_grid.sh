#!/bin/bash
set -a; source /home/yigit/repos/research_stuff/.env; set +a
cd /home/yigit/repos/research_stuff/compilagent
nohup env/bin/python -m scripts.run_pilot \
  --harnesses archetype_ma \
  --workloads softmax_4096,layernorm_2048x1024,matmul_relu_1024,gelu_8192x1024,l2norm_rows_4096x1024,fused_bias_swish_4096x1024 \
  --budgets 8 --seeds 13,42,77,7,21 --max-turns 24 \
  --model openrouter:mistralai/mistral-large-2512 \
  --gpu-pool 1,2,3 --episode-workers 6 \
  --llm-min-interval 0.6 --llm-max-concurrent 2 \
  --out scripts/results/t1lite_or.jsonl > scripts/results/hma_grid.log 2>&1 &
echo "H-MA grid launched (30 episodes), pid $!"
