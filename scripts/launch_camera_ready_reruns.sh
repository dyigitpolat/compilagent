#!/bin/bash
# Camera-ready re-runs (2026-09-15).
#
# Re-runs the cells that scripts/quarantine_faulted.py moved out of the
# ledgers (the 2026-06-11 sandbox import fault and the skill-memory rule it
# distilled) and adds the qwen3-coder seeds 7/21/77. run_pilot skips every
# cell that still has a row, so each command names the ORIGINAL cross-product
# and only the quarantined or new cells actually run. Flags reproduce the
# June launches; --max-turns is not part of the cell key, so it must match
# (8 by default, 50 for the E=40 depth cells).
#
# GPU pool: device 0 is reserved on this machine (run_pilot refuses it) and
# devices 2-3 host a foreign vLLM engine that the lease guard refuses, so
# everything runs on GPU 1. Worker counts are sized for one GPU (about 20
# concurrent episodes; the sandbox lease serializes GPU use, and an episode
# is LLM-bound between evaluations).
set -a; source /home/yigit/repos/research_stuff/.env; set +a
cd /home/yigit/repos/research_stuff/compilagent
W=softmax_4096,layernorm_2048x1024,matmul_relu_1024,gelu_8192x1024,l2norm_rows_4096x1024,fused_bias_swish_4096x1024
M=openrouter:mistralai/mistral-large-2512
Q=openrouter:qwen/qwen3-coder
POOL="--gpu-pool 1 --llm-min-interval 0.6 --llm-max-concurrent 2"
LOG=scripts/results/rerun_20260915
mkdir -p "$LOG"

# 1. Primary grid: H-EVO / H-BAND seeds 7,21 (import fault) and CASCADE
#    seeds 7,21 (memory exposure) -- 34 cells.
nohup env/bin/python -m scripts.run_pilot --workloads $W $POOL --model $M \
  --harnesses archetype_evo,archetype_band,cascade --budgets 8 --seeds 7,21 \
  --episode-workers 3 \
  --out scripts/results/t1lite_or.jsonl > "$LOG/headline.log" 2>&1 &
echo "headline pid $!"

# 2. Ablation: 49 cells across the four mechanism groups.
for B in proposal feedback budget memory; do
  nohup env/bin/python -m scripts.run_pilot --workloads $W $POOL --model $M \
    --harnesses cascade --budgets 8 --seeds 13,42,77 --cascade-disable "$B" \
    --episode-workers 2 \
    --out scripts/results/ablation_bundles.jsonl > "$LOG/abl_$B.log" 2>&1 &
  echo "ablation $B pid $!"
  sleep 1
done

# 3. Depth sweep: H-EVO E=40 seed 13 (4 quarantined + 2 error rows) -- 6 cells.
nohup env/bin/python -m scripts.run_pilot --workloads $W $POOL --model $M \
  --harnesses archetype_evo --budgets 40 --seeds 13 --max-turns 50 \
  --episode-workers 2 \
  --out scripts/results/depth_sweep.jsonl > "$LOG/depth_e40.log" 2>&1 &
echo "depth pid $!"

# 4. qwen3-coder: 28 quarantined cells at seeds 13,42 plus 90 new cells at
#    seeds 7,21,77 -- 118 cells.
nohup env/bin/python -m scripts.run_pilot --workloads $W $POOL --model $Q \
  --harnesses archetype_sr,archetype_bon,archetype_evo,archetype_band,cascade \
  --budgets 8 --seeds 13,42,7,21,77 --episode-workers 4 \
  --out scripts/results/stability_qwen.jsonl > "$LOG/qwen.log" 2>&1 &
echo "qwen pid $!"

# 5. KernelBench-24 CASCADE arm (memory exposure) -- 48 cells.
KB=$(env/bin/python - <<'EOF'
import json
m = json.load(open("src/compilagent/integrations/triton_source/kernelbench_manifest.json"))
ids = [w["workload_id"] for w in m["selected"]]
assert len(ids) == 24, f"expected 24 ids, got {len(ids)}"
print(",".join(ids))
EOF
) || { echo "manifest parse failed"; exit 1; }
nohup env/bin/python -m scripts.run_pilot --workloads "$KB" $POOL --model $M \
  --harnesses cascade --budgets 8 --seeds 13,42 --episode-workers 3 \
  --out scripts/results/kb24_grid.jsonl > "$LOG/kb24_cascade.log" 2>&1 &
echo "kb24 cascade pid $!"

echo "camera-ready re-runs launched: $(date)"
