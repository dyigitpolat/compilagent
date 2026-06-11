#!/bin/bash
set -a; source /home/yigit/repos/research_stuff/.env; set +a
cd /home/yigit/repos/research_stuff/compilagent
KB=$(env/bin/python - <<'EOF'
import json
m = json.load(open("src/compilagent/integrations/triton_source/kernelbench_manifest.json"))
sel = m["selected"] if isinstance(m, dict) and "selected" in m else m
ids = [w["workload_id"] if isinstance(w, dict) else w for w in sel]
assert len(ids) == 24, f"expected 24 ids, got {len(ids)}"
print(",".join(ids))
EOF
) || { echo "manifest parse failed"; exit 1; }
echo "tasks: $(echo "$KB" | tr ',' '\n' | wc -l)"
nohup env/bin/python -m scripts.run_pilot \
  --harnesses archetype_sr,archetype_bon,archetype_evo,archetype_band,cascade \
  --workloads "$KB" --budgets 8 --seeds 13,42 \
  --model openrouter:mistralai/mistral-large-2512 \
  --gpu-pool 1,2,3 --episode-workers 12 --llm-min-interval 0.6 --llm-max-concurrent 2 \
  --out scripts/results/kb24_grid.jsonl > scripts/results/kb24.log 2>&1 &
echo "kb24 driver pid $!"
