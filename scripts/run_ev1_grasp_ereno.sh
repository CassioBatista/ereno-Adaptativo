#!/usr/bin/env bash
# Feature selection of docs/EVALUATION_PROTOCOL.md (evaluation-v1), section 2.2, for ERENO,
# under the corrected implementation (change log of 2026-11-12 and 2026-11-13), with the same
# recipe as scripts/run_ev1_grasp.sh, on the TRAINING split only:
#   1. binary GRASP per attack type (lambda 0.05, sample 60k, benign cap 60k, attack rows
#      capped at the benign count, 5 iterations without improvement, no wall-clock cap);
#   2. penalized global multiclass GRASP (lambda 0.05, sample 150k, 5 without improvement,
#      no wall-clock cap; benign is 93 % of the ERENO training rows, so no balanced copy);
#   3. union -> features/all_in_one_ev1_ereno_train_combined.json.
# The selection published with the earlier work (features/all_in_one_ereno_train_combined.json,
# features/por_ataque/grasppen005_<type>.json) is not touched: every output here carries ev1_ereno.
# Resumable: each finished type is kept; the global step resumes from its checkpoint.
# The environment (Python and package versions) is recorded next to the log.
# Log: results/grasp_ev1_ereno.log
set -euo pipefail
cd ~/ereno-Adaptativo
PY=~/venv-ereno314/bin/python
ds=ev1_ereno
{ date '+%F %T'; git rev-parse HEAD; git status --short python scripts/grasp_*.py; $PY -V; $PY -m pip freeze; } \
    > results/grasp_env_${ds}_$(date +%F_%H%M).txt

[ -f features/por_ataque/grasppen005_${ds}_SUPERSET.json ] || $PY -u scripts/grasp_por_ataque.py \
    --dataset all_in_one_ereno_train --global-file none --lambda 0.05 --sample 60000 \
    --normal-cap 60000 --attack-cap balanced --no-improvement 5 --max-time-h 0 \
    --tag grasppen005_${ds}
[ -f features/global_penalizado_${ds}_L0.05.json ] || $PY -u scripts/grasp_global_penalizado.py \
    --dataset all_in_one_ereno_train --lambda 0.05 --sample 150000 --no-improvement 5 \
    --out features/global_penalizado_${ds}_L0.05.json \
    --checkpoint features/checkpoints/global_penalizado_${ds}_L0.05.jsonl --max-time-h 0
$PY - "$ds" <<'EOF'
import json, sys
ds = sys.argv[1]
s = json.load(open(f"features/por_ataque/grasppen005_{ds}_SUPERSET.json"))
g = json.load(open(f"features/global_penalizado_{ds}_L0.05.json"))
comb = sorted(set(g["features"]) | set(s["superset_grasp"]))
out = {"dataset": "all_in_one_ereno_train",
       "method": "GRASP lambda=0.05: penalized global (multiclass) U binary per attack type "
                 "(evaluation-v1, corrected implementation)",
       "global": g["features"], "per_type": s["per_attack"], "features": comb,
       "n_features": len(comb)}
json.dump(out, open(f"features/all_in_one_{ds}_train_combined.json", "w"), indent=2)
print(ds, "COMBINED", len(comb), comb)
EOF
echo "fim"
