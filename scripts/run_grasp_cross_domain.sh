#!/usr/bin/env bash
# GRASP for the cross-domain datasets with the ERENO recipe (lambda = 0.05):
#   1. build the ARFF files (GRASP sees TRAIN only);
#   2. binary GRASP per category (sample 60k, normal cap 60k, no-improvement 15, attack rows
#      capped at the normal count: when an attack outnumbers the benign sample, F1 rewards
#      the trivial "always attack" classifier);
#   3. penalized global multiclass GRASP (GR-G-VND, XGB clf 6, sample 150k, no-improvement 15);
#   4. union -> features/all_in_one_<ds>_train_combined.json.
# Datasets: $DATASETS (default cicids2017c; CICIoT2023 was dropped after the label audit,
# see docs/RESULTADOS_ciciot2023.md). Idempotent: each step is skipped when its output
# exists. Log: results/grasp_cross_domain.log
set -euo pipefail
cd ~/ereno-Adaptativo
PY=~/venv-ereno314/bin/python
DATASETS=${DATASETS:-cicids2017c}

for ds in $DATASETS; do
  [ -f all_in_one_${ds}_train.csv ] || $PY -u scripts/build_${ds}_dataset.py
  [ -f features/por_ataque/grasppen005_${ds}_SUPERSET.json ] || $PY -u scripts/grasp_por_ataque.py \
      --dataset all_in_one_${ds}_train --global-file none --lambda 0.05 --sample 60000 \
      --normal-cap 60000 --attack-cap balanced --no-improvement 15 --tag grasppen005_${ds}
done
for ds in $DATASETS; do
  # CICIoT2023 trains the global GRASP on a class-balanced copy (benign is 2.4 % of it)
  src=all_in_one_${ds}_train; [ -f all_in_one_${ds}_trainbal.csv ] && src=all_in_one_${ds}_trainbal
  [ -f features/global_penalizado_${ds}_L0.05.json ] || $PY -u scripts/grasp_global_penalizado.py \
      --dataset $src --lambda 0.05 --sample 150000 --no-improvement 15 \
      --out features/global_penalizado_${ds}_L0.05.json
done
for ds in $DATASETS; do
$PY - "$ds" <<'EOF'
import json, sys
ds = sys.argv[1]
s = json.load(open(f"features/por_ataque/grasppen005_{ds}_SUPERSET.json"))
g = json.load(open(f"features/global_penalizado_{ds}_L0.05.json"))
names = json.load(open(f"features/{ds}_feature_names.json"))["names"]
comb = sorted(set(g["features"]) | set(s["superset_grasp"]))
out = {"dataset": f"all_in_one_{ds}_train",
       "method": "GRASP lambda=0.05: penalized global (multiclass) U binary per category",
       "global": g["features"], "per_category": s["per_attack"], "features": comb,
       "n_features": len(comb), "names": [names[str(i)] for i in comb],
       "per_category_names": {k: [names[str(i)] for i in v] for k, v in s["per_attack"].items()}}
json.dump(out, open(f"features/all_in_one_{ds}_train_combined.json", "w"), indent=2)
print(ds, "COMBINED", len(comb), comb)
EOF
done
echo "fim"
