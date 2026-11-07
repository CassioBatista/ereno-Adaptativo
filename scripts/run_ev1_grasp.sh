#!/usr/bin/env bash
# Feature selection of docs/EVALUATION_PROTOCOL.md (evaluation-v1), section 2.2, for the three
# new domains, with the recipe of scripts/run_grasp_cross_domain.sh on the TRAINING split only:
#   1. binary GRASP per attack type (lambda 0.05, sample 60k, benign cap 60k, attack rows
#      capped at the benign count, 15 iterations without improvement); types with fewer than
#      10 training rows get no per-type run;
#   2. penalized global multiclass GRASP (lambda 0.05, sample 150k, 15 without improvement;
#      on the class-balanced copy for ev1_ciciot, whose benign is under 10 %);
#   3. union -> features/all_in_one_<ds>_train_combined.json.
# Idempotent and resumable (each finished type is kept). Log: results/grasp_ev1.log
set -euo pipefail
cd ~/ereno-Adaptativo
PY=~/venv-ereno314/bin/python
DATASETS=${DATASETS:-"ev1_morris4 ev1_ciciot ev1_cicids"}   # Morris-4 replaces Morris-3 (amendment)

for ds in $DATASETS; do
  attacks=$($PY - "$ds" <<'EOF'
import sys, pandas as pd
ds = sys.argv[1]
p = f"all_in_one_{ds}_train.csv"
nattr, skip = 0, 0
for skip, l in enumerate(open(p), 1):
    s = l.strip().lower()
    nattr += s.startswith("@attribute")
    if s.startswith("@data"):
        break
lab = pd.read_csv(p, skiprows=skip, header=None, usecols=[nattr - 1]).iloc[:, 0]
vc = lab.value_counts()
print(",".join(sorted(k for k, v in vc.items() if k != "Benign" and v >= 10)))
EOF
)
  echo "[$ds] per-type GRASP for: $attacks"
  [ -f features/por_ataque/grasppen005_${ds}_SUPERSET.json ] || $PY -u scripts/grasp_por_ataque.py \
      --dataset all_in_one_${ds}_train --global-file none --lambda 0.05 --sample 60000 \
      --normal-cap 60000 --attack-cap balanced --no-improvement 5 --attacks "$attacks" \
      --tag grasppen005_${ds}
  src=all_in_one_${ds}_train; [ -f all_in_one_${ds}_trainbal.csv ] && src=all_in_one_${ds}_trainbal
  [ -f features/global_penalizado_${ds}_L0.05.json ] || $PY -u scripts/grasp_global_penalizado.py \
      --dataset $src --lambda 0.05 --sample 150000 --no-improvement 5 \
      --out features/global_penalizado_${ds}_L0.05.json
  $PY - "$ds" <<'EOF'
import json, sys
ds = sys.argv[1]
s = json.load(open(f"features/por_ataque/grasppen005_{ds}_SUPERSET.json"))
g = json.load(open(f"features/global_penalizado_{ds}_L0.05.json"))
names = json.load(open(f"features/{ds}_feature_names.json"))["names"]
comb = sorted(set(g["features"]) | set(s["superset_grasp"]))
out = {"dataset": f"all_in_one_{ds}_train",
       "method": "GRASP lambda=0.05: penalized global (multiclass) U binary per attack type "
                 "(evaluation-v1)",
       "global": g["features"], "per_type": s["per_attack"], "features": comb,
       "n_features": len(comb), "names": [names[str(i)] for i in comb],
       "per_type_names": {k: [names[str(i)] for i in v] for k, v in s["per_attack"].items()}}
json.dump(out, open(f"features/all_in_one_{ds}_train_combined.json", "w"), indent=2)
print(ds, "COMBINED", len(comb), comb)
EOF
done
echo "fim"
