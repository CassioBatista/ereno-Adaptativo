#!/usr/bin/env bash
# Feature selection of docs/EVALUATION_PROTOCOL.md (evaluation-v1), section 2.2, in the four
# domains, under the best-subset retention rule (change log of 2026-11-12) and the per-type
# seeding of 2026-11-14 (scripts/grasp_por_ataque.py), on the TRAINING split only:
#   1. binary GRASP per attack type, every type from scratch (lambda 0.05, sample 60k, benign
#      cap 60k, attack rows capped at the benign count, 5 iterations without improvement, no
#      wall-clock cap; in the three new domains, types with fewer than 10 training rows get no
#      run), one evaluation checkpoint per type;
#   2. penalized global multiclass GRASP (lambda 0.05, sample 150k, 5 without improvement, no
#      wall-clock cap; class-balanced copy for CICIoT2023), resumed by deterministic replay from
#      the evaluations of the earlier run of the same configuration (its own checkpoint, or one
#      seeded from its log by scripts/grasp_ckpt_from_log.py); the evaluations do not depend on
#      the incumbent, so the result is that of a run from the start under the rule;
#   3. union -> features/all_in_one_<ds>_train_combined.json.
# Outputs carry the tag bsr005 (best-subset retention, lambda 0.05); the earlier outputs are not
# touched, except the combined files, which are replaced (the earlier ones are in git).
# One GRASP job at a time. Resumable: finished types are kept, each type and the global step
# resume from their checkpoints. The environment is recorded next to the log.
# Usage: DOMAINS="ereno morris4 ciciot cicids" scripts/run_ev1_grasp_bsr.sh  (log: results/grasp_bsr005.log)
#        TAG=bsr005q2d DOMAINS=morris4 scripts/run_ev1_grasp_bsr.sh  (log: results/grasp_bsr005q2d.log)
#        TAG=bsr005crc DOMAINS=morris4 scripts/run_ev1_grasp_bsr.sh  (log: results/grasp_bsr005crc.log)
set -euo pipefail
cd ~/ereno-Adaptativo
PY=~/venv-ereno314/bin/python
CK=features/checkpoints
# TAG: bsr005 by default. Another tag (bsr005q2d: Morris-4 rebuilt with Q2d before Q2a,
# amendment of 2026-11-16) writes its own outputs and runs the global step from the start,
# since the earlier evaluations were made on other data. bsr005crc: Morris-4 rebuilt with
# crc rate as content (16 features, 35 types), amendment of 2026-11-17; bsr005q2d was
# stopped in its global step and is not used.
TAG=${TAG:-bsr005}
if pgrep -f 'grasp_(por_ataque|global_penalizado)\.py' > /dev/null; then
  echo "another GRASP job is running; not starting"; exit 1
fi

for d in ${DOMAINS:-ereno morris4 ciciot cicids}; do
  ds=ev1_$d
  echo "$(date '+%F %T') start $ds"
  { date '+%F %T'; git rev-parse HEAD; git status --short python scripts/grasp_*.py scripts/run_ev1_grasp_bsr.sh
    git diff HEAD -- python scripts/grasp_*.py scripts/run_ev1_grasp_bsr.sh; $PY -V; $PY -m pip freeze; } \
      > results/grasp_env_${TAG}_${ds}_$(date +%F_%H%M).txt
  if [ "$d" = ereno ]; then src=all_in_one_ereno_train; gsrc=$src; attacks=""
  else
    src=all_in_one_${ds}_train; gsrc=$src
    [ -f all_in_one_${ds}_trainbal.csv ] && gsrc=all_in_one_${ds}_trainbal
    attacks=$($PY - "$src" <<'EOF'
import sys, pandas as pd
p = f"{sys.argv[1]}.csv"
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
  fi
  echo "[$ds] per-type GRASP for: ${attacks:-all types}"

  # 1. per type, from scratch
  [ -f features/por_ataque/${TAG}_${ds}_SUPERSET.json ] || $PY -u scripts/grasp_por_ataque.py \
      --dataset $src --global-file none --lambda 0.05 --sample 60000 \
      --normal-cap 60000 --attack-cap balanced --no-improvement 5 --max-time-h 0 \
      ${attacks:+--attacks "$attacks"} --tag ${TAG}_${ds} --checkpoint-dir $CK/por_ataque

  # 2. global, resumed from the earlier evaluations
  gck=$CK/global_${TAG}_${ds}.jsonl
  if [ ! -f features/global_${TAG}_${ds}.json ]; then
    if [ ! -f $gck ] && [ "$TAG" = bsr005 ]; then
      if [ -f $CK/global_penalizado_${ds}_L0.05.jsonl ]; then
        cp $CK/global_penalizado_${ds}_L0.05.jsonl $gck
      else
        $PY scripts/grasp_ckpt_from_log.py --log results/grasp_${ds}.log --dataset $gsrc \
            --no-improvement 5 --out $gck
      fi
    fi
    $PY -u scripts/grasp_global_penalizado.py --dataset $gsrc --lambda 0.05 --sample 150000 \
        --no-improvement 5 --out features/global_${TAG}_${ds}.json --checkpoint $gck --max-time-h 0
  fi

  # 3. union
  $PY - "$ds" "$src" "$TAG" <<'EOF'
import json, os, sys
ds, src, tag = sys.argv[1], sys.argv[2], sys.argv[3]
s = json.load(open(f"features/por_ataque/{tag}_{ds}_SUPERSET.json"))
g = json.load(open(f"features/global_{tag}_{ds}.json"))
comb = sorted(set(g["features"]) | set(s["superset_grasp"]))
out = {"dataset": src, "tag": tag,
       "method": "GRASP lambda=0.05: penalized global (multiclass) U binary per attack type "
                 "(evaluation-v1, best-subset retention rule, per-type seeding"
                 + ("; Morris-4 data rebuilt with Q2d before Q2a, amendment of 2026-11-16"
                    if tag == "bsr005q2d" else "")
                 + ("; Morris-4 data rebuilt with Q2d before Q2a and crc rate as content, "
                    "amendments of 2026-11-16 and 2026-11-17" if tag == "bsr005crc" else "") + ")",
       "global": g["features"], "per_type": s["per_attack"], "features": comb,
       "n_features": len(comb)}
nf = f"features/{ds}_feature_names.json"
if os.path.exists(nf):
    names = json.load(open(nf))["names"]
    out["names"] = [names[str(i)] for i in comb]
    out["per_type_names"] = {k: [names[str(i)] for i in v] for k, v in s["per_attack"].items()}
json.dump(out, open(f"features/all_in_one_{ds}_train_combined.json", "w"), indent=2)
print(ds, "COMBINED", len(comb), comb)
EOF
  echo "$(date '+%F %T') done $ds"
done
echo "fim"
