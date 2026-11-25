#!/bin/bash
# evaluation-v1 section 9 grid (paper 2, 5.3): 6 configurations x traces T2, T1 x 4 domains,
# delta 60 s, ONE RUN AT A TIME. Each run: pipeline (run_v2.py) then per-node evaluation
# (scripts/ev1_tl_node_eval.py). Resumable: a run whose node_eval.txt exists is skipped.
#   bash run_tl_grid.sh                  # full grid, delta 60 s
#   SMOKE=1 bash run_tl_grid.sh          # delta 600 s, two configurations, T2 only
# Progress: grid_progress.log (start/end/minutes per run).
cd ~/ereno-Adaptativo/scripts/ev1_tl_runners || exit 1
PY=${PY:-$HOME/venv-ereno314/bin/python}
if [ -n "$SMOKE" ]; then
  DELTA=600; R=~/ereno-Adaptativo/results/ev1_tl/smoke_grid; TRACES="T2"
  CONFIGS="switching:retain fixed_federated:reformed"
else
  DELTA=60; R=~/ereno-Adaptativo/results/ev1_tl/grid_d60; TRACES="T2 T1"
  CONFIGS="switching:retain switching:reformed fixed_local:retain fixed_gossip:retain fixed_federated:retain fixed_federated:reformed"
fi
mkdir -p "$R"
LOG="$R/grid_progress.log"
nrounds() { echo $(( ( $1 + DELTA - 1 ) / DELTA - 1 )); }      # ceil(t_end/delta) - 1
for dspec in "ereno ereno all_in_one_ereno_train all_in_one_ereno_test 500000" \
             "cicids cicids all_in_one_ev1_cicids_train all_in_one_ev1_cicids_test -" \
             "morris4 morris4 all_in_one_ev1_morris4_train all_in_one_ev1_morris4_test -" \
             "ciciot ciciot all_in_one_ev1_ciciot_train all_in_one_ev1_ciciot_test -"; do
  set -- $dspec
  dom=$1; sel=$2; train=$3; test=$4
  cap="benign_cap: $5"; [ "$5" = "-" ] && cap=""
  for tr in $TRACES; do
    if [ "$tr" = "T2" ]; then trace=~/datasets/disaster_fd_scenario2; tend=5880
    else trace=~/datasets/disaster_fd_run1; tend=4980; fi
    nr=$(nrounds $tend)
    for c in $CONFIGS; do
      pol=${c%%:*}; hold=${c##*:}
      name="${dom}_${tr}_${pol}_${hold}_d${DELTA}"
      [ -s "$R/$name/node_eval.txt" ] && { echo "skip $name"; continue; }
      rm -rf "$R/$name"
      sed -e "s#RUNNAME#ev1-tl-$name#" -e "s/DOMSEL/$sel/" -e "s/TRAIN/$train/" -e "s/TEST/$test/" \
          -e "s/CAPLINE/$cap/" -e "s/NROUNDS/$nr/" -e "s#TRACE#$trace#" -e "s/TEND/$tend/" \
          -e "s/DELTA/$DELTA/" -e "s/POLICY/$pol/" -e "s/HOLDINGS/$hold/" -e "s#LOGDIR#$R/$name#" \
          tl_grid_template.yaml > "$R/cfg_$name.yaml"
      t0=$(date +%s); echo "$(date '+%F %T') start $name ($nr rounds)" | tee -a "$LOG"
      $PY run_v2.py "$R/cfg_$name.yaml" 14 > "$R/run_$name.log" 2>&1 \
        || { echo "$(date '+%F %T') FAIL $name" | tee -a "$LOG"; tail -30 "$R/run_$name.log"; exit 1; }
      $PY ~/ereno-Adaptativo/scripts/ev1_tl_node_eval.py "$R/$name" > "$R/eval_$name.log" 2>&1 \
        || { echo "$(date '+%F %T') FAIL eval $name" | tee -a "$LOG"; tail -20 "$R/eval_$name.log"; exit 1; }
      echo "$(date '+%F %T') done  $name in $(( ( $(date +%s) - t0 ) / 60 )) min" | tee -a "$LOG"
    done
  done
done
echo "$(date '+%F %T') grid complete" | tee -a "$LOG"
