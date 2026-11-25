#!/bin/bash
# Arrival-offset check of the trust-level grid (paper 2, 5.3; exploratory, not registered).
# The registered blocks join at 30 and 60 min, on phase boundaries of both traces; here both
# blocks are shifted by +5 min (35/65: T1 local phase / FL phase; T2 gossip phase / partition)
# and by +20 min (50/80: T1 gossip phase / FL; T2 partial partition / FL resumed).
# ReSIDS, fixed GL and fixed FL (all with retention); ERENO and CICIDS2017; T1 and T2;
# delta 60 s; ONE RUN AT A TIME; resumable (a run with node_eval.txt is skipped).
#   bash run_tl_arrival_offsets.sh            # 24 runs, then the summary
#   DRY=1 bash run_tl_arrival_offsets.sh      # write the configs only
# Progress: results/ev1_tl/arrival_offsets/progress.log.
cd ~/ereno-Adaptativo/scripts/ev1_tl_runners || exit 1
PY=${PY:-$HOME/venv-ereno314/bin/python}
R=~/ereno-Adaptativo/results/ev1_tl/arrival_offsets
mkdir -p "$R"
LOG="$R/progress.log"
OFFSETS="5 20"
CONFIGS="switching fixed_gossip fixed_federated"
for off in $OFFSETS; do
  a1=$(( 1800 + 60 * off )); a2=$(( 3600 + 60 * off ))
  for dspec in "ereno ereno all_in_one_ereno_train all_in_one_ereno_test 500000" \
               "cicids cicids all_in_one_ev1_cicids_train all_in_one_ev1_cicids_test -"; do
    set -- $dspec
    dom=$1; sel=$2; train=$3; test=$4
    cap="benign_cap: $5"; [ "$5" = "-" ] && cap=""
    for tr in T2 T1; do
      if [ "$tr" = "T2" ]; then trace=~/datasets/disaster_fd_scenario2; tend=5880
      else trace=~/datasets/disaster_fd_run1; tend=4980; fi
      nr=$(( ( tend + 59 ) / 60 - 1 ))
      for pol in $CONFIGS; do
        name="${dom}_${tr}_${pol}_retain_off${off}"
        [ -s "$R/$name/node_eval.txt" ] && { echo "skip $name"; continue; }
        rm -rf "$R/$name"
        sed -e "s#RUNNAME#ev1-tl-$name#" -e "s/DOMSEL/$sel/" -e "s/TRAIN/$train/" -e "s/TEST/$test/" \
            -e "s/CAPLINE/$cap/" -e "s/NROUNDS/$nr/" -e "s#TRACE#$trace#" -e "s/TEND/$tend/" \
            -e "s/POLICY/$pol/" -e "s/ARR1/$a1/g" -e "s/ARR2/$a2/g" -e "s#LOGDIR#$R/$name#" \
            tl_offset_template.yaml > "$R/cfg_$name.yaml"
        [ -n "$DRY" ] && { echo "config $name (arrivals $a1 / $a2 s, $nr rounds)"; continue; }
        t0=$(date +%s); echo "$(date '+%F %T') start $name ($nr rounds, arrivals $a1/$a2 s)" | tee -a "$LOG"
        $PY run_v2.py "$R/cfg_$name.yaml" 14 > "$R/run_$name.log" 2>&1 \
          || { echo "$(date '+%F %T') FAIL $name" | tee -a "$LOG"; tail -30 "$R/run_$name.log"; exit 1; }
        $PY ~/ereno-Adaptativo/scripts/ev1_tl_node_eval.py "$R/$name" > "$R/eval_$name.log" 2>&1 \
          || { echo "$(date '+%F %T') FAIL eval $name" | tee -a "$LOG"; tail -20 "$R/eval_$name.log"; exit 1; }
        echo "$(date '+%F %T') done  $name in $(( ( $(date +%s) - t0 ) / 60 )) min" | tee -a "$LOG"
      done
    done
  done
done
[ -n "$DRY" ] && exit 0
$PY ~/ereno-Adaptativo/scripts/ev1_tl_offset_summary.py | tee -a "$LOG"
echo "$(date '+%F %T') offsets complete" | tee -a "$LOG"
