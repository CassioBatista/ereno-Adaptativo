#!/bin/bash
# Sensitivity to the round period (evaluation-v1 section 9: "sensitivity at 5, 15, 60, 180, 300
# and 600 s"), on the FAITHFUL replay (links: scenario; rounds offset by half a period, so that
# no round falls on a scenario event): 60 s = results/ev1_tl/faithful, not re-run here.
# ReSIDS, fixed FL and fixed GL (retention), registered arrivals (30/60 min), ERENO and
# CICIDS2017 (ALL_DOMAINS=1 adds Morris-4 and CICIoT2023), T2 and T1.
# ONE RUN AT A TIME, cheapest period first; resumable. For delta <= 15 s the per-round printed
# evaluation of the pipeline (not used by the analysis) uses 5,000 test rows.
#   bash run_tl_delta.sh                       # 600 300 180 15 5 s
#   DELTAS="600 300 180" bash run_tl_delta.sh  # a subset
#   DRY=1 bash run_tl_delta.sh                 # configs only
# Rough cost: 600-180 s about 1 h together; 15 s about 4 h; 5 s about 12 h.
# Progress: results/ev1_tl/delta_sens/progress.log.
cd ~/ereno-Adaptativo/scripts/ev1_tl_runners || exit 1
PY=${PY:-$HOME/venv-ereno314/bin/python}
R=~/ereno-Adaptativo/results/ev1_tl/delta_sens
mkdir -p "$R"
LOG="$R/progress.log"
DELTAS=${DELTAS:-"600 300 180 15 5"}
DOMS=("ereno ereno all_in_one_ereno_train all_in_one_ereno_test 500000"
      "cicids cicids all_in_one_ev1_cicids_train all_in_one_ev1_cicids_test -")
[ -n "$ALL_DOMAINS" ] && DOMS+=("morris4 morris4 all_in_one_ev1_morris4_train all_in_one_ev1_morris4_test -"
                                "ciciot ciciot all_in_one_ev1_ciciot_train all_in_one_ev1_ciciot_test -")
for D in $DELTAS; do
  [ "$D" = "60" ] && { echo "skip delta 60 (results/ev1_tl/faithful)"; continue; }
  OFF=$(awk "BEGIN{print $D/2}")                               # half a period
  ev=100000; [ "$D" -le 15 ] && ev=5000
  for dspec in "${DOMS[@]}"; do
    set -- $dspec
    dom=$1; sel=$2; train=$3; test=$4
    cap="benign_cap: $5"; [ "$5" = "-" ] && cap=""
    for tr in T2 T1; do
      if [ "$tr" = "T2" ]; then trace=~/datasets/disaster_fd_scenario2; tend=5880
      else trace=~/datasets/disaster_fd_run1; tend=4980; fi
      nr=$(awk "BEGIN{n=($tend-$OFF)/$D; c=(n==int(n))?n:int(n)+1; print c-1}")   # rounds at D r + OFF < t_end
      for pol in switching fixed_federated fixed_gossip; do
        name="${dom}_${tr}_${pol}_d${D}"
        [ -s "$R/$name/node_eval.txt" ] && { echo "skip $name"; continue; }
        rm -rf "$R/$name"
        sed -e "s#RUNNAME#ev1-tl-faithful-$name#" -e "s/DOMSEL/$sel/" -e "s/TRAIN/$train/" -e "s/TEST/$test/" \
            -e "s/CAPLINE/$cap/" -e "s/NROUNDS/$nr/" -e "s#TRACE#$trace#" -e "s/TEND/$tend/" \
            -e "s/POLICY/$pol/" -e "s#LOGDIR#$R/$name#" \
            -e "s/delta_s: 60/delta_s: $D/" -e "s/round_offset_s: 30/round_offset_s: $OFF/" \
            -e "s/eval_sample: 100000/eval_sample: $ev/" \
            tl_faithful_template.yaml > "$R/cfg_$name.yaml"
        [ -n "$DRY" ] && { echo "config $name ($nr rounds, offset $OFF s, eval_sample $ev)"; continue; }
        t0=$(date +%s); echo "$(date '+%F %T') start $name ($nr rounds)" | tee -a "$LOG"
        $PY run_v2.py "$R/cfg_$name.yaml" 14 > "$R/run_$name.log" 2>&1 \
          || { echo "$(date '+%F %T') FAIL $name" | tee -a "$LOG"; tail -30 "$R/run_$name.log"; exit 1; }
        $PY ~/ereno-Adaptativo/scripts/ev1_tl_node_eval.py "$R/$name" > "$R/eval_$name.log" 2>&1 \
          || { echo "$(date '+%F %T') FAIL eval $name" | tee -a "$LOG"; tail -20 "$R/eval_$name.log"; exit 1; }
        echo "$(date '+%F %T') done  $name in $(( ( $(date +%s) - t0 ) / 60 )) min" | tee -a "$LOG"
      done
    done
  done
  [ -z "$DRY" ] && $PY ~/ereno-Adaptativo/scripts/ev1_tl_delta_summary.py > /dev/null \
    && echo "$(date '+%F %T') summary updated after delta $D" | tee -a "$LOG"
done
[ -n "$DRY" ] && exit 0
$PY ~/ereno-Adaptativo/scripts/ev1_tl_delta_summary.py | tee -a "$LOG"
echo "$(date '+%F %T') delta sensitivity complete" | tee -a "$LOG"
