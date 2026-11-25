#!/bin/bash
# Faithful replay of the Disaster-FD traces (paper 2, 5.3): ReSIDS (switching FL/GL on the trust
# level), FL (fixed) and GL (fixed), all with retention, under the same network of the scenario
# (links: scenario) and rounds at 60 r + 30 s; four domains; T2 and T1. 24 runs, ONE AT A TIME,
# resumable (a run with node_eval.txt is skipped); then the summary.
#   bash run_tl_faithful.sh
#   DRY=1 bash run_tl_faithful.sh     # configs only
# Progress: results/ev1_tl/faithful/progress.log.
cd ~/ereno-Adaptativo/scripts/ev1_tl_runners || exit 1
PY=${PY:-$HOME/venv-ereno314/bin/python}
R=~/ereno-Adaptativo/results/ev1_tl/faithful
mkdir -p "$R"
LOG="$R/progress.log"
for dspec in "ereno ereno all_in_one_ereno_train all_in_one_ereno_test 500000" \
             "cicids cicids all_in_one_ev1_cicids_train all_in_one_ev1_cicids_test -" \
             "morris4 morris4 all_in_one_ev1_morris4_train all_in_one_ev1_morris4_test -" \
             "ciciot ciciot all_in_one_ev1_ciciot_train all_in_one_ev1_ciciot_test -"; do
  set -- $dspec
  dom=$1; sel=$2; train=$3; test=$4
  cap="benign_cap: $5"; [ "$5" = "-" ] && cap=""
  for tr in T2 T1; do
    if [ "$tr" = "T2" ]; then trace=~/datasets/disaster_fd_scenario2; tend=5880
    else trace=~/datasets/disaster_fd_run1; tend=4980; fi
    nr=$(( ( tend - 30 + 59 ) / 60 - 1 ))                  # rounds at 60 r + 30 < t_end
    for pol in switching fixed_federated fixed_gossip; do
      name="${dom}_${tr}_${pol}"
      [ -s "$R/$name/node_eval.txt" ] && { echo "skip $name"; continue; }
      rm -rf "$R/$name"
      sed -e "s#RUNNAME#ev1-tl-faithful-$name#" -e "s/DOMSEL/$sel/" -e "s/TRAIN/$train/" -e "s/TEST/$test/" \
          -e "s/CAPLINE/$cap/" -e "s/NROUNDS/$nr/" -e "s#TRACE#$trace#" -e "s/TEND/$tend/" \
          -e "s/POLICY/$pol/" -e "s#LOGDIR#$R/$name#" tl_faithful_template.yaml > "$R/cfg_$name.yaml"
      [ -n "$DRY" ] && { echo "config $name ($nr rounds)"; continue; }
      t0=$(date +%s); echo "$(date '+%F %T') start $name ($nr rounds)" | tee -a "$LOG"
      $PY run_v2.py "$R/cfg_$name.yaml" 14 > "$R/run_$name.log" 2>&1 \
        || { echo "$(date '+%F %T') FAIL $name" | tee -a "$LOG"; tail -30 "$R/run_$name.log"; exit 1; }
      $PY ~/ereno-Adaptativo/scripts/ev1_tl_node_eval.py "$R/$name" > "$R/eval_$name.log" 2>&1 \
        || { echo "$(date '+%F %T') FAIL eval $name" | tee -a "$LOG"; tail -20 "$R/eval_$name.log"; exit 1; }
      echo "$(date '+%F %T') done  $name in $(( ( $(date +%s) - t0 ) / 60 )) min" | tee -a "$LOG"
    done
  done
done
[ -n "$DRY" ] && exit 0
$PY ~/ereno-Adaptativo/scripts/ev1_tl_faithful_summary.py | tee -a "$LOG"
echo "$(date '+%F %T') faithful replay complete" | tee -a "$LOG"
