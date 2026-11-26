#!/bin/bash
# Baseline of aggregator re-election (exploratory, protocol change log 2026-11-26): federated
# learning that elects a new aggregator among the connected nodes when the original one is lost
# (policy failover_federated of fd/arch_manager.py), on the faithful replay, seed 42, 4 domains x
# T2/T1, three variants: election of 1 round with a majority quorum (main), 3 rounds with a
# majority quorum, and 1 round without quorum. 24 runs, ONE AT A TIME, resumable; then the
# summary (scripts/ev1_tl_failover_summary.py).
#   bash run_tl_failover.sh
#   DRY=1 bash run_tl_failover.sh     # configs only
# Results: results/ev1_tl/faithful_failover/; about 2 h.
cd ~/ereno-Adaptativo/scripts/ev1_tl_runners || exit 1
if [ -z "$DRY" ] && pgrep -f 'run_tl_delta.sh|run_v2.py' > /dev/null; then
  echo "another pipeline run is active; wait for it to finish"; exit 1
fi
PY=${PY:-$HOME/venv-ereno314/bin/python}
R=~/ereno-Adaptativo/results/ev1_tl/faithful_failover
mkdir -p "$R"
LOG="$R/progress.log"
for var in "majority 1" "majority 3" "none 1"; do
  set -- $var; Q=$1; E=$2
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
      nr=$(( ( tend - 30 + 59 ) / 60 - 1 ))
      name="${dom}_${tr}_failover_${Q}${E}"
      [ -s "$R/$name/node_eval.txt" ] && { echo "skip $name"; continue; }
      rm -rf "$R/$name"
      sed -e "s#RUNNAME#ev1-tl-failover-$name#" -e "s/DOMSEL/$sel/" -e "s/TRAIN/$train/" -e "s/TEST/$test/" \
          -e "s/CAPLINE/$cap/" -e "s/NROUNDS/$nr/" -e "s#TRACE#$trace#" -e "s/TEND/$tend/" \
          -e "s/QUORUM/$Q/" -e "s/ELECTION/$E/" -e "s#LOGDIR#$R/$name#" tl_failover_template.yaml > "$R/cfg_$name.yaml"
      [ -n "$DRY" ] && { echo "config $name ($nr rounds)"; continue; }
      t0=$(date +%s); echo "$(date '+%F %T') start $name ($nr rounds)" | tee -a "$LOG"
      $PY run_v2.py "$R/cfg_$name.yaml" 14 > "$R/run_$name.log" 2>&1 \
        || { echo "$(date '+%F %T') FAIL $name" | tee -a "$LOG"; tail -30 "$R/run_$name.log"; exit 1; }
      grep -q "policy 'failover_federated'" "$R/run_$name.log" \
        || { echo "$(date '+%F %T') FAIL $name: not the failover policy" | tee -a "$LOG"; exit 1; }
      $PY ~/ereno-Adaptativo/scripts/ev1_tl_node_eval.py "$R/$name" > "$R/eval_$name.log" 2>&1 \
        || { echo "$(date '+%F %T') FAIL eval $name" | tee -a "$LOG"; tail -20 "$R/eval_$name.log"; exit 1; }
      echo "$(date '+%F %T') done  $name in $(( ( $(date +%s) - t0 ) / 60 )) min" | tee -a "$LOG"
    done
  done
done
[ -n "$DRY" ] && exit 0
$PY ~/ereno-Adaptativo/scripts/ev1_tl_failover_summary.py | tee -a "$LOG"
echo "$(date '+%F %T') failover baseline complete" | tee -a "$LOG"
