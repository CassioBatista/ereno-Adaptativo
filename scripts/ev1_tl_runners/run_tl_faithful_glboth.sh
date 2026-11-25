#!/bin/bash
# Faithful replay, fixed GL re-run with gl_peers: both (2026-11-24): fixed GL gossips only with
# the nodes that have both ring links up, the condition of GL in the scenario documents and of
# the gossip band of ReSIDS, so that ReSIDS and fixed GL gossip with the same nodes (checked
# without training: scripts/ev1_checks/check_gl_peers.py). 8 runs, ONE AT A TIME, resumable.
# The previous fixed-GL runs (gl_peers: any) are MOVED, not deleted, to
# results/ev1_tl/faithful_gl_any/ and kept as record. Then the summary.
#   bash run_tl_faithful_glboth.sh
#   DRY=1 bash run_tl_faithful_glboth.sh     # configs only
# Run it only when no other pipeline run is active (WSL memory).
cd ~/ereno-Adaptativo/scripts/ev1_tl_runners || exit 1
if [ -z "$DRY" ] && pgrep -f 'run_tl_delta.sh|run_v2.py' > /dev/null; then
  echo "another pipeline run is active; wait for it to finish"; exit 1
fi
PY=${PY:-$HOME/venv-ereno314/bin/python}
R=~/ereno-Adaptativo/results/ev1_tl/faithful
OLD=~/ereno-Adaptativo/results/ev1_tl/faithful_gl_any
mkdir -p "$OLD"
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
    nr=$(( ( tend - 30 + 59 ) / 60 - 1 ))
    pol=fixed_gossip
    name="${dom}_${tr}_${pol}"
    if grep -q 'gl_peers: both' "$R/cfg_$name.yaml" 2>/dev/null && [ -s "$R/$name/node_eval.txt" ]; then
      echo "skip $name (already gl_peers: both)"; continue
    fi
    [ -n "$DRY" ] && { echo "would move $name to $OLD and re-run ($nr rounds)"; continue; }
    for f in "$name" "cfg_$name.yaml" "run_$name.log" "eval_$name.log"; do
      [ -e "$R/$f" ] && [ ! -e "$OLD/$f" ] && mv "$R/$f" "$OLD/$f"
    done
    rm -rf "$R/$name"          # only a partial re-run can be here; the old one is in $OLD
    sed -e "s#RUNNAME#ev1-tl-faithful-glboth-$name#" -e "s/DOMSEL/$sel/" -e "s/TRAIN/$train/" -e "s/TEST/$test/" \
        -e "s/CAPLINE/$cap/" -e "s/NROUNDS/$nr/" -e "s#TRACE#$trace#" -e "s/TEND/$tend/" \
        -e "s/POLICY/$pol/" -e "s#LOGDIR#$R/$name#" tl_faithful_glboth_template.yaml > "$R/cfg_$name.yaml"
    t0=$(date +%s); echo "$(date '+%F %T') start $name gl_peers both ($nr rounds)" | tee -a "$LOG"
    $PY run_v2.py "$R/cfg_$name.yaml" 14 > "$R/run_$name.log" 2>&1 \
      || { echo "$(date '+%F %T') FAIL $name" | tee -a "$LOG"; tail -30 "$R/run_$name.log"; exit 1; }
    $PY ~/ereno-Adaptativo/scripts/ev1_tl_node_eval.py "$R/$name" > "$R/eval_$name.log" 2>&1 \
      || { echo "$(date '+%F %T') FAIL eval $name" | tee -a "$LOG"; tail -20 "$R/eval_$name.log"; exit 1; }
    echo "$(date '+%F %T') done  $name gl_peers both in $(( ( $(date +%s) - t0 ) / 60 )) min" | tee -a "$LOG"
  done
done
[ -n "$DRY" ] && exit 0
$PY ~/ereno-Adaptativo/scripts/ev1_tl_faithful_summary.py | tee -a "$LOG"
echo "$(date '+%F %T') fixed-GL re-run (gl_peers both) complete" | tee -a "$LOG"
