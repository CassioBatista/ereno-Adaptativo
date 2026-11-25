#!/bin/bash
# Faithful replay, ReSIDS with the gossip band starting at 15 (t_gl: 15) instead of 30: a node
# that trusts a single neighbour gossips instead of learning locally (2026-11-25; protocol change
# log). Same settings as run_tl_faithful.sh otherwise. 8 runs (4 domains x T2/T1), ONE AT A TIME,
# resumable (a run with node_eval.txt is skipped). Results in results/ev1_tl/faithful_band15/;
# nothing of results/ev1_tl/faithful/ is touched. Comparison: fixed GL with gl_peers any (nodes
# with at least one ring link up), kept in results/ev1_tl/faithful_gl_any/ after
# run_tl_faithful_glboth.sh; in T2 its gossip nodes equal those of this ReSIDS in every round
# (scripts/ev1_checks/check_band15.py), in T1 they do not (see that check).
#   bash run_tl_faithful_band15.sh
#   DRY=1 bash run_tl_faithful_band15.sh     # configs only
# Run it only when no other pipeline run is active (WSL memory).
cd ~/ereno-Adaptativo/scripts/ev1_tl_runners || exit 1
if [ -z "$DRY" ] && pgrep -f 'run_tl_delta.sh|run_v2.py' > /dev/null; then
  echo "another pipeline run is active; wait for it to finish"; exit 1
fi
grep -q '^  t_gl: 15' tl_faithful_band15_template.yaml || { echo "template lacks t_gl: 15"; exit 1; }
PY=${PY:-$HOME/venv-ereno314/bin/python}
R=~/ereno-Adaptativo/results/ev1_tl/faithful_band15
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
    name="${dom}_${tr}_switching_gl15"
    [ -s "$R/$name/node_eval.txt" ] && { echo "skip $name"; continue; }
    rm -rf "$R/$name"          # only a partial run of this script can be here
    sed -e "s#RUNNAME#ev1-tl-faithful-band15-$name#" -e "s/DOMSEL/$sel/" -e "s/TRAIN/$train/" -e "s/TEST/$test/" \
        -e "s/CAPLINE/$cap/" -e "s/NROUNDS/$nr/" -e "s#TRACE#$trace#" -e "s/TEND/$tend/" \
        -e "s#LOGDIR#$R/$name#" tl_faithful_band15_template.yaml > "$R/cfg_$name.yaml"
    [ -n "$DRY" ] && { echo "config $name ($nr rounds)"; continue; }
    t0=$(date +%s); echo "$(date '+%F %T') start $name ($nr rounds)" | tee -a "$LOG"
    $PY run_v2.py "$R/cfg_$name.yaml" 14 > "$R/run_$name.log" 2>&1 \
      || { echo "$(date '+%F %T') FAIL $name" | tee -a "$LOG"; tail -30 "$R/run_$name.log"; exit 1; }
    grep -q "'gl': 15.0" "$R/run_$name.log" || { echo "$(date '+%F %T') FAIL $name: band not at 15" | tee -a "$LOG"; exit 1; }
    $PY ~/ereno-Adaptativo/scripts/ev1_tl_node_eval.py "$R/$name" > "$R/eval_$name.log" 2>&1 \
      || { echo "$(date '+%F %T') FAIL eval $name" | tee -a "$LOG"; tail -20 "$R/eval_$name.log"; exit 1; }
    echo "$(date '+%F %T') done  $name in $(( ( $(date +%s) - t0 ) / 60 )) min" | tee -a "$LOG"
  done
done
[ -n "$DRY" ] && exit 0
echo "$(date '+%F %T') band-15 replay complete (8 runs)" | tee -a "$LOG"
