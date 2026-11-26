#!/bin/bash
# Faithful replay with seeds 43-46 (exploratory, protocol change log 2026-11-26): the same 24 runs
# as run_tl_faithful.sh (ReSIDS, fixed FL, fixed GL with gl_peers both; 4 domains x T2/T1), with
# the seed of the partition and of the training changed, so that seed 42 (results/ev1_tl/faithful)
# and seeds 43-46 give the dispersion of Section 5.3. 96 runs, ONE AT A TIME, resumable (a run
# with node_eval.txt is skipped); then scripts/ev1_tl_seeds_summary.py.
#   bash run_tl_faithful_seeds.sh
#   SEEDS="43" bash run_tl_faithful_seeds.sh      # one seed only
#   DRY=1 bash run_tl_faithful_seeds.sh           # configs only (new directory, nothing overwritten)
# Results: results/ev1_tl/faithful_seeds/s<seed>/; progress: results/ev1_tl/faithful_seeds/progress.log.
# About 2 h per seed on the workstation of the paper (the 24 runs of seed 42 took about 2 h).
cd ~/ereno-Adaptativo/scripts/ev1_tl_runners || exit 1
if [ -z "$DRY" ] && pgrep -f 'run_tl_delta.sh|run_v2.py' > /dev/null; then
  echo "another pipeline run is active; wait for it to finish"; exit 1
fi
grep -q '^seed: 42$' tl_faithful_template.yaml || { echo "template has no 'seed: 42' line"; exit 1; }
grep -q 'gl_peers: both' tl_faithful_template.yaml || { echo "template lacks gl_peers: both"; exit 1; }
PY=${PY:-$HOME/venv-ereno314/bin/python}
BASE=~/ereno-Adaptativo/results/ev1_tl/faithful_seeds
mkdir -p "$BASE"
LOG="$BASE/progress.log"
for SEED in ${SEEDS:-43 44 45 46}; do
  R=$BASE/s$SEED
  mkdir -p "$R"
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
      nr=$(( ( tend - 30 + 59 ) / 60 - 1 ))                # rounds at 60 r + 30 < t_end
      for pol in switching fixed_federated fixed_gossip; do
        name="${dom}_${tr}_${pol}"
        [ -s "$R/$name/node_eval.txt" ] && { echo "skip s$SEED $name"; continue; }
        rm -rf "$R/$name"                                  # only a partial run of this script can be here
        sed -e "s#RUNNAME#ev1-tl-faithful-s$SEED-$name#" -e "s/^seed: 42$/seed: $SEED/" \
            -e "s/DOMSEL/$sel/" -e "s/TRAIN/$train/" -e "s/TEST/$test/" \
            -e "s/CAPLINE/$cap/" -e "s/NROUNDS/$nr/" -e "s#TRACE#$trace#" -e "s/TEND/$tend/" \
            -e "s/POLICY/$pol/" -e "s#LOGDIR#$R/$name#" tl_faithful_template.yaml > "$R/cfg_$name.yaml"
        grep -q "^seed: $SEED$" "$R/cfg_$name.yaml" || { echo "seed not set in cfg_$name.yaml"; exit 1; }
        [ -n "$DRY" ] && { echo "config s$SEED $name ($nr rounds)"; continue; }
        t0=$(date +%s); echo "$(date '+%F %T') start s$SEED $name ($nr rounds)" | tee -a "$LOG"
        $PY run_v2.py "$R/cfg_$name.yaml" 14 > "$R/run_$name.log" 2>&1 \
          || { echo "$(date '+%F %T') FAIL s$SEED $name" | tee -a "$LOG"; tail -30 "$R/run_$name.log"; exit 1; }
        $PY ~/ereno-Adaptativo/scripts/ev1_tl_node_eval.py "$R/$name" > "$R/eval_$name.log" 2>&1 \
          || { echo "$(date '+%F %T') FAIL eval s$SEED $name" | tee -a "$LOG"; tail -20 "$R/eval_$name.log"; exit 1; }
        echo "$(date '+%F %T') done  s$SEED $name in $(( ( $(date +%s) - t0 ) / 60 )) min" | tee -a "$LOG"
      done
    done
  done
  [ -z "$DRY" ] && echo "$(date '+%F %T') seed $SEED complete" | tee -a "$LOG"
done
[ -n "$DRY" ] && exit 0
$PY ~/ereno-Adaptativo/scripts/ev1_tl_seeds_summary.py | tee -a "$LOG"
echo "$(date '+%F %T') faithful replay, seeds 43-46, complete" | tee -a "$LOG"
