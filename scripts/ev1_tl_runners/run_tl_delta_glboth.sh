#!/bin/bash
# Delta sensitivity, fixed GL re-run with gl_peers: both (2026-11-24; protocol change log).
# The template (tl_faithful_template.yaml) has carried gl_peers: both since 2026-11-24 ~17:35,
# so the delta-5 runs of fixed GL already follow it. This script MOVES (does not delete) every
# fixed-GL delta run whose config lacks gl_peers: both to results/ev1_tl/delta_sens_gl_any/,
# then calls run_tl_delta.sh, which re-runs only what is missing (those runs) and updates the
# summary. ONE RUN AT A TIME; refuses to start while another pipeline run is active.
#   bash run_tl_delta_glboth.sh
#   DRY=1 bash run_tl_delta_glboth.sh     # list what would be moved
cd ~/ereno-Adaptativo/scripts/ev1_tl_runners || exit 1
if [ -z "$DRY" ] && pgrep -f 'run_tl_delta.sh|run_v2.py' > /dev/null; then
  echo "another pipeline run is active; wait for it to finish"; exit 1
fi
grep -q 'gl_peers: both' tl_faithful_template.yaml || { echo "template lacks gl_peers: both"; exit 1; }
R=~/ereno-Adaptativo/results/ev1_tl/delta_sens
OLD=~/ereno-Adaptativo/results/ev1_tl/delta_sens_gl_any
mkdir -p "$OLD"
n=0
for cfg in "$R"/cfg_*_fixed_gossip_d*.yaml; do
  [ -e "$cfg" ] || continue
  grep -q 'gl_peers: both' "$cfg" && continue
  name=$(basename "$cfg" .yaml); name=${name#cfg_}
  n=$((n + 1))
  if [ -n "$DRY" ]; then echo "would move $name"; continue; fi
  for f in "$name" "cfg_$name.yaml" "run_$name.log" "eval_$name.log"; do
    [ -e "$R/$f" ] && [ ! -e "$OLD/$f" ] && mv "$R/$f" "$OLD/$f"
  done
done
echo "$n fixed-GL runs with gl_peers any"
[ -n "$DRY" ] && exit 0
echo "$(date '+%F %T') fixed-GL re-run with gl_peers both: $n runs moved to $OLD" | tee -a "$R/progress.log"
DELTAS="600 300 180 15 5" bash run_tl_delta.sh
