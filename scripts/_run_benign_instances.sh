#!/usr/bin/env bash
# Waits for the benign-scenario fan-out to finish, then runs the monitor against those 15
# instances in BOTH views. The benign stream has no attack sample, so every alarm the
# monitor sees here is a false positive by construction: this is where fan-out either
# inflates the triage load or does not.
set -uo pipefail
cd "$(dirname "$0")/.."

until [ -f results/instance_corroboration_availability.csv ] \
      && [ -d results/instances/availability_sharded ]; do
  sleep 15
done
sleep 5   # let the last writes land

for view in replicated sharded; do
  echo "=============== availability_$view ==============="
  bash scripts/_run_multi_instance_demo.sh "results/instances/availability_$view" \
       "availability_$view" 60 2>&1 | sed -n '/per-instance reconciliation/,$p'
done
