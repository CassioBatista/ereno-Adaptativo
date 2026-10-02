#!/usr/bin/env bash
# Derived artifacts after the two-cadence time reference: all from the cache, no retraining.
set -uo pipefail
cd "$(dirname "$0")/.."
S="availability intrusion fd_watchdog two_cadences"

echo "=== conformance, listings, tables, implied commands ==="
for s in $S; do
  printf "%-13s " "$s"
  python3 scripts/validate_events.py "results/events_$s.jsonl" | tail -1
  python3 scripts/round_by_round.py "results/events_$s.jsonl" > /dev/null
  python3 scripts/round_by_round_table.py "results/events_$s.jsonl" > /dev/null
  python3 scripts/commands_from_stream.py "results/events_$s.jsonl" | sed -n '1p'
  python3 scripts/validate_commands.py "results/commands_$s.jsonl" | tail -1
done

echo "=== 15 instances ==="
python3 scripts/instance_streams.py conf/scenarios/availability.yaml | grep -E "events across"
python3 scripts/instance_streams.py conf/scenarios/intrusion.yaml | grep -E "events across"

echo "=== monitor over HTTP ==="
bash scripts/_run_multi_instance_demo.sh results/instances/intrusion_sharded intrusion_sharded 20 > /dev/null 2>&1
bash scripts/_run_multi_instance_demo.sh results/instances/availability_replicated availability_replicated 60 > /dev/null 2>&1
bash scripts/_run_multi_instance_demo.sh results/instances/availability_sharded availability_sharded 60 > /dev/null 2>&1
bash scripts/_run_fanout_abort_demo.sh > /dev/null 2>&1
grep -h "gaps:" results/multi_instance_monitor_*.txt

echo "=== census ==="
python3 scripts/trace_census_table.py | tail -2
