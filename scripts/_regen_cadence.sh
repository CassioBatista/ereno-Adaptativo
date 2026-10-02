#!/usr/bin/env bash
# Two-cadence generator, configured as the single cadence it replaces (1-s local tick,
# federated every round): every event must be identical to the committed streams except
# for the ADDED fields fed_round and traffic_time_start/end.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /tmp/prev_cad
for s in availability intrusion fd_watchdog; do git show "HEAD:results/events_$s.jsonl" > "/tmp/prev_cad/$s.jsonl"; done

python3 -u scripts/scenario_events.py conf/scenarios/availability.yaml \
    conf/scenarios/intrusion.yaml conf/scenarios/fd_watchdog.yaml | grep -E "windows|events ->"

python3 - <<'EOF'
import json
NEW = {"fed_round", "traffic_time_start", "traffic_time_end"}
for name in ("availability", "intrusion", "fd_watchdog"):
    old = [json.loads(l) for l in open(f"/tmp/prev_cad/{name}.jsonl")]
    new = [json.loads(l) for l in open(f"results/events_{name}.jsonl")]
    same_len = len(old) == len(new)
    diff = [o["seq"] for o, n in zip(old, new)
            if o != {k: v for k, v in n.items() if k not in NEW}]
    added = {k for n in new for k in n} - {k for o in old for k in o}
    print(f"{name:13s} events {len(new)} (same count: {same_len}); "
          f"differences outside the new fields: {diff or 'NONE'}; added: {sorted(added)}")
EOF

for s in availability intrusion fd_watchdog; do
  python3 scripts/validate_events.py "results/events_$s.jsonl" | tail -1
done
