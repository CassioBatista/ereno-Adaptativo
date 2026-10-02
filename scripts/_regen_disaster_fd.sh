#!/usr/bin/env bash
# Authority rule moved to Disaster-FD priority: regenerate the streams and check that the
# ONLY differences from the previous ones are the decider of the FL->GL and the detail
# texts -- i.e. that the cached model reproduces the old inline training exactly.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /tmp/prev_streams
cp results/events_availability.jsonl results/events_intrusion.jsonl /tmp/prev_streams/

python3 -u scripts/scenario_events.py conf/scenarios/availability.yaml \
    conf/scenarios/intrusion.yaml conf/scenarios/fd_watchdog.yaml

python3 - <<'EOF'
import json
IGN = {"decided_by", "reason", "detail"}
for name in ("availability", "intrusion"):
    old = [json.loads(l) for l in open(f"/tmp/prev_streams/events_{name}.jsonl")]
    new = [json.loads(l) for l in open(f"results/events_{name}.jsonl")]
    assert len(old) == len(new), (name, len(old), len(new))
    diffs = []
    for o, n in zip(old, new):
        ko = {k: v for k, v in o.items() if k not in IGN}
        kn = {k: v for k, v in n.items() if k not in IGN}
        if ko != kn:
            diffs.append(o["seq"])
    changed = [(o["seq"], o["type"], o.get("decided_by"), "->", n.get("decided_by"), n.get("reason"))
               for o, n in zip(old, new) if o.get("decided_by") != n.get("decided_by")]
    print(f"{name}: {len(new)} events; differences outside decided_by/reason/detail: "
          f"{diffs or 'NONE'}")
    print(f"  decider changes: {changed or 'none'}")
EOF

for s in availability intrusion fd_watchdog; do
  python3 scripts/validate_events.py "results/events_$s.jsonl" | tail -2
done
