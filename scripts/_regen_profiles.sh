#!/usr/bin/env bash
# Profiles are a domain binding: they enter the manifest and /status, never the events.
# Regenerate and prove the four streams are byte-identical to the committed ones.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/_add_profile.py
S="availability intrusion fd_watchdog two_cadences"
for s in $S; do python3 scripts/validate_events.py "conf/scenarios/$s.yaml" | tail -1; done
python3 -u scripts/scenario_events.py conf/scenarios/availability.yaml conf/scenarios/intrusion.yaml \
    conf/scenarios/fd_watchdog.yaml conf/scenarios/two_cadences.yaml | grep -E "events ->"
for s in $S; do
  if git diff --quiet -- "results/events_$s.jsonl"; then echo "$s: events byte-identical"; \
  else echo "$s: EVENTS CHANGED"; fi
  python3 -c "import json; print('  manifest profile:', json.load(open('results/manifest_$s.json'))['stream']['profile'])"
  python3 scripts/validate_events.py "results/events_$s.jsonl" | tail -1
  python3 scripts/commands_from_stream.py "results/events_$s.jsonl" > /dev/null
  python3 scripts/validate_commands.py "results/commands_$s.jsonl" | tail -1
done
