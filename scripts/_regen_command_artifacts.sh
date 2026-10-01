#!/usr/bin/env bash
# Event schema changed (command_id, two reason values), so re-check the streams still
# conform, then derive and validate the command logs they imply.
set -euo pipefail
cd "$(dirname "$0")/.."

for s in availability intrusion; do
  echo "=== events: $s ==="
  python3 scripts/validate_events.py "results/events_$s.jsonl" | tail -3
done

for s in availability intrusion; do
  echo "=== commands: $s ==="
  python3 scripts/commands_from_stream.py "results/events_$s.jsonl"
  python3 scripts/validate_commands.py "results/commands_$s.jsonl" | tail -4
done
