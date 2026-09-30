#!/usr/bin/env bash
# Regenerate every artifact DERIVED from the event streams, so the committed set is
# self-consistent (the streams are the source; tables/listings/census are derived).
# Does NOT regenerate the streams themselves (that retrains 14 specialists).
set -euo pipefail
cd "$(dirname "$0")/.."

for s in availability intrusion; do
  echo "=== $s ==="
  python3 scripts/validate_events.py "results/events_$s.jsonl"
  python3 scripts/validate_events.py "conf/scenarios/$s.yaml"
  python3 scripts/round_by_round.py "results/events_$s.jsonl" | tail -2
  python3 scripts/round_by_round_table.py "results/events_$s.jsonl"
done

echo "=== full-trace census ==="
python3 scripts/trace_census_table.py
