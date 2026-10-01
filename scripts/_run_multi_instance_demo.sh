#!/usr/bin/env bash
# Bring up the 15 endpoints and point the monitor at them over real HTTP.
# Explicit PID, never pkill -f: that pattern once matched its own command line.
set -uo pipefail
cd "$(dirname "$0")/.."

DIR="${1:-results/instances/intrusion_sharded}"
TAG="${2:-$(basename "$DIR")}"
SPEED="${3:-20}"

python3 scripts/serve_instances.py "$DIR" --paced "$SPEED" > "results/serve_${TAG}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

# wait for the last instance's port to answer rather than sleeping blind
for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:8736/health" >/dev/null 2>&1; then break; fi
  sleep 0.25
done
curl -fsS "http://127.0.0.1:8736/health" >/dev/null 2>&1 || { echo "endpoints não subiram"; cat "results/serve_${TAG}.log"; exit 1; }
echo "[demo] 15 endpoints up ($DIR, paced ${SPEED}x)"

python3 scripts/multi_instance_monitor.py --seconds 22 --every 1.0 --tag "$TAG"
echo "[demo] serve log: results/serve_${TAG}.log"
