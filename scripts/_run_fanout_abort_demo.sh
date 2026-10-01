#!/usr/bin/env bash
# Exercise the asymmetric unanimity rule with one instance genuinely absent:
# FL->GL must proceed (the straggler converges on GL, the safe resting state),
# GL->FL must ABORT (a partial return splits the federation).
set -uo pipefail
cd "$(dirname "$0")/.."

DIR="${1:-results/instances/intrusion_sharded}"

python3 scripts/serve_instances.py "$DIR" --skip c07 > results/serve_fanout_abort.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for _ in $(seq 1 40); do
  curl -fsS "http://127.0.0.1:8736/health" >/dev/null 2>&1 && break
  sleep 0.25
done
echo "[demo] endpoints up, c07 (port 8729) deliberately closed"
python3 scripts/multi_instance_monitor.py --seconds 4 --every 1.0 --tag fanout_abort 2>&1 \
  | sed -n '/command fan-out/,$p'
