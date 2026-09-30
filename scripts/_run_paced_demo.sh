#!/usr/bin/env bash
# Paced replay + polling monitor, killed by PID (pkill -f would match this script).
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ereno314/bin/python

$PY scripts/replay_events_server.py --paced 150 --port 8722 > /tmp/replay.log 2>&1 &
SRV=$!
sleep 3
$PY scripts/monitor_poll_client.py --every 2 --seconds 40
echo "=== replay server: control-plane trace ==="
grep -E "node_failure|architecture_change|node_recovery|done" /tmp/replay.log | head -30
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit 0
