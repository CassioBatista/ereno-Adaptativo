#!/usr/bin/env python3
"""Serve the recorded event timeline to a real external monitor.

Replays results/adaptive_events_demo.jsonl over the REST surface that already exists in
fd/monitor.py (EventStore + MonitorServer, stdlib only) — so any monitor that speaks
HTTP/JSON can consume the 558 events without ReSIDS running.

Modes
  --all             preload every event, serve immediately (state = final state)
  --paced SPEED     release events on their own round schedule, compressed by SPEED
                    (e.g. --paced 100 -> one 1 s round every 10 ms; the 4984-round
                    timeline plays in ~50 s), with /status tracking mode and membership
                    as the timeline advances

Poll it the way the API says (docs/API.md §2): with the highest CONTIGUOUS seq.
  curl 'http://127.0.0.1:8722/status'
  curl 'http://127.0.0.1:8722/events?since=0&limit=5'
  curl 'http://127.0.0.1:8722/events?since=0&type=architecture_change'
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.monitor import EventStore, MonitorServer

IN = "results/adaptive_events_demo.jsonl"


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def push(store, ev):
    """Insert an event preserving its original seq/ts (EventStore.append would mint new
    ones — correct for live emission, wrong for a replay)."""
    with store._lock:                      # noqa: SLF001 - replay helper
        store._events.append(ev)
        store._seq = max(store._seq, int(ev["seq"]))


def state_from(ev, state):
    state["round"] = ev["round"]
    if ev.get("mode"):
        state["current_mode"] = ev["mode"]
    if ev.get("active_nodes") is not None:
        state["active_nodes"] = ev["active_nodes"]
        state["failed_nodes"] = sorted(set(range(14)) - set(ev["active_nodes"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=IN)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8722)
    ap.add_argument("--all", action="store_true", help="preload everything")
    ap.add_argument("--paced", type=float, metavar="SPEED",
                    help="replay on the round schedule, compressed by SPEED")
    a = ap.parse_args()

    events = load(a.file)
    print(f"[replay] {len(events)} events from {a.file}")
    store = EventStore(jsonl_path=None, keep_in_memory=len(events) + 10)
    state = {"current_mode": "federated", "round": 0,
             "active_nodes": sorted(range(14)), "failed_nodes": []}
    srv = MonitorServer(store, state, host=a.host, port=a.port).start()

    if a.paced:
        def run():
            t0 = time.time()
            last = 0
            for ev in events:
                target = t0 + (ev["round"] / a.paced)
                time.sleep(max(0.0, target - time.time()))
                state_from(ev, state)
                push(store, ev)
                if ev["type"] != "intrusion_detected":
                    print(f"[replay] round {ev['round']:5d}  seq {ev['seq']:4d}  "
                          f"{ev['type']:20s} n_active={ev.get('n_active')}")
                last = ev["round"]
            print(f"[replay] done: {len(events)} events, {last} rounds, "
                  f"{time.time() - t0:.1f}s wall")
        threading.Thread(target=run, name="replay", daemon=True).start()
    else:
        for ev in events:
            state_from(ev, state)
            push(store, ev)
        print(f"[replay] preloaded; last_seq={store.last_seq()}")

    print(f"[replay] try:  curl 'http://{a.host}:{a.port}/events?since=0&limit=3'")
    print("[replay] Ctrl-C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
        print("\n[replay] stopped")


if __name__ == "__main__":
    main()
