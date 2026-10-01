#!/usr/bin/env python3
"""Bring up 15 ReSIDS endpoints — the FL server and the 14 clients — one per instance.

Each instance gets its own EventStore and its own MonitorServer on its own port, so the
REST surface of docs/API.md is served fifteen times over, independently. This is what the
multi-instance monitor polls; nothing here is a mock of the API, it IS the API
(fd/monitor.py, stdlib only).

  srv  -> 127.0.0.1:8722      the FL aggregator: control plane only, silent in GL
  c00  -> 127.0.0.1:8723
  ...
  c13  -> 127.0.0.1:8736

  python scripts/serve_instances.py results/instances/intrusion_sharded
  python scripts/serve_instances.py results/instances/intrusion_sharded --paced 200

Poll one of them the way the API prescribes, with the highest CONTIGUOUS seq of THAT
instance — the seq spaces are independent, so a watermark from one is meaningless on
another:
  curl 'http://127.0.0.1:8723/events?since=0&limit=3'
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.monitor import EventStore, MonitorServer

N = 14


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def push(store, ev):
    with store._lock:                      # noqa: SLF001 - replay helper
        store._events.append(ev)
        store._seq = max(store._seq, int(ev["seq"]))


def state_from(ev, state):
    state["round"] = ev["round"]
    if ev.get("mode"):
        state["current_mode"] = ev["mode"]
    if ev.get("active_nodes") is not None:
        state["active_nodes"] = ev["active_nodes"]
        state["failed_nodes"] = sorted(set(range(N)) - set(ev["active_nodes"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="results/instances/<scenario>_<view>")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--base-port", type=int, default=8722)
    ap.add_argument("--paced", type=float, metavar="SPEED",
                    help="release each instance's events on the round schedule, compressed")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--skip", default="",
                    help="comma-separated instances NOT to start (e.g. c07), to exercise "
                         "the unreachable-instance path: a fan-out that needs unanimity "
                         "must abort, one that does not must proceed")
    a = ap.parse_args()

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    names = ["srv"] + [f"c{c:02d}" for c in range(N)]
    servers, counts = [], {}
    for k, nm in enumerate(names):
        if nm in skip:
            print(f"[serve] {nm} deliberately NOT started (port {a.base_port + k} closed)")
            continue
        evs = load(f"{a.dir}/{nm}.jsonl")
        counts[nm] = len(evs)
        store = EventStore(jsonl_path=None, keep_in_memory=len(evs) + 10)
        state = {"current_mode": "federated", "round": 0,
                 "active_nodes": sorted(range(N)), "failed_nodes": [],
                 "instance": nm, "role": "aggregator" if nm == "srv" else "agent"}
        srv = MonitorServer(store, state, host=a.host, port=a.base_port + k).start()
        servers.append((nm, srv, store, state, evs))

    print(f"[serve] {len(servers)}/{len(names)} instances up on "
          f"{a.host}:{a.base_port}-{a.base_port + 14}")
    print(f"[serve] events per instance: srv={counts.get('srv', 0)} "
          f"clients={[counts.get(f'c{c:02d}', 0) for c in range(N)]}")
    if counts["srv"] == 0:
        print("[serve] NOTE: the server instance has no events in this stream")

    if a.paced:
        def run(nm, store, state, evs):
            t0 = time.time()
            for ev in evs:
                time.sleep(max(0.0, t0 + ev["round"] / a.paced - time.time()))
                state_from(ev, state)
                push(store, ev)
        for nm, _, store, state, evs in servers:
            threading.Thread(target=run, args=(nm, store, state, evs),
                             name=f"replay-{nm}", daemon=True).start()
        print(f"[serve] paced at {a.paced}x; events are released as the rounds advance")
    else:
        for nm, _, store, state, evs in servers:
            for ev in evs:
                state_from(ev, state)
                push(store, ev)
        print("[serve] preloaded; last_seq per instance: "
              + " ".join(f"{nm}={st.last_seq()}" for nm, _, st, _, _ in servers))

    print("[serve] Ctrl-C to stop")
    try:
        t0 = time.time()
        while a.seconds <= 0 or time.time() - t0 < a.seconds:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    for _, s, _, _, _ in servers:
        s.stop()
    print("\n[serve] stopped")


if __name__ == "__main__":
    main()
