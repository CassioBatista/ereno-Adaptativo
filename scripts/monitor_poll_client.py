#!/usr/bin/env python3
"""A minimal external monitor: polls the ReSIDS REST surface and reacts.

Implements exactly what docs/API.md §2 prescribes: poll GET /events?since=<highest
CONTIGUOUS seq>, and read /status only to show state. Prints control-plane events as
they arrive and escalates intrusion alarms by volume (n_flags).
"""
import argparse
import json
import time
import urllib.request

ESCALATE = 10          # n_flags threshold to call an alarm an incident


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=3) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8722")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=45.0)
    a = ap.parse_args()

    seen, contiguous, incidents, total = set(), 0, 0, 0
    t0 = time.time()
    print("  t     round  mode        n_active  last_seq  new  incidents")
    while time.time() - t0 < a.seconds:
        try:
            st = get(a.base, "/status")
            new = get(a.base, f"/events?since={contiguous}")
        except Exception as e:                      # server not up yet / stopped
            print("  poll failed:", e)
            time.sleep(a.every)
            continue
        for ev in new:
            seen.add(ev["seq"])
            if ev["type"] == "intrusion_detected":
                if ev.get("n_flags", 0) >= ESCALATE:
                    incidents += 1
            else:
                print(f"    >>> round {ev['round']:5d}  seq {ev['seq']:4d}  {ev['type']:20s}"
                      f" {ev.get('failed_nodes') or ev.get('recovered_nodes') or ''}"
                      f" {'' if ev['type'] != 'architecture_change' else ev['from_mode'] + '->' + ev['to_mode']}"
                      f"  n_active={ev.get('n_active', len(ev.get('active_nodes') or []))}")
        while contiguous + 1 in seen:
            contiguous += 1
        total += len(new)
        print(f"{time.time()-t0:5.1f}s  {st['round']:5d}  {st['current_mode']:10s}"
              f"  {len(st['active_nodes']):6d}  {st['last_seq']:8d}  {len(new):4d}  {incidents:7d}")
        time.sleep(a.every)
    print(f"\nseen={len(seen)}  contiguous={contiguous}  gaps={'none' if len(seen)==contiguous else 'YES'}"
          f"  incidents(n_flags>={ESCALATE})={incidents}")


if __name__ == "__main__":
    main()
