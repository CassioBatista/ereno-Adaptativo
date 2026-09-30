#!/usr/bin/env python3
"""Full-trace table: ONE ROW PER WINDOW of the whole trace, silence included.

The per-scenario tables only contain rounds that produced an event. This one covers the
entire timeline: every window of the test trace, whether it carried traffic, which
scenario stream it was routed to, and what (if anything) was emitted for it.

Rows are indexed by the window's position in the ORIGINAL trace, so the two scenario
streams — which are disjoint, interleaved views of this same trace and re-index their
rounds from 0 — can be read side by side on one clock.

Out: results/table_trace_full.csv   (LaTeX is not offered: ~5k rows is not a paper table)
"""
import csv
import datetime as dt
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util

TEST = "all_in_one_ereno_test"
DELTA = 1.0
SIM_EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
STREAMS = {"availability": "results/events_availability.jsonl",
           "intrusion": "results/events_intrusion.jsonl"}
OUT = "results/table_trace_full.csv"
COLS = ["window", "window_start", "window_end", "n_samples", "traffic", "stream",
        "scenario_round", "n_events", "event_types", "attack", "n_flags", "k_votes",
        "source_node", "decided_by"]


def iso(t):
    return (SIM_EPOCH + dt.timedelta(seconds=float(t))).isoformat(timespec="microseconds")


def main():
    print("[census] loading test labels + time column...")
    X, y, _ = util.load_arff(f"{TEST}.csv")
    nc = util.normal_class
    t = X[:, 0].copy()
    del X
    order = np.argsort(t, kind="stable")
    ts, y = t[order], y[order]
    t0 = float(ts[0])
    win = np.floor((ts - t0) / DELTA).astype(np.int64)
    n_win = int(win[-1]) + 1
    bnd = np.searchsorted(win, np.arange(n_win + 1))

    # events, mapped back to their ORIGINAL window via window_start
    ev_by_w, round_by_w, stream_by_w = {}, {}, {}
    for name, path in STREAMS.items():
        if not os.path.exists(path):
            print(f"[census] WARNING: missing {path}")
            continue
        for e in (json.loads(l) for l in open(path, encoding="utf-8")):
            secs = (dt.datetime.fromisoformat(e["window_start"]) - SIM_EPOCH).total_seconds()
            w = int(round((secs - t0) / DELTA))
            ev_by_w.setdefault(w, []).append(e)
            round_by_w[w] = e["round"]
            stream_by_w[w] = name

    rows, stats = [], Counter()
    for w in range(n_win):
        lo, hi = bnd[w], bnd[w + 1]
        n = int(hi - lo)
        if n == 0:
            traffic = "none"
        elif (y[lo:hi] == nc).all():
            traffic = "benign"
        else:
            traffic = "attack"
        stats[traffic] += 1
        evl = ev_by_w.get(w, [])
        al = [e for e in evl if e["type"] == "intrusion_detected"]
        one = al[0] if len(al) == 1 else None
        rows.append({
            "window": w, "window_start": iso(t0 + w * DELTA), "window_end": iso(t0 + (w + 1) * DELTA),
            "n_samples": n, "traffic": traffic,
            "stream": stream_by_w.get(w, ""), "scenario_round": round_by_w.get(w, ""),
            "n_events": len(evl),
            "event_types": "|".join(sorted({e["type"] for e in evl})),
            "attack": one["attack"] if one else "",
            "n_flags": one["n_flags"] if one else "",
            "k_votes": one["k_votes"] if one else "",
            "source_node": (one.get("source_node") if one and one.get("source_node") is not None else ""),
            "decided_by": "|".join(sorted({e["decided_by"] for e in evl if e.get("decided_by")})),
        })
        if evl:
            stats["windows_with_events"] += 1

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS)
        wr.writeheader()
        wr.writerows(rows)

    print(f"[census] {n_win} windows -> {OUT}")
    print(f"[census]   no traffic : {stats['none']:5d} ({100*stats['none']/n_win:.1f}%)")
    print(f"[census]   benign     : {stats['benign']:5d} ({100*stats['benign']/n_win:.1f}%)")
    print(f"[census]   with attack: {stats['attack']:5d} ({100*stats['attack']/n_win:.1f}%)")
    print(f"[census]   carried >=1 event: {stats['windows_with_events']}")
    mapped = len(ev_by_w)
    total_ev = sum(len(v) for v in ev_by_w.values())
    print(f"[census]   events mapped back: {total_ev} across {mapped} windows")


if __name__ == "__main__":
    main()
