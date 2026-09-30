#!/usr/bin/env python3
"""Round-by-round listing of a scenario stream — EVERY event explicit.

One round = one traffic window (window_seconds; ~1 s ~ 4.7k SV samples at 0.214 ms each).
Every event gets its own line, alarms included: nothing is summarised away. Rounds with
no event at all are simply absent (they carry no information), and their count is
reported in the header.

  python scripts/round_by_round.py results/events_availability.jsonl
  python scripts/round_by_round.py results/events_intrusion.jsonl --collapse   # compact
Out: prints, and writes results/round_by_round_<stem>.txt
"""
import argparse
import json
import os
from collections import Counter, defaultdict

SAMPLE_MS = 0.214


def describe(e):
    """One line of type-specific payload."""
    t = e["type"]
    if t == "intrusion_detected":
        src = "" if e.get("source_node") is None else f" source_node={e['source_node']}"
        return (f"attack={e['attack']:<22} n_flags={e['n_flags']:<5} k_votes={e['k_votes']:<3}"
                f" specialists={e.get('detector_specialists')}{src}")
    if t == "node_failure":
        return f"failed_nodes={e['failed_nodes']}"
    if t == "node_recovery":
        return f"recovered_nodes={e['recovered_nodes']}"
    if t == "node_isolated":
        return (f"source_node={e['source_node']} reason={e['reason']} "
                f"attack={e.get('attack')} k_votes={e.get('k_votes')}")
    if t == "architecture_change":
        return f"{e['from_mode']} -> {e['to_mode']}  reason={e['reason']}"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("--window-seconds", type=float, default=1.0)
    ap.add_argument("--collapse", action="store_true",
                    help="collapse rounds that carry only alarms (for slides)")
    a = ap.parse_args()

    evs = [json.loads(l) for l in open(a.stream, encoding="utf-8")]
    by_round = defaultdict(list)
    for e in evs:
        by_round[e["round"]].append(e)
    rounds = sorted(by_round)
    span = max(rounds) + 1
    per_round = a.window_seconds / (SAMPLE_MS / 1000)

    L = []
    L.append(f"# Round-by-round — {a.stream}")
    L.append(f"# 1 round = {a.window_seconds} s = ~{per_round:,.0f} samples at {SAMPLE_MS} ms each")
    L.append(f"# {len(evs)} events in {len(rounds)} rounds; the timeline spans {span} rounds, "
             f"so {span - len(rounds)} ({100*(span-len(rounds))/span:.1f}%) carry no event")
    L.append(f"# types: {dict(Counter(e['type'] for e in evs))}")
    L.append(f"# decided_by: {dict(Counter(e.get('decided_by') for e in evs))}")
    L.append("")
    L.append(f"{'seq':>5} {'round':>6} {'mode':<10} {'n_act':>5} {'decided_by':<11} "
             f"{'type':<20} payload")
    L.append("-" * 150)

    mode, n_act, shown = "federated", None, 0
    for r in rounds:
        evl = sorted(by_round[r], key=lambda e: e["seq"])
        ctrl = [e for e in evl if e["type"] != "intrusion_detected"]
        if a.collapse and not ctrl:
            L.append(f"{'':>5} {r:>6} {mode:<10} {str(n_act):>5} {'':<11} "
                     f"{'(alarms only)':<20} {len(evl)} alarm(s)")
        else:
            for e in evl:
                L.append(f"{e['seq']:>5} {e['round']:>6} {e.get('mode', ''):<10} "
                         f"{e.get('n_active', ''):>5} {e.get('decided_by') or '':<11} "
                         f"{e['type']:<20} {describe(e)}")
                shown += 1
        for e in evl:
            if e.get("mode"):
                mode = e["mode"]
            if e.get("n_active") is not None:
                n_act = e["n_active"]
    L.append("-" * 150)
    L.append(f"# {shown} event lines written" + ("  (collapsed mode)" if a.collapse else ""))
    txt = "\n".join(L)

    os.makedirs("results", exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.stream))[0]
    out = f"results/round_by_round_{stem}.txt"
    open(out, "w", encoding="utf-8").write(txt + "\n")
    print(txt)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
