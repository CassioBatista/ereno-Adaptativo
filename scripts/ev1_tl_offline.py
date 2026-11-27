#!/usr/bin/env python3
"""Offline replay of the held specialists (no training, no Flower): propagates the boosters of a
pipeline run round by round under a schedule of modes and participants, with the rules of the
pipeline (retention; FL: every participant receives the deduplicated union of the participants'
sets; GL: the head active[(r-1) % len(active)] takes the union of its set with those of its ring
neighbours that are active; local: nothing; a node holds its own specialists from the first round
at or after its arrival time). The boosters and their hit vectors on the test split come from a
finished run (header.json, rounds.jsonl, scores.npz), so any schedule can be scored in seconds.

  python scripts/ev1_tl_offline.py --validate
     re-derives, for every run of results/ev1_tl/faithful, the holdings of every node in every
     round from the logged schedule, and compares them with the logged holdings.
Used by scripts/ev1_tl_sweep.py (constructed scenarios)."""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")


class Run:
    """Boosters of a finished run: own specialists per node, arrival times, hit vectors."""

    def __init__(self, d):
        self.d = d
        self.hdr = json.load(open(os.path.join(d, "header.json")))
        self.rounds = [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]
        self.n = int(self.hdr["num_nodes"])
        self.arrival = {int(k): float(v) for k, v in (self.hdr.get("arrival_s") or {}).items()}
        # own specialists: what a node holds at its first round at or after arrival that no node
        # held in the previous round
        self.own = [set() for _ in range(self.n)]
        prev = [set() for _ in range(self.n)]
        for r in self.rounds:
            seen = set().union(*prev)
            for i in range(self.n):
                new = set(r["holdings"][i]) - seen
                if new and not self.own[i]:
                    self.own[i] = new
            prev = [set(h) for h in r["holdings"]]
        z = os.path.join(d, "scores.npz")
        self.hits = dict(np.load(z)) if os.path.exists(z) else None

    def schedule(self):
        """The logged schedule: per round (t_s, mode, active)."""
        return [(r["t_s"], r["mode"], list(r["active"])) for r in self.rounds]


def propagate(run, schedule, arrival=None):
    """Holdings of every node after each round of `schedule` [(t_s, mode, active)]; `arrival`
    overrides the arrival time of the nodes (default: that of the run)."""
    n = run.n
    arrival = run.arrival if arrival is None else arrival
    hold = [set() for _ in range(n)]
    out = []
    for k, (t, mode, active) in enumerate(schedule, start=1):
        for i in range(n):
            if arrival.get(i, 0.0) <= t:
                hold[i] |= run.own[i]
        if mode == "federated" and active:
            u = set().union(*(hold[i] for i in active))
            for i in active:
                hold[i] = set(u)
        elif mode == "gossip" and active:
            h = active[(k - 1) % len(active)]
            act = set(active)
            for j in ((h - 1) % n, (h + 1) % n):
                if j in act and j != h:
                    hold[h] |= hold[j]
        out.append([set(x) for x in hold])
    return out


def validate():
    bad, total = 0, 0
    for d in sorted(glob.glob(os.path.join(ROOT, "results/ev1_tl/faithful/*_T[12]_*"))):
        if not os.path.exists(os.path.join(d, "rounds.jsonl")):
            continue
        run = Run(d)
        sim = propagate(run, run.schedule())
        diff = sum(sim[k][i] != set(r["holdings"][i]) for k, r in enumerate(run.rounds)
                   for i in range(run.n))
        total += 1
        bad += diff > 0
        print(f"{os.path.basename(d):<32} node-rounds differing: {diff}")
    print(f"{total} runs, {bad} with differences")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    if ap.parse_args().validate:
        validate()
