#!/usr/bin/env python3
"""EXPLORATORY (no training): when does switching matter? Constructed outage scenarios scored
with the offline replay (scripts/ev1_tl_offline.py, which reproduces the holdings of the 24
faithful runs exactly) and the boosters of the faithful T2 switching run of each domain (seed 42).

Scenario: ring of 14 nodes, every link up throughout (so C1 and C2 hold in every round). For
W = 15 rounds the aggregator answers and the initial specialists spread. In round W + 1, the late
block of nodes 3, 4, 11 and 12 arrives (their specialists; nodes 9 and 10 hold theirs from the
start) and the aggregator is lost for D rounds, D = 0 (never lost) ... 60; it then answers again.
Policies: ReSIDS (FL while the aggregator answers, GL otherwise), fixed FL, fixed GL.
Metric: share of node-rounds without protection against the block over the H = 60 rounds from
its arrival (protected: recall on the block's types at least 95 % of that of the detector holding
every specialist). With rounds of 60 s, D rounds is an outage of D minutes.

  python scripts/ev1_tl_sweep.py
Out: results/ev1_tl_sweep.txt
"""
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ev1_tl_offline import Run, propagate  # noqa: E402

DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017"), ("morris4", "Morris-4"), ("ciciot", "CICIoT2023")]
BLOCK = [2, 3, 10, 11]            # nodes 3, 4, 11 and 12 (0-based)
W, H = 15, 60
DS = [0, 1, 2, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 60]
REL = 0.95
SEED = 42                          # scripts/ev1_tl_sweep_seeds.py sets 43-46
_LOADED = {}


def run_dir(dom):
    base = "results/ev1_tl/faithful" if SEED == 42 else f"results/ev1_tl/faithful_seeds/s{SEED}"
    return os.path.join(ROOT, base, f"{dom}_T2_switching")


def load_test(hdr):
    import ev1_step1 as E
    k = (hdr["test_file"], tuple(hdr["features"]))
    if k not in _LOADED:
        _LOADED[k] = E.load(f"{hdr['test_file']}.csv", hdr["features"])
    return _LOADED[k]


def labels(run):
    hdr = run.hdr
    _, y, classes, nc = load_test(hdr)
    code = {c: i for i, c in enumerate(classes)}
    types = sorted({code[s] for i in BLOCK for s in str(hdr["node_specialist"][i]).split("+") if s in code})
    return np.isin(y, types)


def schedule(policy, D):
    n = 14
    allnodes = list(range(n))
    out = []
    for r in range(1, W + H + 1):
        up = not (W + 1 <= r < W + 1 + D)
        if policy == "FL":
            out.append((float(r), "federated" if up else "local", allnodes if up else []))
        elif policy == "GL":
            out.append((float(r), "gossip", allnodes))
        else:
            out.append((float(r), "federated" if up else "gossip", allnodes))
    return out


BLOCK_B = [8, 9]                  # nodes 9 and 10 (0-based), second family


def labels_of(run, nodes):
    hdr = run.hdr
    _, y, classes, nc = load_test(hdr)
    code = {c: i for i, c in enumerate(classes)}
    return [np.isin(y, sorted({code[s] for i in ns for s in str(hdr["node_specialist"][i]).split("+")
                                if s in code})) for ns in nodes]


def two_blocks(lines):
    """Second family: block A (nodes 3, 4, 11, 12) arrives as the aggregator is lost for D rounds,
    block B (nodes 9, 10) as it returns; a node is protected when it is protected against every
    block arrived so far. The better fixed mode differs between the two blocks."""
    lines += ["Second family: block A (nodes 3, 4, 11, 12) arrives in round "
              f"{W + 1} as the aggregator is lost for D rounds; block B (nodes 9, 10) arrives as it "
              f"returns (round {W + 1} + D); unprotected node-rounds (%) over the {H} rounds from "
              "the arrival of A, against every block arrived.", ""]
    for dom, name in DOMS:
        run = Run(run_dir(dom))
        mA, mB = labels_of(run, [BLOCK, BLOCK_B])
        allb = set().union(*run.own)
        pall = np.zeros(mA.size, bool)
        for h in allb:
            pall |= run.hits[h]
        refA, refB = pall[mA].mean(), pall[mB].mean()
        memo = {}

        def prot(hs, withB):
            k = (frozenset(hs), withB)
            if k not in memo:
                p = np.zeros(mA.size, bool)
                for h in hs:
                    p |= run.hits[h]
                ok = p[mA].mean() >= REL * refA
                if withB:
                    ok = ok and p[mB].mean() >= REL * refB
                memo[k] = ok
            return memo[k]

        lines.append(f"{name} (reference recall A {100 * refA:.1f} %, B {100 * refB:.1f} %)")
        lines.append(f"  {'D':>3}  {'ReSIDS':>7} {'FL':>7} {'GL':>7}   {'gain/FL':>8} {'gain/GL':>8} {'gain/best fixed':>16}")
        for D in DS:
            arrival = {i: (float(W + 1) if i in BLOCK else float(W + 1 + D) if i in BLOCK_B else 0.0)
                       for i in range(run.n)}
            un = {}
            for p in ("ReSIDS", "FL", "GL"):
                hold = propagate(run, schedule(p, D), arrival)
                vals = []
                for k in range(W, W + H):
                    withB = (k + 1) >= W + 1 + D
                    vals += [not prot(h, withB) for h in hold[k]]
                un[p] = 100 * np.mean(vals)
            gfl, ggl = un["FL"] - un["ReSIDS"], un["GL"] - un["ReSIDS"]
            lines.append(f"  {D:>3}  {un['ReSIDS']:>7.1f} {un['FL']:>7.1f} {un['GL']:>7.1f}   "
                         f"{gfl:>8.1f} {ggl:>8.1f} {min(gfl, ggl):>16.1f}")
        lines.append("")


def main():
    lines = ["EXPLORATORY (no training): constructed outage scenarios, offline replay of the held "
             "specialists (scripts/ev1_tl_offline.py), boosters of the faithful T2 runs, seed 42.",
             f"First family: late block of nodes 3, 4, 11, 12 arrives in round {W + 1} as the aggregator "
             f"is lost for D rounds; unprotected node-rounds (%) over the {H} rounds from the arrival.", ""]
    for dom, name in DOMS:
        run = Run(run_dir(dom))
        m = labels(run)
        allb = set().union(*run.own)
        pred_all = np.zeros(m.size, bool)
        for h in allb:
            pred_all |= run.hits[h]
        ref = pred_all[m].mean()
        arrival = {i: (float(W + 1) if i in BLOCK else 0.0) for i in range(run.n)}
        memo = {}

        def rec(hs):
            k = frozenset(hs)
            if k not in memo:
                p = np.zeros(m.size, bool)
                for h in hs:
                    p |= run.hits[h]
                memo[k] = p[m].mean()
            return memo[k]

        lines.append(f"{name} (reference recall on the block {100 * ref:.1f} %)")
        lines.append(f"  {'D':>3}  {'ReSIDS':>7} {'FL':>7} {'GL':>7}   {'gain/FL':>8} {'gain/GL':>8} {'gain/best fixed':>16}")
        for D in DS:
            un = {}
            for p in ("ReSIDS", "FL", "GL"):
                hold = propagate(run, schedule(p, D), arrival)
                rows = hold[W:W + H]
                un[p] = 100 * np.mean([rec(h) < REL * ref for rr in rows for h in rr])
            gfl, ggl = un["FL"] - un["ReSIDS"], un["GL"] - un["ReSIDS"]
            lines.append(f"  {D:>3}  {un['ReSIDS']:>7.1f} {un['FL']:>7.1f} {un['GL']:>7.1f}   "
                         f"{gfl:>8.1f} {ggl:>8.1f} {min(gfl, ggl):>16.1f}")
        lines.append("")
    two_blocks(lines)
    out = os.path.join(ROOT, "results/ev1_tl_sweep.txt")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
