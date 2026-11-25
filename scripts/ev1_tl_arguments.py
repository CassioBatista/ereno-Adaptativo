#!/usr/bin/env python3
"""Three analyses of the trust-level grid for the arguments of paper 2, 5.3 (no training;
logs of results/ev1_tl/grid_d60 and the Disaster-FD traces only). ReSIDS (switching), fixed GL
and fixed FL, all with retention; ERENO and CICIDS2017; T1 and T2.

(b) Avoidable exposure: the exposure of scripts/ev1_tl_grid_summary.py counted only over the
    node-rounds in which the node reaches the aggregator or a ring neighbour (the probes'
    link states, the same for every policy), so that nodes the trace cuts off do not count.
(c) Communication, as the pipeline exchanges (full state) and as an ideal delta exchange:
    - FL round with P participants and an aggregate of K boosters: P uploads of the node's own
      specialists and P downloads of the K boosters; 2P messages;
    - GL round (one head): the head receives the current pools of its up neighbours;
      one message per neighbour;
    - delta: every booster a node acquires from another node, counted once (with retention the
      holdings only grow, so it is the final holdings minus the node's own specialists).
    Booster size from the run's logged boosters (mean JSON size on disk).
(d) Stability and coherence of the decisions, from the traces (1-s grid over the run):
    - committed mode changes per node with the trust level and settle S = 5 s (registered),
      and with S = 0;
    - changes per node of the mode a node would take from its own probes (F if the aggregator
      answers, G if a neighbour answers, else L), the signal fixed policies use;
    - share of the time the trust-level mode and the probe mode differ;
    - in the switching runs, rounds in which the nodes' modes mix federated and gossip.

  python scripts/ev1_tl_arguments.py
Out: results/ev1_tl_arguments.txt
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
sys.path.insert(0, ROOT)
G = os.path.join(ROOT, "results/ev1_tl/grid_d60")
DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017")]
TRACES = [("T1", "~/datasets/disaster_fd_run1", 4980), ("T2", "~/datasets/disaster_fd_scenario2", 5880)]
POLS = [("switching", "ReSIDS"), ("fixed_gossip", "Fixed GL"), ("fixed_federated", "Fixed FL")]
BLOCKS = (1800, 3600)
REL = 0.95


def rdir(dom, tr, pol):
    return os.path.join(G, f"{dom}_{tr}_{pol}_retain_d60")


def node_eval(path):
    f = lambda v: float(v) if v not in ("", "nan") else np.nan
    return [{"t": float(r["t_s"]), "node": int(r["node"]), "reach": r["reachable"] == "1",
             **{a: f(r[f"recall_late_{a}"]) for a in BLOCKS}}
            for r in csv.DictReader(open(os.path.join(path, "node_eval.csv")))]


def reference(dom):
    ref = {a: 0.0 for a in BLOCKS}
    for p in glob.glob(os.path.join(G, f"{dom}_*_d60")):
        for r in node_eval(p):
            for a in BLOCKS:
                if not np.isnan(r[a]):
                    ref[a] = max(ref[a], r[a])
    return ref


def part_b(lines):
    lines.append("(b) exposure after each block's arrival: all node-rounds / only the node-rounds of "
                 "nodes that reach the aggregator or a neighbour")
    tot = {lab: [] for _, lab in POLS}
    for dom, dn in DOMS:
        ref = reference(dom)
        for tr, _, _ in TRACES:
            cells = []
            for pol, lab in POLS:
                rows = node_eval(rdir(dom, tr, pol))
                s = []
                for a in BLOCKS:
                    thr = REL * ref[a]
                    after = [r for r in rows if r["t"] >= a]
                    reach = [r for r in after if r["reach"]]
                    e_all = np.mean([not (r[a] >= thr) for r in after])
                    e_reach = np.mean([not (r[a] >= thr) for r in reach]) if reach else np.nan
                    tot[lab].append((e_all, e_reach))
                    s.append(f"{100 * e_all:5.1f}/{100 * e_reach:5.1f}")
                cells.append(f"{lab}: " + "  ".join(s))
            lines.append(f"  {dn:<11}{tr}  " + " | ".join(cells))
    lines.append("  mean over the 8 domain x trace x block cells (all / reachable): " + "; ".join(
        f"{lab} {100 * np.mean([x[0] for x in v]):.1f} / {100 * np.nanmean([x[1] for x in v]):.1f} %"
        for lab, v in tot.items()))


def own_counts(dom):
    spec = json.load(open(os.path.join(ROOT, f"results/ev1_step1_{dom}_s42.json")))["specialists"]
    c = {}
    for s in spec:
        c[s["node"] - 1] = c.get(s["node"] - 1, 0) + 1
    return c


def part_c(lines):
    lines.append("\n(c) communication over the trace: full-state boosters moved (MB) / messages; "
                 "delta boosters (MB); FL rounds / GL rounds")
    for dom, dn in DOMS:
        own = own_counts(dom)
        for tr, _, _ in TRACES:
            cells = []
            for pol, lab in POLS:
                p = rdir(dom, tr, pol)
                log = open(os.path.join(G, f"run_{dom}_{tr}_{pol}_retain_d60.log"), errors="replace").read()
                rounds = [json.loads(l) for l in open(os.path.join(p, "rounds.jsonl"))]
                hold = {r["round"]: r["holdings"] for r in rounds}
                sizes = [os.path.getsize(f) for f in glob.glob(os.path.join(p, "boosters", "*.json"))]
                kb = np.mean(sizes) / 1024
                full, msgs, nfl, ngl = 0, 0, 0, 0
                for m in re.finditer(r"\[Hybrid\] federated \(retain\): (\d+) participants \[([0-9, ]*)\] "
                                     r"-> aggregate of (\d+) boosters", log):
                    P, K = int(m.group(1)), int(m.group(3))
                    parts = [int(x) for x in m.group(2).split(",") if x.strip()]
                    full += P * K + sum(own.get(i, 0) for i in parts)
                    msgs += 2 * P
                    nfl += 1
                for m in re.finditer(r"\[GLow\] round=(\d+)\s+head=(\d+)\s+neighbours=\[([0-9, ]*)\]", log):
                    r = int(m.group(1))
                    nb = [int(x) for x in m.group(3).split(",") if x.strip()]
                    prev = hold.get(r - 1) or hold.get(r)
                    full += sum(len(prev[i]) for i in nb) if prev else 0
                    msgs += len(nb)
                    ngl += 1
                last = rounds[-1]["holdings"]
                delta = sum(max(len(h) - own.get(i, 0), 0) for i, h in enumerate(last))
                cells.append(f"{lab}: {full * kb / 1024:6.2f} MB / {msgs:4d} msg; delta {delta * kb / 1024:5.2f} MB; "
                             f"{nfl:2d} FL / {ngl:2d} GL")
            lines.append(f"  {dn:<11}{tr}  booster ~{kb:.1f} KB  " + " | ".join(cells))


def part_d(lines):
    from fd.trust_level import NodeModes, LinkStates
    lines.append("\n(d) decisions per node over the trace (1-s grid): mode changes with the trust level "
                 "(S = 5 s / S = 0), changes of the probe-driven mode, share of time the two differ; "
                 "mixed federated+gossip rounds in the switching runs")
    for tr, run, t_end in TRACES:
        run = os.path.expanduser(run)
        m5 = NodeModes(run, t_end, settle=5.0)
        m0 = NodeModes(run, t_end, settle=0.0)
        links = LinkStates(m5.trace)
        ts = np.arange(0, t_end, 1.0)
        ch5, ch0, chp, diff = [], [], [], []
        for n in range(m5.n_nodes):
            a5 = [m5.mode(n, t) for t in ts]
            a0 = [m0.mode(n, t) for t in ts]
            pr = ["federated" if links.aggregator_up(n, t) else "gossip" if links.peer_up(n, t) else "local"
                  for t in ts]
            c = lambda a: sum(1 for i in range(1, len(a)) if a[i] != a[i - 1])
            ch5.append(c(a5)); ch0.append(c(a0)); chp.append(c(pr))
            diff.append(np.mean([x != y for x, y in zip(a5, pr)]))
        mixed = {}
        for dom, dn in DOMS:
            rounds = [json.loads(l) for l in open(os.path.join(rdir(dom, tr, "switching"), "rounds.jsonl"))]
            mixed[dn] = sum(1 for r in rounds if {"federated", "gossip"} <= set(r["node_modes"] or []))
        lines.append(f"  {tr}: changes per node, trust level S=5 s: {np.min(ch5)}-{np.max(ch5)} (mean {np.mean(ch5):.1f}); "
                     f"S=0: {np.min(ch0)}-{np.max(ch0)} (mean {np.mean(ch0):.1f}); probe-driven: "
                     f"{np.min(chp)}-{np.max(chp)} (mean {np.mean(chp):.1f}); time trust-level mode != probe mode: "
                     f"{100 * np.mean(diff):.1f} % (per node {100 * np.min(diff):.1f}-{100 * np.max(diff):.1f} %); "
                     f"mixed F+G rounds: " + ", ".join(f"{k} {v}" for k, v in mixed.items()))


def main():
    lines = ["trust-level grid, delta 60 s, seed 42 -- arguments (b), (c), (d)", ""]
    part_b(lines)
    part_c(lines)
    part_d(lines)
    txt = "\n".join(lines)
    open(os.path.join(ROOT, "results/ev1_tl_arguments.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
