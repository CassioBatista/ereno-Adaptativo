#!/usr/bin/env python3
"""Stability and coherence of the decisions on the faithful replay (paper 2, 5.3):
per node over a 1-s grid of each trace, the mode changes with the trust level (settle 5 s, the
registered value, and 0 s) and those of a node that took its mode from the network of the
scenario directly (F if it reaches the aggregator, G if a ring link is up in both directions,
else L); and, in the ReSIDS runs of results/ev1_tl/faithful, the rounds that mix federated and
gossip nodes.

  python scripts/ev1_tl_faithful_stability.py
Out: results/ev1_tl_faithful_stability.txt
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
sys.path.insert(0, ROOT)
from fd.trust_level import LinkStates, NodeModes  # noqa: E402

TRACES = [("T1", "~/datasets/disaster_fd_run1", 4980), ("T2", "~/datasets/disaster_fd_scenario2", 5880)]
DOMS = ["ereno", "cicids", "morris4", "ciciot"]
lines = ["decisions per node over the trace (1-s grid); network of the scenario (links both ways)", ""]
for tr, run, t_end in TRACES:
    run = os.path.expanduser(run)
    m5, m0 = NodeModes(run, t_end, settle=5.0), NodeModes(run, t_end, settle=0.0)
    links = LinkStates(m5.trace, symmetric=True)
    ts = np.arange(0, t_end, 1.0)
    c = lambda a: sum(1 for i in range(1, len(a)) if a[i] != a[i - 1])
    ch5, ch0, chn = [], [], []
    for n in range(m5.n_nodes):
        ch5.append(c([m5.mode(n, t) for t in ts]))
        ch0.append(c([m0.mode(n, t) for t in ts]))
        chn.append(c(["federated" if links.aggregator_up(n, t) else "gossip" if links.peer_up(n, t) else "local"
                      for t in ts]))
    mixed = {}
    for d in DOMS:
        p = os.path.join(ROOT, f"results/ev1_tl/faithful/{d}_{tr}_switching/rounds.jsonl")
        if os.path.exists(p):
            rounds = [json.loads(l) for l in open(p)]
            mixed[d] = sum(1 for r in rounds if {"federated", "gossip"} <= set(r["node_modes"] or []))
    lines.append(f"{tr}: mode changes per node -- trust level, settle 5 s: {min(ch5)}-{max(ch5)} (mean {np.mean(ch5):.1f}); "
                 f"settle 0 s: {min(ch0)}-{max(ch0)} (mean {np.mean(ch0):.1f}); from the network directly: "
                 f"{min(chn)}-{max(chn)} (mean {np.mean(chn):.1f}). ReSIDS rounds mixing F and G: "
                 + ", ".join(f"{k} {v}" for k, v in mixed.items()))
txt = "\n".join(lines)
open(os.path.join(ROOT, "results/ev1_tl_faithful_stability.txt"), "w").write(txt + "\n")
print(txt)
