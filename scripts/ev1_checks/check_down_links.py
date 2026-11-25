"""No training: in every gossip round, pairs of ring-adjacent nodes that are both in the active
set (so GLow lets them exchange: fd/topology.py up_neighbors uses node status) while the ring link
between them is down under links: scenario. Per policy and trace. Read-only on the traces."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M

TR = {"T1": (os.path.expanduser("~/datasets/disaster_fd_run1"), 4980),
      "T2": (os.path.expanduser("~/datasets/disaster_fd_scenario2"), 5880)}
POL = [("ReSIDS band 30", dict(policy="switching")), ("ReSIDS band 15", dict(policy="switching", gl=15.0)),
       ("GL both", dict(policy="fixed_gossip", gl_peers="both")), ("GL any", dict(policy="fixed_gossip", gl_peers="any"))]
for tr, (run, tend) in TR.items():
    for lab, extra in POL:
        m = M(run=run, t_end=tend, delta_s=60, links="scenario", round_offset_s=30, **extra)
        n, bad = 14, []
        for r in range(1, m.n_rounds + 1):
            mode, nodes, _ = m._decide(r)
            if mode != "gossip":
                continue
            act, t = set(nodes), m.time_of(r)
            for i in act:
                j = (i + 1) % n
                if j in act and not m.links.link_up(i, j + 1, t):
                    bad.append((round(t / 60, 1), i + 1, j + 1))
        mins = sorted({b[0] for b in bad})
        print(f"{tr} {lab:<15} exchanges over down links: {len(bad)} (node pairs x rounds)"
              + (f", {mins[0]}-{mins[-1]} min, pairs {sorted({b[1:] for b in bad})}" if bad else ""))
