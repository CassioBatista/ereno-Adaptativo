"""No training: for T1 and T2 (faithful settings), compare the nodes of each gossip round under
ReSIDS (switching) with fixed GL under gl_peers 'any' and 'both'. Read-only on the traces."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M

TR = {"T1": (os.path.expanduser("~/datasets/disaster_fd_run1"), 4980),
      "T2": (os.path.expanduser("~/datasets/disaster_fd_scenario2"), 5880)}
for tr, (run, tend) in TR.items():
    kw = dict(run=run, t_end=tend, delta_s=60, links="scenario", round_offset_s=30)
    sw = M(policy="switching", **kw)
    ga = M(policy="fixed_gossip", gl_peers="any", **kw)
    gb = M(policy="fixed_gossip", gl_peers="both", **kw)
    print(f"==== {tr}")
    diff_any = diff_both = 0
    for r in range(1, sw.n_rounds + 1):
        ms, ns, _ = sw._decide(r)
        a = set(ga._decide(r)[1]); b = set(gb._decide(r)[1])
        g = set(ns) if ms == "gossip" else set()
        gl_res = {i for i, x in enumerate(sw.node_modes(r)) if x == "gossip"}
        if a != b:
            diff_any += 1
        if ms == "gossip" and g != b:
            diff_both += 1
            print(f"  round {r} t={sw.time_of(r)/60:.1f} min: ReSIDS gossip {sorted(x+1 for x in g)} "
                  f"vs GL both {sorted(x+1 for x in b)}")
        if r in (46, 47, 61, 62) or (a != b and r % 5 == 0):
            print(f"  t={sw.time_of(r)/60:5.1f} ReSIDS {ms:<9}{sorted(x+1 for x in ns)} | "
                  f"GL any {sorted(x+1 for x in a)} | GL both {sorted(x+1 for x in b)}")
    print(f"  rounds where GL any != GL both: {diff_any}; ReSIDS gossip rounds with other nodes "
          f"than GL both: {diff_both}")
