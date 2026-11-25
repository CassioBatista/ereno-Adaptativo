"""No training: ReSIDS with the gossip band at 15 (t_gl 15) vs fixed GL with gl_peers 'any'
(nodes with at least one ring link up). Per round: mode and nodes of ReSIDS, nodes of GL any,
and the rounds where ReSIDS changes against the band at 30. Read-only on the traces."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M

TR = {"T1": (os.path.expanduser("~/datasets/disaster_fd_run1"), 4980),
      "T2": (os.path.expanduser("~/datasets/disaster_fd_scenario2"), 5880)}
for tr, (run, tend) in TR.items():
    kw = dict(run=run, t_end=tend, delta_s=60, links="scenario", round_offset_s=30)
    s15 = M(policy="switching", gl=15.0, **kw)
    s30 = M(policy="switching", **kw)
    ga = M(policy="fixed_gossip", gl_peers="any", **kw)
    print(f"==== {tr}  ({s15.n_rounds} rounds)")
    diff_any = changed = 0
    for r in range(1, s15.n_rounds + 1):
        m15, n15, _ = s15._decide(r)
        m30, n30, _ = s30._decide(r)
        a = set(ga._decide(r)[1])
        t = s15.time_of(r) / 60
        if m15 == "gossip" and set(n15) != a:
            diff_any += 1
            print(f"  t={t:5.1f} ReSIDS15 gossip {sorted(x+1 for x in n15)} vs GL any {sorted(x+1 for x in a)}")
        if (m15, set(n15)) != (m30, set(n30)):
            changed += 1
            print(f"  t={t:5.1f} band 15: {m15:<9}{sorted(x+1 for x in n15)} | band 30: {m30:<9}{sorted(x+1 for x in n30)}")
    print(f"  gossip rounds of ReSIDS15 with other nodes than GL any: {diff_any}; "
          f"rounds changed against the band at 30: {changed}")
