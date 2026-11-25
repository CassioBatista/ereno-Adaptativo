"""No training: federated rounds of ReSIDS vs the nodes of fixed FL (aggregator link up)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M

TR = {"T1": (os.path.expanduser("~/datasets/disaster_fd_run1"), 4980),
      "T2": (os.path.expanduser("~/datasets/disaster_fd_scenario2"), 5880)}
for tr, (run, tend) in TR.items():
    kw = dict(run=run, t_end=tend, delta_s=60, links="scenario", round_offset_s=30)
    sw, fl = M(policy="switching", **kw), M(policy="fixed_federated", **kw)
    bad = [(r, sorted(sw._decide(r)[1]), sorted(fl._decide(r)[1])) for r in range(1, sw.n_rounds + 1)
           if sw._decide(r)[0] == "federated" and set(sw._decide(r)[1]) != set(fl._decide(r)[1])]
    print(tr, "federated rounds of ReSIDS with other nodes than fixed FL:", len(bad), bad[:5])
