"""No training: the regional-partition variant of T2 (scripts/build_regional_trace.py). Per change
of state, the modes of the 14 nodes under ReSIDS (L/G/F per node), the nodes of fixed GL (both
neighbours) and of fixed FL, and the federated group of aggregator re-election (majority quorum,
1 round). Read-only on the traces."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M

run, tend = os.path.expanduser("~/datasets/disaster_fd_scenario2_regional"), 5880
kw = dict(run=run, t_end=tend, delta_s=60, links="scenario", round_offset_s=30, gl_peers="both")
sw, fl, gl = M(policy="switching", **kw), M(policy="fixed_federated", **kw), M(policy="fixed_gossip", **kw)
fo = M(policy="failover_federated", election_rounds=1, quorum="majority", **kw)
prev = None
for r in range(1, sw.n_rounds + 1):
    m = "".join(x[0].upper() for x in sw.node_modes(r))
    key = (m, tuple(gl._decide(r)[1]), tuple(fl._decide(r)[1]), tuple(fo._decide(r)[1]))
    if key != prev:
        one = lambda xs: [x + 1 for x in xs]
        print(f"{sw.time_of(r) / 60:5.1f} min  ReSIDS {m}  | GL {one(key[1])} | FL {one(key[2])} | failover {one(key[3])}")
        prev = key
print("conflicting rounds of ReSIDS (FL and GL at once):", len(sw.conflicts))
