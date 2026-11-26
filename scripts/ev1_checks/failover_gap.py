"""No training: what ReSIDS keeps doing where aggregator re-election cannot serve, on T1, T2 and the
regional-partition variant of T2 (constructed). From 15 min on (before, the trust levels are being
built and every node is isolated), for re-election (policy failover_federated, majority quorum,
election of 1 and 3 rounds), per round: the nodes in a federated group (the original aggregator, or
an elected one) are served; ReSIDS node-rounds in gossip outside any federated group are what it
adds, split by the cause: election pending (a majority is connected, no aggregator yet), no quorum
(no connected majority and no original aggregator), and outside the group (a federated group exists,
but the node is not in it, as a region cut off from it). Read-only on the traces."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M

TR = {"T1": (os.path.expanduser("~/datasets/disaster_fd_run1"), 4980),
      "T2": (os.path.expanduser("~/datasets/disaster_fd_scenario2"), 5880),
      "T2 regional (constructed)": (os.path.expanduser("~/datasets/disaster_fd_scenario2_regional"), 5880)}
for tr, (run, tend) in TR.items():
    kw = dict(run=run, t_end=tend, delta_s=60, links="scenario", round_offset_s=30, gl_peers="both")
    sw = M(policy="switching", **kw)
    for d in (1, 3):
        fo = M(policy="failover_federated", election_rounds=d, quorum="majority", **kw)
        n = fo.modes.n_nodes
        add = {"election pending": [0, 0], "no quorum": [0, 0], "outside the group": [0, 0]}  # rounds, node-rounds
        span = {}
        for r in range(1, fo.n_rounds + 1):
            t = fo.time_of(r)
            if t < 900:
                continue
            mode, group, _ = fo._decide(r)
            g = [i for i, x in enumerate(sw.node_modes(r)) if x == "gossip" and i not in group]
            if not g:
                continue
            if mode == "federated":
                k = "outside the group"
            elif len(fo._largest_component(t)) >= fo._quorum_size():
                k = "election pending"
            else:
                k = "no quorum"
            add[k][0] += 1
            add[k][1] += len(g)
            span.setdefault(k, [t / 60, t / 60])[1] = t / 60
        print(f"{tr}, election {d} round(s): " + "; ".join(
            f"{k} {v[0]} rounds, {v[1]} node-rounds" + (f" ({span[k][0]:.1f}-{span[k][1]:.1f} min)" if k in span else "")
            for k, v in add.items()))
