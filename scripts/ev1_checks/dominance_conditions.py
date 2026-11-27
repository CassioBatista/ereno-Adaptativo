"""No training: checks, round by round on the recorded traces T1 and T2 (faithful replay settings:
links scenario, rounds at 60 r + 30 s, settle 5 s, gl_peers both), the two conditions of the
dominance proposition of Section 4.3:
  (C1) mode consistency: under ReSIDS a node federates iff it reaches the aggregator (the
       participants of fixed FL in that round), and gossips iff it is gossip-eligible (a
       participant of fixed GL) and does not reach the aggregator;
  (C2) uniform reach: if any node reaches the aggregator, every gossip-eligible node reaches it.
Under C1 and C2 every round of ReSIDS is the round of fixed FL (some node reaches the aggregator)
or of fixed GL (none does), never a mix; the simulation runs one mode per round, the majority one
if the modes of the nodes differ.
Also run with settle 10 s and on the constructed noisy / regional traces when present.
Read-only on the traces.
  python scripts/ev1_checks/dominance_conditions.py"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M  # noqa: E402

TRACES = [("T1", "~/datasets/disaster_fd_run1", 4980), ("T2", "~/datasets/disaster_fd_scenario2", 5880),
          ("T2 regional (constructed)", "~/datasets/disaster_fd_scenario2_regional", 5880)]


def check(run, tend, settle):
    kw = dict(run=os.path.expanduser(run), t_end=tend, delta_s=60, links="scenario",
              round_offset_s=30, settle=settle, gl_peers="both")
    s, fl, gl = (M(policy=p, **kw) for p in ("switching", "fixed_federated", "fixed_gossip"))
    n = s.n_rounds
    c1, c2, bad = 0, 0, []
    for k in range(1, n + 1):
        A, G = set(fl.get_active_clients(k)), set(gl.get_active_clients(k))
        m = s.node_modes(k)
        F_s = {i for i, x in enumerate(m) if x == "federated"}
        G_s = {i for i, x in enumerate(m) if x == "gossip"}
        ok1 = F_s == A and G_s == G - A
        ok2 = not A or G <= A
        c1 += ok1
        c2 += ok2
        if not (ok1 and ok2) and len(bad) < 5:
            bad.append((k, sorted(A), sorted(G), sorted(F_s), sorted(G_s)))
    return n, c1, c2, bad


for label, run, tend in TRACES:
    if not os.path.isdir(os.path.expanduser(run)):
        print(f"{label}: trace not found, skipped")
        continue
    for S in (5.0, 10.0):
        n, c1, c2, bad = check(run, tend, S)
        print(f"{label}, S={S:g} s: rounds {n}; C1 holds in {c1}, C2 holds in {c2}")
        for b in bad:
            print("   round %d: A=%s G=%s ReSIDS FL=%s GL=%s" % b)
