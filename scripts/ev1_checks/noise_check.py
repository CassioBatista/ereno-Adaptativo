"""No training: the switching on noisy variants of T1 and T2 (scripts/build_noisy_trace.py), against
the clean traces. Per trace and noise: mode switches per node over a 1-s grid, with the settle time
of 5 s and without it (min-max over the nodes); share of node-rounds (rounds at 60 r + 30 s) in which
the mode of ReSIDS differs from its mode on the clean trace; rounds mixing federated and gossip
nodes; and, for fixed FL and fixed GL, the share of node-rounds whose participation differs from the
clean trace (the link states also carry the probe loss). Read-only on the traces."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M
from fd.trust_level import NodeModes

CLEAN = {"T1": ("~/datasets/disaster_fd_run1", 4980), "T2": ("~/datasets/disaster_fd_scenario2", 5880)}
NOISE = ["l0.01_m0", "l0.05_m0", "l0.1_m0", "l0_m0.01", "l0_m0.05"]


def switches(run, tend, settle):
    nm = NodeModes(os.path.expanduser(run), tend, settle=settle)
    grid = np.arange(0, tend, 1.0)
    seq = np.array([nm.modes(t) for t in grid])          # time x nodes
    return [(seq[1:, i] != seq[:-1, i]).sum() for i in range(seq.shape[1])]


def rounds(run, tend):
    kw = dict(run=os.path.expanduser(run), t_end=tend, delta_s=60, links="scenario", round_offset_s=30,
              gl_peers="both")
    return {p: M(policy=p, **kw) for p in ("switching", "fixed_federated", "fixed_gossip")}


for tr, (run, tend) in CLEAN.items():
    base = rounds(run, tend)
    nr = base["switching"].n_rounds
    s5, s0 = switches(run, tend, 5.0), switches(run, tend, 0.0)
    print(f"{tr} clean: switches per node S=5 {min(s5)}-{max(s5)}, S=0 {min(s0)}-{max(s0)}; "
          f"mixed rounds {sum(1 for r in range(1, nr + 1) if base['switching']._decide(r) and False)}"
          f"{len(base['switching'].conflicts)}")
    for nz in NOISE:
        nrun = f"~/datasets/disaster_fd_{tr}_noise_{nz}_s1"
        m = rounds(nrun, tend)
        s5, s0 = switches(nrun, tend, 5.0), switches(nrun, tend, 0.0)
        diff = sum(a != b for r in range(1, nr + 1)
                   for a, b in zip(m["switching"].node_modes(r), base["switching"].node_modes(r)))
        part = {}
        for p in ("fixed_federated", "fixed_gossip"):
            part[p] = sum(len(set(m[p]._decide(r)[1]) ^ set(base[p]._decide(r)[1])) for r in range(1, nr + 1))
        tot = nr * 14
        print(f"  {nz:<10} switches per node S=5 {min(s5)}-{max(s5)}, S=0 {min(s0)}-{max(s0)}; "
              f"ReSIDS mode differs in {100 * diff / tot:4.1f} % of node-rounds; mixed rounds "
              f"{len(m['switching'].conflicts)}; participation differs: fixed FL {100 * part['fixed_federated'] / tot:4.1f} %, "
              f"fixed GL {100 * part['fixed_gossip'] / tot:4.1f} %")
