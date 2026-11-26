"""No training: settle time S against probe loss. For T1 and T2, clean and with probe loss of 1, 5 and
10 % (scripts/build_noisy_trace.py, seed 1), per S in {5, 10, 15} s: mode switches per node over a
1-s grid (min-max over the nodes) and the share of node-rounds (rounds at 60 r + 30 s) in which the
mode of ReSIDS differs from its mode on the clean trace with S = 5 s (the reported setting).
Read-only on the traces."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M
from fd.trust_level import NodeModes

CLEAN = {"T1": ("~/datasets/disaster_fd_run1", 4980), "T2": ("~/datasets/disaster_fd_scenario2", 5880)}
for tr, (run, tend) in CLEAN.items():
    kw = dict(t_end=tend, delta_s=60, links="scenario", round_offset_s=30, policy="switching")
    ref = M(run=os.path.expanduser(run), settle=5.0, **kw)
    nr = ref.n_rounds
    for label, r in [("clean", run)] + [(f"loss {p}", f"~/datasets/disaster_fd_{tr}_noise_l{p}_m0_s1")
                                        for p in ("0.01", "0.05", "0.1")]:
        parts = []
        for S in (5.0, 10.0, 15.0):
            nm = NodeModes(os.path.expanduser(r), tend, settle=S)
            seq = np.array([nm.modes(t) for t in np.arange(0, tend, 1.0)])
            sw = [(seq[1:, i] != seq[:-1, i]).sum() for i in range(seq.shape[1])]
            m = M(run=os.path.expanduser(r), settle=S, **kw)
            diff = sum(a != b for k in range(1, nr + 1) for a, b in zip(m.node_modes(k), ref.node_modes(k)))
            parts.append(f"S={S:g}: switches {min(sw)}-{max(sw)}, differs {100 * diff / (nr * 14):4.1f} %")
        print(f"{tr} {label:<10} " + " | ".join(parts))
