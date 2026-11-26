"""No training: settle time S = 5 s and 10 s against noisy trust levels, over three draws of the noise
(seeds 1-3; scripts/build_noisy_trace.py). For T1 and T2 and each noise (probe loss 1, 5, 10 %;
misreport 1, 5 %), per S: mode switches per node over a 1-s grid (min-max over nodes and draws), and
the share of node-rounds (rounds at 60 r + 30 s) in which the mode of ReSIDS differs from its mode on
the recorded trace with S = 5 s (min-max over draws). Writes results/ev1_tl_noise_summary.txt.
Read-only on the traces."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.arch_manager import TrustLevelArchManager as M
from fd.trust_level import NodeModes

CLEAN = {"T1": ("~/datasets/disaster_fd_run1", 4980), "T2": ("~/datasets/disaster_fd_scenario2", 5880)}
NOISE = [("probe loss 1 %", "l0.01_m0"), ("probe loss 5 %", "l0.05_m0"), ("probe loss 10 %", "l0.1_m0"),
         ("misreport 1 %", "l0_m0.01"), ("misreport 5 %", "l0_m0.05")]
lines = ["EXPLORATORY (protocol change log 2026-11-26): settle time against noisy trust levels, "
         "constructed variants of T1 and T2, three draws of the noise", ""]
for tr, (run, tend) in CLEAN.items():
    kw = dict(t_end=tend, delta_s=60, links="scenario", round_offset_s=30, policy="switching")
    ref = M(run=os.path.expanduser(run), settle=5.0, **kw)
    nr = ref.n_rounds
    refm = [ref.node_modes(k) for k in range(1, nr + 1)]
    for label, tag in NOISE:
        parts = []
        for S in (5.0, 10.0):
            sw_all, diffs = [], []
            for s in (1, 2, 3):
                r = os.path.expanduser(f"~/datasets/disaster_fd_{tr}_noise_{tag}_s{s}")
                nm = NodeModes(r, tend, settle=S)
                seq = np.array([nm.modes(t) for t in np.arange(0, tend, 1.0)])
                sw_all += [(seq[1:, i] != seq[:-1, i]).sum() for i in range(seq.shape[1])]
                m = M(run=r, settle=S, **kw)
                diffs.append(100 * sum(a != b for k in range(1, nr + 1)
                                       for a, b in zip(m.node_modes(k), refm[k - 1])) / (nr * 14))
            parts.append(f"S={S:g} s: switches per node {min(sw_all)}-{max(sw_all)}, "
                         f"mode differs {min(diffs):.1f}-{max(diffs):.1f} %")
        lines.append(f"{tr} {label:<16} " + " | ".join(parts))
out = os.path.expanduser("~/ereno-Adaptativo/results/ev1_tl_noise_summary.txt")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
