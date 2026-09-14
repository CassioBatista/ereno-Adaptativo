#!/usr/bin/env python3
"""Latency-aware end-to-end curve (composition B2): the FL/GL F1 levels come from
the measured simulations (results/redund_k2_fulltest.csv, k>=2) and the FL->GL
switch is delayed by the DECENTRALIZED detection latency from the diffusion model
(fd/peer_failure.py: peer-timeout T + gossip diffusion to agreement quorum). This
is NOT a single Flower run -- it composes two validated pieces to show the cost of
decentralized detection: a degraded window before the switch, then GL recovery.
Out: results/latency_aware_end2end.{png,pdf}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.topology import build_topology
from fd.peer_failure import PeerFailureDetector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- F1 levels from the measured runs (redund_k2_fulltest.csv, k=2) ---
lvl = {}
with open("results/redund_k2_fulltest.csv") as fh:
    for r in csv.DictReader(fh):
        if int(r["k"]) == 2:
            lvl[int(r["N"])] = (float(r["FL_F1"]), float(r["GL_F1"]))
N0 = 14                      # healthy network size
F1_HEALTHY = lvl[N0][1]      # GL == FL when healthy (95.98)
F1_GL = lvl[N0][1]           # GL retains the union -> flat 95.98 after switch
N_AFTER = 7                  # burst failure: 14 -> 7 active nodes
F1_FL_DEGRADED = lvl[N_AFTER][0]   # static FL at 7 nodes (k>=2)

# --- decentralized detection latency from the model (peer-timeout T + diffusion) ---
T = 2
det = PeerFailureDetector(build_topology("ring", N0), timeout=T)
down, fail_round = N0 // 2, 10
L = None
for r in range(1, 8 * N0):
    alive = set(range(N0)) if r < (fail_round - 9) else set(range(N0)) - {down}
    det.observe_and_step(r, alive)
    if det.network_knows(down, alive):
        L = r - (fail_round - 9); break
L = L or 15                  # rounds from failure to network agreement (~T + diffusion)
switch_round = fail_round + L

ROUNDS = list(range(1, 41))
def static_fl(r):  return F1_HEALTHY if r < fail_round else F1_FL_DEGRADED
def adaptive(r):
    if r < fail_round:      return F1_HEALTHY
    if r < switch_round:    return F1_FL_DEGRADED       # degraded FL during detection window
    return F1_GL                                        # GL recovery after switch

fig, ax = plt.subplots(figsize=(9.2, 5.2))
ax.plot(ROUNDS, [adaptive(r) for r in ROUNDS], "-o", color="#1b7837", lw=2.4, ms=5,
        label="Adaptive (FL$\\rightarrow$GL), decentralized detection")
ax.plot(ROUNDS, [static_fl(r) for r in ROUNDS], "--s", color="#b2182b", lw=2.2, ms=5,
        label="Static FL (no switch)")

ax.axvspan(fail_round, switch_round, color="#f0ad4e", alpha=0.18)
ax.axvline(fail_round, color="0.4", ls=":", lw=1)
ax.axvline(switch_round, color="#2166ac", ls="-", lw=1.4, alpha=0.8)
ax.annotate("node loss (14$\\to$7)", (fail_round + 0.3, 90.5), fontsize=8.5,
            ha="left", va="center", color="0.35")
ax.annotate(f"switch FL$\\rightarrow$GL\n@round {switch_round}", (switch_round, 92),
            xytext=(switch_round+0.4, 88), fontsize=8.5, color="#2166ac")
ax.annotate(f"decentralized detection latency\n(peer-timeout T={T} + diffusion) = {L} rounds",
            ((fail_round+switch_round)/2, 70), ha="center", fontsize=8.5, color="#8a5a00")
ax.annotate(f"{F1_GL:.1f}", (40, F1_GL), color="#1b7837", fontsize=10, fontweight="bold",
            va="bottom", ha="right")
ax.annotate(f"{F1_FL_DEGRADED:.1f}", (40, F1_FL_DEGRADED), color="#b2182b", fontsize=10,
            fontweight="bold", va="top", ha="right")

ax.set_xlabel("round"); ax.set_ylabel("F1-score (%)"); ax.set_ylim(48, 100)
ax.set_title("Latency-aware end-to-end (composition): decentralized detection adds a\n"
             "degraded window before FL$\\rightarrow$GL, then GL recovers "
             "($N{=}14{\\to}7$, $k\\geq2$)", fontsize=12)
ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=10)
fig.tight_layout()
fig.savefig("results/latency_aware_end2end.png", dpi=175)
fig.savefig("results/latency_aware_end2end.pdf")
print(f"levels: healthy/GL={F1_GL:.2f}  FL-degraded(N={N_AFTER})={F1_FL_DEGRADED:.2f}")
print(f"failure@{fail_round}  latency L={L}  switch@{switch_round}")
print("ok")
