#!/usr/bin/env python3
"""Cost of running GL as the NORMAL mode: the new-attack detection-latency window
GROWS WITH N (while FL stays ~1 round). Reframes the convergence result from a
(tautological) "FL is faster" into a concrete, security-relevant cost that scales.

The recall LEVELS (ceiling ~99.64%, zero-day cross-firing baseline ~3.81%) are the
MEASURED values from the real N=12 ensemble (scripts/new_attack_convergence.py) and
do NOT depend on N -- they are properties of the specialist and the test set. What
scales with N is the DIFFUSION SPAN: the number of GLow gossip rounds for a novel
expert created at one node to reach the whole network. That is a topology property,
computed here from the actual GLow head-election neighbourhood-union diffusion on a
ring of N nodes (no retraining needed to measure a diffusion latency).

Message: as N grows, GL-as-normal imposes a detection-latency window ~N rounds for
new/emerging attacks -> at N=100 a novel attack needs ~100 rounds to be detected
network-wide. FL (central redistribution) closes it in ~1 round regardless of N.
Together with the communication overhead, this is why GL is a FALLBACK, not the
default -- FL is the efficient, timely normal mode WHILE the server is available.
Out: results/new_attack_scaling_{ramps,latency}.{png,pdf} + .csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.topology import build_topology
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NLIST = [10, 20, 50, 100]
CEILING = 99.64          # novel-attack recall of the expert (measured, N=12 run)
BASELINE = 3.81          # zero-day cross-firing baseline (measured, N=12 run)
SRC = 0                  # the node that creates the novel expert
MAXR = 400               # safety cap


def gl_diffusion(n):
    """GLow head-election neighbourhood-union spread of ONE expert from node SRC
    over a ring of n nodes. Returns reached-count per round (round 0 = injection)."""
    topo = build_topology("ring", n)
    reached = {SRC}
    counts = [len(reached)]           # r=0: only the source
    r = 1
    while len(reached) < n and r <= MAXR:
        head = (r - 1) % n                          # active = range(n); head round-robin
        if head not in reached and any(nb in reached for nb in topo.neighbors(head)):
            reached.add(head)
        counts.append(len(reached))
        r += 1
    return counts


rows = []
ramps = {}
for n in NLIST:
    counts = gl_diffusion(n)
    # GL network-mean recall(round) = baseline + (ceiling-baseline)*reached_fraction
    recall = [BASELINE + (CEILING - BASELINE) * (c / n) for c in counts]
    ramps[n] = recall
    lat_full = next((r for r, c in enumerate(counts) if c >= n), None)
    lat_90 = next((r for r, c in enumerate(counts) if c >= 0.9 * n), None)
    rows.append({"N": n, "gl_latency_full": lat_full, "gl_latency_90": lat_90,
                 "fl_latency": 1})
    print(f"N={n:3d}  GL full-network latency={lat_full:3d} rounds  "
          f"GL 90%-latency={lat_90:3d}  FL latency=1")

os.makedirs("results", exist_ok=True)
with open("results/new_attack_scaling.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["N", "gl_latency_full", "gl_latency_90", "fl_latency"])
    w.writeheader()
    for r in rows: w.writerow(r)

# ── Fig 1: GL-k1 ramps overlaid (the window widens with N); FL = 1-round step ──
fig, ax = plt.subplots(figsize=(9.4, 5.3))
colors = ["#1b7837", "#2166ac", "#e08214", "#b2182b"]
for n, col in zip(NLIST, colors):
    R = list(range(len(ramps[n])))
    ax.plot(R, ramps[n], "-", color=col, lw=2.2, label=f"GL, N={n} (window {rows[NLIST.index(n)]['gl_latency_full']} rounds)")
ax.plot([0, 1], [BASELINE, CEILING], "-o", color="0.15", lw=2.6, ms=5,
        label="FL, any N (~1 round)")
ax.axhline(CEILING, color="0.6", ls=":", lw=1)
ax.set_xlabel("rounds since the novel expert is ready (@1 node)")
ax.set_ylabel("network recall on the NOVEL attack (%)")
ax.set_xlim(0, max(rows[-1]["gl_latency_full"], 10) * 1.02)
ax.set_ylim(-3, 103)
ax.set_title("Communication and convergence are ONE axis: the hub gives FL depth-1\n"
             "for any $N$; GL trades degree for depth $\\Rightarrow$ dissemination window ~$N$",
             fontsize=11.5)
ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("results/new_attack_scaling_ramps.png", dpi=175)
fig.savefig("results/new_attack_scaling_ramps.pdf")

# ── Fig 2: latency vs N (GL ~linear, FL flat=1) ──
fig2, ax2 = plt.subplots(figsize=(8.6, 5.0))
Ns = [r["N"] for r in rows]
ax2.plot(Ns, [r["gl_latency_full"] for r in rows], "-o", color="#1b7837", lw=2.4, ms=8,
         label="GL: rounds to network-wide detection")
ax2.plot(Ns, [r["gl_latency_90"] for r in rows], "--^", color="#74c476", lw=1.8, ms=6,
         label="GL: rounds to 90% of nodes")
ax2.plot(Ns, [r["fl_latency"] for r in rows], "-s", color="#b2182b", lw=2.4, ms=8,
         label="FL: central redistribution (~1 round)")
ax2.plot(Ns, Ns, ":", color="0.6", lw=1.2, label="reference $y=N$")
ax2.set_xlabel("number of peers $N$")
ax2.set_ylabel("new-attack detection latency (rounds)")
ax2.set_title("One efficiency axis (dissemination work ~$\\Theta(N)$): the hub concentrates\n"
              "it at depth 1 (FL); decentralized GL pays depth ~$N$", fontsize=11.5)
ax2.grid(True, alpha=0.3); ax2.legend(loc="upper left", fontsize=9.5)
fig2.tight_layout()
fig2.savefig("results/new_attack_scaling_latency.png", dpi=175)
fig2.savefig("results/new_attack_scaling_latency.pdf")

print("\n[scaling] ok -> results/new_attack_scaling_{ramps,latency}.{png,pdf,csv}")
