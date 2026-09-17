#!/usr/bin/env python3
"""ReSIDS two-tier detection architecture (Paper 1 revision figure).

Extends the existing 3-stage diagram (attack specialists -> FL/GL aggregation ->
k-of-n) with a parallel Tier 2: a benign-only anomaly detector for zero-day attacks
(no specialist). A novel alarm graduates to a specialist once labelled.
Out: results/architecture_two_tier.{png,pdf}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GREY = dict(fc="#ececec", ec="#8a8a8a")
BLUE = dict(fc="#cfe0f3", ec="#3b5b92")
GREEN = dict(fc="#d9efdd", ec="#3a7d4f")
ORANGE = dict(fc="#f8e6cf", ec="#c07a29")
LAV = dict(fc="#eadcf5", ec="#6a3d9a")

fig, ax = plt.subplots(figsize=(12.4, 9.6))
ax.set_xlim(0, 12.4); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, st, title, sub=None, tfs=12, sfs=9.5, tcol=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                 fc=st["fc"], ec=st["ec"], lw=1.7))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.62, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=tcol or st["ec"])
        ax.text(cx, y + h * 0.28, sub, ha="center", va="center", fontsize=sfs, color="0.25")
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=tcol or st["ec"])

def arrow(x0, y0, x1, y1, color="0.35", style="-", lw=2.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=18, color=color, lw=lw, ls=style,
                 shrinkA=2, shrinkB=2))

def stage(x, y, text, col="#333"):
    ax.text(x, y, text, ha="left", va="center", fontsize=12.5, fontweight="bold", color=col)

# ── Stage 1 ──────────────────────────────────────────────────────────────────
stage(0.2, 9.55, "1 · Detection — two tiers")
# Tier 1: specialists
ax.text(0.4, 9.05, "Tier 1 — attack specialists (supervised)", ha="left", fontsize=10,
        color="0.3", style="italic")
for i, lab in enumerate(["B₁\nattack 1", "B₂\nattack 2", "B₃\nattack 3", "… B_N\nattack N"]):
    box(0.4 + i * 1.55, 8.15, 1.4, 0.75, GREY, lab, tfs=10.5)
# Tier 2: anomaly
ax.text(7.5, 9.05, "Tier 2 — zero-day detector (unsupervised)", ha="left", fontsize=10,
        color="0.3", style="italic")
box(7.5, 8.05, 4.6, 0.95, LAV, "Anomaly model  A",
    "Isolation Forest / autoencoder  ·  trained on BENIGN only", tfs=12, sfs=8.8)

# ── Stage 2 ──────────────────────────────────────────────────────────────────
stage(0.2, 7.35, "2 · Aggregation & scoring")
# Tier 1: FL/GL
ax.add_patch(FancyBboxPatch((0.4, 5.55, ), 6.4, 1.35 - 0.0,
             boxstyle="round,pad=0.03,rounding_size=0.08", fc="#f6f6f6", ec="0.75",
             lw=1.2, ls=(0, (5, 4))))
box(0.7, 5.75, 2.6, 0.95, BLUE, "Federated", "server unites boosters", tfs=11, sfs=8.5)
box(4.0, 5.75, 2.6, 0.95, GREEN, "Gossip", "peers diffuse boosters", tfs=11, sfs=8.5)
ax.annotate("⇄", (3.55, 6.35), ha="center", va="center", fontsize=20, color="#c0392b")
ax.text(3.55, 5.95, "runtime", ha="center", va="center", fontsize=8, color="#c0392b")
box(1.6, 4.35, 4.0, 0.8, GREY, "Booster pool", "content-dedup union {B₁ … B_M}", tfs=11, sfs=8.8)
# Tier 2: novelty score
box(7.5, 5.9, 4.6, 0.8, LAV, "novelty score  s(x) ≥ τ", tfs=11.5)
box(7.5, 4.75, 4.6, 0.75, LAV, "k-of-n over peers' anomaly verdicts",
    "(corroboration — optional)", tfs=10, sfs=8.5)

# ── Stage 3 ──────────────────────────────────────────────────────────────────
stage(0.2, 3.55, "3 · Decision — two-tier")
box(0.7, 1.75, 8.2, 1.5, ORANGE, "", tfs=12)
ax.text(4.8, 2.92, "Decision", ha="center", va="center", fontsize=12.5, fontweight="bold",
        color=ORANGE["ec"])
ax.text(4.8, 2.42, "known attack:  votes ≥ k   →  ALARM (attack type)", ha="center",
        va="center", fontsize=10.5, color="0.15")
ax.text(4.8, 2.02, "zero-day:  anomalous  AND  unclaimed by any specialist  →  ALARM (novel)",
        ha="center", va="center", fontsize=10.5, color="#6a3d9a", fontweight="bold")
ax.text(0.62, 2.5, "sample x", ha="right", va="center", fontsize=9.5, color="0.4")
arrow(0.62, 2.5, 0.7, 2.5, color="0.4")
ax.text(10.6, 2.5, "ALARM", ha="center", va="center", fontsize=13, fontweight="bold",
        color="#c0392b")
arrow(8.9, 2.5, 9.9, 2.5, color="#c0392b", lw=2.4)

# ── arrows ───────────────────────────────────────────────────────────────────
arrow(3.6, 8.1, 3.6, 6.95)                          # specialists -> aggregation
arrow(3.6, 5.55, 3.6, 5.17)                         # FL/GL -> pool
arrow(3.6, 4.33, 3.6, 3.27)                         # pool -> decision
arrow(9.8, 8.03, 9.8, 6.72)                         # anomaly -> novelty
arrow(9.8, 5.88, 9.8, 5.52)                         # novelty -> corroboration
arrow(9.8, 4.73, 6.2, 3.05, color="#6a3d9a")        # Tier2 -> decision (novel)

# graduation loop: novel alarm -> (label) -> new specialist
grad = FancyArrowPatch((8.9, 2.0), (6.3, 8.4), connectionstyle="arc3,rad=-0.32",
                       arrowstyle="-|>", mutation_scale=16, color="#6a3d9a",
                       lw=1.6, ls=(0, (6, 3)))
ax.add_patch(grad)
ax.text(6.75, 3.95, "graduation: novel → (operator labels) → new specialist B_{N+1}",
        ha="left", va="center", fontsize=8.6, color="#6a3d9a", style="italic")

ax.text(6.2, 0.95,
        "Tier 1 (supervised, known attacks; FL = GL under OR, runtime switch)  +  "
        "Tier 2 (unsupervised, benign-only, zero-day).",
        ha="center", va="center", fontsize=9.2, color="0.45", style="italic")
ax.text(6.2, 0.6,
        "A novel alarm graduates to a specialist once labelled, then diffuses via FL/GL.",
        ha="center", va="center", fontsize=9.2, color="0.45", style="italic")

ax.set_title("ReSIDS — two-tier detection: known attacks (k-of-n) + zero-day (anomaly)",
             fontsize=13.5, fontweight="bold")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/architecture_two_tier.png", dpi=165)
fig.savefig("results/architecture_two_tier.pdf")
print("ok -> results/architecture_two_tier.{png,pdf}")
