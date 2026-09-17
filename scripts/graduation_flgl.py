#!/usr/bin/env python3
"""Graduation via FL/GL (Tier-2 zero-day -> Tier-1 specialist) — design figure.

The Tier-2 benign-spec detector stays replicated & local (every node detects
novelties in its own traffic, no SPOF). Only the DISCOVERY graduates: a corroborated
novel cluster becomes a new supervised specialist B_{N+1} that diffuses via the SAME
FL<->GL booster machinery. Two labelling paths:
  A · operator confirms + names (ground truth)      -> ACTIVE in production
  B · auto-label (weak): cluster=+ 'novel-#'         -> PROBATION -> validation -> ACTIVE
Out: results/graduation_flgl.{png,pdf}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GREY = dict(fc="#ececec", ec="#8a8a8a")
LAV = dict(fc="#eadcf5", ec="#6a3d9a")
GREEN = dict(fc="#d9efdd", ec="#3a7d4f")
AMBER = dict(fc="#f8e6cf", ec="#c07a29")
FLGL = dict(fc="#eef2ff", ec="#3b3b8f")

fig, ax = plt.subplots(figsize=(12.8, 10.8))
ax.set_xlim(0, 12.4); ax.set_ylim(0, 11.5); ax.axis("off")


def box(x, y, w, h, st, title, sub=None, tfs=11.5, sfs=8.6, tcol=None, lw=1.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                 fc=st["fc"], ec=st["ec"], lw=lw))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.63, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=tcol or st["ec"])
        ax.text(cx, y + h * 0.27, sub, ha="center", va="center", fontsize=sfs, color="0.25")
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=tcol or st["ec"])


def arrow(x0, y0, x1, y1, color="0.35", style="-", lw=2.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=17, color=color, lw=lw, ls=style, shrinkA=2, shrinkB=2))


def stage(y, text):
    ax.text(0.2, y, text, ha="left", va="center", fontsize=12, fontweight="bold", color="#333")


# ── 1 · Tier-2 fires (replicated) ─────────────────────────────────────────────
stage(11.05, "1 · Tier-2 fires — replicated on every node (local detection)")
for i, lab in enumerate(["node a", "node b", "node c"]):
    st = GREY if i == 0 else LAV
    box(0.5 + i * 1.5, 9.85, 1.3, 0.7, st, lab, "benign spec", tfs=9.5, sfs=7.6)
ax.text(1.55, 10.72, "✦ novel!", ha="center", va="center", fontsize=8.5,
        color="#c0392b", fontweight="bold")
ax.text(3.05, 10.72, "✦ novel!", ha="center", va="center", fontsize=8.5,
        color="#c0392b", fontweight="bold")
ax.text(5.5, 10.2, "same benign spec on all nodes →\nseveral flag the novelty independently",
        ha="left", va="center", fontsize=9, color="0.35", style="italic")

# ── 2 · buffer + corroborate ──────────────────────────────────────────────────
stage(9.35, "2 · Buffer + corroborate")
box(0.5, 8.5, 4.4, 0.72, GREY, "Buffer novel samples",
    "+ signature: field/family (VALUE / RATE)", tfs=10.5, sfs=8.2)
box(5.4, 8.5, 6.3, 0.72, LAV, "Corroboration  ≥ k nodes",
    "matching signature · rides the GLow control digest (v2)", tfs=10.5, sfs=8.2)
arrow(2.7, 9.85, 2.7, 9.22)                     # chips -> buffer
arrow(4.9, 8.86, 5.4, 8.86)                     # buffer -> corroboration

# ── 3 · label — two paths ─────────────────────────────────────────────────────
stage(7.75, "3 · Label — two paths")
box(0.7, 6.7, 4.8, 0.95, GREEN, "Path A — Operator",
    "confirms + names the class (ground truth)", tfs=11.5, sfs=8.4)
box(6.9, 6.7, 4.8, 0.95, AMBER, "Path B — Auto-label (weak)",
    "pseudo-label: cluster = +  'novel-#',  benign = −", tfs=11.5, sfs=8.4)
arrow(6.0, 8.5, 3.1, 7.65, color="#6a3d9a")     # corroboration -> A
arrow(9.3, 8.5, 9.3, 7.65, color="#6a3d9a")     # corroboration -> B

# ── 4 · train specialist ──────────────────────────────────────────────────────
stage(5.95, "4 · Train specialist  B_{N+1}")
box(0.7, 5.05, 4.8, 0.7, GREEN, "Train  B_{N+1}  (named)", tfs=11)
box(6.9, 5.05, 4.8, 0.7, AMBER, "Train  B_{N+1}  ('novel-#')", tfs=11)
ax.text(6.2, 5.4, "same XGBoost recipe\nGRASP-24 · nbr=10", ha="center", va="center",
        fontsize=7.8, color="0.4", style="italic")
arrow(3.1, 6.7, 3.1, 5.75)                      # A -> train
arrow(9.3, 6.7, 9.3, 5.75)                      # B -> train

# ── 5 · admit + diffuse via FL/GL  (THE HIGHLIGHT) ────────────────────────────
stage(4.2, "5 · Admit to pool + diffuse via  FL ⇄ GL   ★ reuses the booster diffusion")
ax.add_patch(FancyBboxPatch((0.7, 3.0), 11.0, 1.0, boxstyle="round,pad=0.03,rounding_size=0.10",
             fc=FLGL["fc"], ec="#c0392b", lw=2.6))
ax.text(6.2, 3.72, "Admit  B_{N+1}  to the content-dedup booster pool   +   diffuse via  FL ⇄ GL",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="#3b3b8f")
ax.text(3.55, 3.24, "FL: server unites → depth-1 (1 round)", ha="center", va="center",
        fontsize=8.8, color="#3b5b92", fontweight="bold")
ax.text(8.7, 3.24, "GL: gossip diffuses → ~Θ(N) rounds", ha="center", va="center",
        fontsize=8.8, color="#3a7d4f", fontweight="bold")
arrow(3.1, 5.05, 3.1, 4.0)                      # train A -> diffuse
arrow(9.3, 5.05, 9.3, 4.0)                      # train B -> diffuse

# ── 6 · steady state ──────────────────────────────────────────────────────────
stage(2.35, "6 · Steady state")
box(0.7, 1.35, 4.8, 0.8, GREEN, "→ ACTIVE in production", tfs=11)
box(6.9, 1.35, 4.8, 0.8, AMBER, "→ PROBATION → validation → ACTIVE",
    "provisional alarms until validated", tfs=10.5, sfs=8.2)
arrow(3.1, 3.0, 3.1, 2.15)                      # diffuse -> A active
arrow(9.3, 3.0, 9.3, 2.15)                      # diffuse -> B probation

# ── closing notes ─────────────────────────────────────────────────────────────
ax.text(6.2, 0.72,
        "Once diffused, the attack is KNOWN → Tier-1 catches it (k-of-n);  Tier-2 stops "
        "firing on it.   N → N+1.",
        ha="center", va="center", fontsize=9.4, color="0.2", fontweight="bold")
ax.text(6.2, 0.36,
        "The detector (benign spec) stays replicated & local — only the DISCOVERY graduates "
        "and diffuses via FL/GL.",
        ha="center", va="center", fontsize=9.2, color="0.45", style="italic")

ax.set_title("ReSIDS — graduation via FL/GL: a zero-day discovery becomes a diffused specialist",
             fontsize=13.5, fontweight="bold")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/graduation_flgl.png", dpi=165)
fig.savefig("results/graduation_flgl.pdf")
print("ok -> results/graduation_flgl.{png,pdf}")
