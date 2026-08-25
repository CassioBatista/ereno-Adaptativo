#!/usr/bin/env python3
"""Diagrama de nesting do protocolo (outer held-out / inner tuning).
Gera results/nesting_diagram.{pdf,png} para importar no Overleaf.
Rodar: ~/venv-ereno314/bin/python scripts/plot_nesting.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, RED, GREEN, GREY = "#2a78d6", "#d84a3a", "#3a8c3a", "#8a8880"

fig, ax = plt.subplots(figsize=(7.2, 8.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")


def box(x, y, w, h, title, lines, face, edge, tsize=11, lsize=9.3, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                 linewidth=1.7, edgecolor=edge, facecolor=face, linestyle=ls))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
            fontsize=tsize, fontweight="bold")
    if lines:
        ax.text(x + w / 2, y + h - 0.82, "\n".join(lines), ha="center", va="top",
                fontsize=lsize, linespacing=1.5)


def arrow(x1, y1, x2, y2, color="#444", cs=None):
    kw = dict(arrowstyle="-|>", mutation_scale=17, lw=1.7, color=color, shrinkA=2, shrinkB=2)
    if cs:
        kw["connectionstyle"] = cs
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **kw))


# OUTER frame (held-out)
ax.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 11.6, boxstyle="round,pad=0.1",
             linewidth=2.0, edgecolor="#333", facecolor="#f6f5f2", linestyle=(0, (6, 4))))
ax.text(0.55, 11.55, "OUTER  —  held-out, evaluated once",
        fontsize=11.5, fontweight="bold", color="#333")

# Train / Test (top row)
box(0.7, 9.2, 4.7, 1.9, "ERENO train", ["2,955,738 records"], "#dbe9f6", BLUE)
box(5.9, 9.2, 3.2, 1.9, "ERENO test", ["2,955,648", "held out"], "#f7e2dc", RED)

# INNER tuning (nested)
box(0.7, 3.9, 8.4, 4.4,
    "INNER  —  all tuning on the training split",
    ["benign cap  →  696,313 pool",
     "per-client 80/20  (local train / validation)",
     "GRASP feature selection (5-fold CV)  →  24 features",
     "λ penalty  →  tuned on validation",
     "k quorum,  threshold τ = 0.5  →  reported levers",
     "train N attack-specialist boosters"],
    "#e9f2e4", GREEN, tsize=11.5, lsize=9.6)

# Single evaluation
box(2.4, 1.0, 5.0, 1.7, "Single evaluation",
    ["F1 / recall / FPR / #FP", "on the held-out test"], "#f2eee5", GREY, tsize=10.5)

# arrows
arrow(3.05, 9.2, 3.6, 8.35)      # train -> inner
arrow(4.9, 3.9, 4.9, 2.72)       # inner -> eval
arrow(7.3, 2.0, 7.55, 9.2, color=RED, cs="arc3,rad=-0.32")  # eval -> test (curved, single shot)
ax.text(9.45, 5.6, "evaluate once", fontsize=8.8, color=RED, style="italic",
        ha="center", va="center", rotation=90)

os.makedirs("results", exist_ok=True)
fig.savefig("results/nesting_diagram.pdf", bbox_inches="tight")
fig.savefig("results/nesting_diagram.png", dpi=200, bbox_inches="tight")
print("wrote results/nesting_diagram.{pdf,png}")
