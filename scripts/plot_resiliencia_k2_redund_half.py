#!/usr/bin/env python3
"""Paper 1 (JPDC) Fig. 4 at HALF HEIGHT: churn resilience, FL k>=2 with redundant
specialists -> GL k>=2 (English variant used in the paper, without the switch marker).

Flat levels (FL->GL seeding makes the pool transparent to churn), network 14 -> 3:
  k>=2, 14 nodes (2/attack)       95.98  (recall ~100%)
  k>=1 (OR)                       95.74
  k>=2, no redundancy (1/attack)  87.63  (recall 78%)
Same data/style as results of plot_resiliencia_k2_redund.py; only height halved
(9.0 x 5.1 in -> 9.0 x 2.55 in).
Out: results/resiliencia_k2_redund_en_half.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "resiliencia_k2_redund_en_half")

NODES = [14, 12, 10, 8, 7, 6, 5, 3]
LEVELS = [
    ("$k\\geq2$, 14 nodes (2/attack)", 95.98, "#1d9e75", "-", 3.0),
    ("$k\\geq1$ (OR)", 95.74, "#2a78d6", "-", 2.2),
    ("$k\\geq2$, no redundancy (1/attack)", 87.63, "#eb6834", "--", 2.2),
]


def main():
    fig, ax = plt.subplots(figsize=(9.0, 2.55))
    for lab, val, col, ls, lw in LEVELS:
        ax.plot(NODES, [val] * len(NODES), color=col, ls=ls, lw=lw,
                marker="o", markersize=4.5, label=lab)
    ax.set_xticks(NODES)
    ax.invert_xaxis()   # 14 -> 3, left to right
    ax.set_ylim(84, 98)
    ax.set_yticks([84, 88, 92, 96])
    ax.set_xlabel("number of nodes  (network shrinking →)", fontsize=11)
    ax.set_ylabel("F1-score (%)", fontsize=11)
    ax.set_title("Resilience to churn: redundant FL $k\\geq2$ specialists → GL $k\\geq2$",
                 fontsize=11.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=8.5, framealpha=0.95)
    ax.annotate("recall ~100%", xy=(7, 95.98), xytext=(8.25, 92.2), fontsize=8.5,
                color="#0f6e56", arrowprops=dict(arrowstyle="->", color="#0f6e56"))
    ax.annotate("recall 78%", xy=(7, 87.63), xytext=(8.25, 89.4), fontsize=8.5,
                color="#993c1d", arrowprops=dict(arrowstyle="->", color="#993c1d"))
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print("wrote", OUT + ".{png,pdf}")


if __name__ == "__main__":
    main()
