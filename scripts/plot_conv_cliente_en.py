#!/usr/bin/env python3
"""Per-node convergence, FL (flat) vs. GL (diffusing), k = 1 -- English version of
scripts/plot_conv_cliente.py, with the size and fonts of
scripts/plot_retention_vs_reformed.py so that both figures match in the paper.

GL: one curve per node, rising as gossip brings it the other specialists, and their mean.
FL: the ten specialists, each partial and constant, and the aggregate (OR union).

Reads results/conv_por_cliente.csv (written by scripts/conv_por_cliente.py).
  python scripts/plot_conv_cliente_en.py
Out: results/conv_por_cliente_en.{pdf,png}
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "conv_por_cliente.csv")
OUT = os.path.join(HERE, "results", "conv_por_cliente_en")

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5, "legend.fontsize": 12})


def load():
    series = defaultdict(lambda: ([], []))
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            xs, ys = series[(row["serie"], row["tipo"])]
            xs.append(int(row["round"]))
            ys.append(float(row["recall"]))
    return series


def main():
    series = load()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for (name, kind), (xs, ys) in series.items():
        if kind == "gl_node":
            ax.plot(xs, ys, color="#85b7eb", lw=1, alpha=0.8, zorder=2)
        elif kind == "fl_node":
            ax.plot(xs, ys, color="#f0997b", lw=1, alpha=0.7, ls="--", zorder=1)
    ax.plot(*series[("GL_agg", "gl_agg")], color="#185fa5", lw=3.0,
            label="GL, mean over nodes (diffusion)", zorder=4)
    ax.plot(*series[("FL_agg", "fl_agg")], color="#0f6e56", lw=3.0,
            label="FL, aggregate (OR union)", zorder=4)
    ax.plot([], [], color="#85b7eb", lw=1.2, label="GL, per node (10)")
    ax.plot([], [], color="#f0997b", lw=1.2, ls="--", label="FL, per specialist (10)")
    ax.set_xlim(1, 14)
    ax.set_ylim(0, 102)
    ax.set_xlabel("gossip round")
    ax.set_ylabel("recall on the global test set (%)")
    ax.grid(alpha=0.22, zorder=0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cfcdc4")
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print("wrote", OUT + ".png / .pdf")


if __name__ == "__main__":
    main()
