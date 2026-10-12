#!/usr/bin/env python3
"""Detection under node loss with ONE variable: is the booster pool retained?

Replaces the four-curve figure of scripts/plot_transicao_adaptativa.py, which compared
FL->GL with GL->FL: those runs differ in start (federated vs cold gossip), pool size and
retention at once, and its labels attributed the gap to the direction of the switch.
Here both curves of a panel start from the same state at round 11 and follow the same
node-loss schedule (10 -> 3, one node every 10 rounds); they differ only in the pool:
  retained    every node keeps the union seeded at the switch (run flgl_*)
  re-formed   the aggregate is rebuilt each round from the nodes present (run glfl_*)
Two panels, side by side, for a single LaTeX figure at full text width:
(a) OR fusion (k = 1), (b) k >= 2. No in-plot titles; the sub-captions carry them.

Reads results/conv_adapt_shrink1.csv (same data as the original figure).
  python scripts/plot_retention_vs_reformed.py
Out: results/retention_vs_reformed.{pdf,png}
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "conv_adapt_shrink1.csv")
OUT = os.path.join(HERE, "results", "retention_vs_reformed")

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5, "legend.fontsize": 13})
BLUE, ORANGE = "#2a78d6", "#eb6834"
FIRST, LAST = 11, 90                                   # after the switch only
SHRINK = [20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5]
NODES = {15.5: 10, 25.5: 9, 35.5: 8, 45.5: 7, 55.5: 6, 65.5: 5, 75.5: 4, 85.5: 3}


def load():
    d = {}
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            d.setdefault(row["run"], {})[int(row["round"])] = float(row["f1"])
    return d


def panel(ax, d, k, tag):
    x = list(range(FIRST, LAST + 1))
    for r in SHRINK:
        ax.axvline(r, color="#d3d1c8", lw=0.9, ls=":", zorder=1)
    top = ax.secondary_xaxis("top")                    # active nodes, one label per stretch
    top.set_xticks(list(NODES))
    top.set_xticklabels([str(n) for n in NODES.values()])
    top.tick_params(length=0, labelsize=12, colors="#5f5d56")
    top.set_xlabel("active nodes", fontsize=12.5, color="#5f5d56", labelpad=6)
    ax.plot(x, [d[f"flgl_k{k}"][i] for i in x], color=BLUE, lw=3.0, label="pool retained at every node",
            solid_capstyle="round", zorder=3)
    ax.plot(x, [d[f"glfl_k{k}"][i] for i in x], color=ORANGE, lw=2.4, ls="--",
            label="aggregate re-formed from the nodes present", zorder=3)
    ax.text(0.02, 0.04, tag, transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_xlim(FIRST - 0.5, LAST + 0.5)
    ax.set_ylim(48, 100.5)
    ax.set_xlabel("round")
    ax.grid(alpha=0.22, zorder=0)


def main():
    d = load()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    panel(axes[0], d, 1, "(a)")
    panel(axes[1], d, 2, "(b)")
    axes[0].set_ylabel("F1-score (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print("wrote", OUT + ".png / .pdf")


if __name__ == "__main__":
    main()
