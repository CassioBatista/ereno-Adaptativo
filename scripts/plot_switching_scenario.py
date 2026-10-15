#!/usr/bin/env python3
"""Runtime switching scenario (14 nodes): F1-score and recall per round, mean over the
deployed nodes, for OR and k >= 2, with the phases marked. Each node is scored with the pool
it holds, unreachable nodes included (they keep detecting with their retained pool).

Reads results/switching_scenario_n14.csv (scripts/switching_scenario.py).
  python scripts/plot_switching_scenario.py
Out: results/switching_scenario_n14.{pdf,png}
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "switching_scenario_n14.csv")
OUT = os.path.join(HERE, "results", "switching_scenario_n14")

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5, "legend.fontsize": 12})
# (first round, last round, label, background)
PHASES = [(1, 10, "FL\n12 nodes", "#eef3fb"),
          (11, 20, "GL\nserver down", "#fbf1ea"),
          (21, 30, "GL\n8 unreach.", "#f6e3e3"),
          (31, 35, "GL", "#fbf1ea"),
          (36, 50, "FL\n14 nodes", "#eef3fb")]
STYLE = {1: dict(color="#2a78d6", ls="-", label="OR ($k=1$)"),
         2: dict(color="#eb6834", ls="--", label="$k\\geq2$")}


def load():
    d = defaultdict(list)                       # (k, round) -> [(f1, recall), ...]
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            d[(int(row["k"]), int(row["round"]))].append((float(row["f1"]), float(row["recall"])))
    return d


def main():
    d = load()
    rounds = sorted({r for _, r in d})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, (mi, ylabel) in zip(axes, [(0, "F1-score (%)"), (1, "recall (%)")]):
        for a, b, label, bg in PHASES:
            ax.axvspan(a - 0.5, b + 0.5, color=bg, zorder=0, lw=0)
        for k, st in STYLE.items():
            vals = [[v[mi] for v in d[(k, r)]] for r in rounds]
            mean = [sum(v) / len(v) for v in vals]
            ax.step(rounds, mean, where="mid", color=st["color"], ls=st["ls"], lw=2.6,
                    label=st["label"], zorder=3)
        ax.set_title(ylabel, fontsize=15, pad=36)
        ax.set_ylim(75, 101)
        ax.grid(alpha=0.22, zorder=0)
        ax.set_xlim(0.5, rounds[-1] + 0.5)
        ax.set_xlabel("round")
        for a, b, label, _ in PHASES:
            ax.text((a + b) / 2, 1.01, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10, color="#5f5d56")
    axes[0].set_ylabel("%")
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cfcdc4")
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print("wrote", OUT + ".png / .pdf")


if __name__ == "__main__":
    main()
