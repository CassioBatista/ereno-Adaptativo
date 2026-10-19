#!/usr/bin/env python3
"""Runtime switching scenario (14 nodes): F1-score and recall per round, for OR and k >= 2,
with the phases marked. Each node is scored with the pool it holds, unreachable nodes
included (they keep detecting with their retained pool).

Default: mean over the deployed nodes. With --split, the 12 nodes deployed from round 1
and the 2 nodes that join at round 11 (the masquerade-as-fault pair, nodes 12 and 13) are
averaged separately: the joining nodes start with their own specialist only, until gossip
brings them the rest, which is what pulls the overall mean down at rounds 11-13.

Reads results/switching_scenario_n14.csv (scripts/switching_scenario.py).
  python scripts/plot_switching_scenario.py [--split]
Out: results/switching_scenario_n14.{pdf,png}, or results/switching_scenario_n14_split.{pdf,png}
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "switching_scenario_n14.csv")
OUT = os.path.join(HERE, "results", "switching_scenario_n14")
JOINING = {12, 13}

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5, "legend.fontsize": 12})
# (first round, last round, label, background)
PHASES = [(1, 10, "FL\n12 nodes", "#eef3fb"),
          (11, 20, "GL\nserver down", "#fbf1ea"),
          (21, 30, "GL\n8 unreach.", "#f6e3e3"),
          (31, 35, "GL", "#fbf1ea"),
          (36, 50, "FL\n14 nodes", "#eef3fb")]
COLOR = {1: "#2a78d6", 2: "#eb6834"}
KLABEL = {1: "OR ($k=1$)", 2: "$k\\geq2$"}


def load():
    d = defaultdict(list)                       # (k, round, group) -> [(f1, recall), ...]
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            k, r = int(row["k"]), int(row["round"])
            v = (float(row["f1"]), float(row["recall"]))
            d[(k, r, "all")].append(v)
            d[(k, r, "joining" if int(row["node"]) in JOINING else "initial")].append(v)
    return d


def series(d, k, group, mi):
    rounds = sorted({r for kk, r, g in d if kk == k and g == group})
    return rounds, [sum(v[mi] for v in d[(k, r, group)]) / len(d[(k, r, group)]) for r in rounds]


def main(split):
    d = load()
    last = max(r for _, r, _ in d)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, (mi, title) in zip(axes, [(0, "F1-score (%)"), (1, "recall (%)")]):
        for a, b, label, bg in PHASES:
            ax.axvspan(a - 0.5, b + 0.5, color=bg, zorder=0, lw=0)
        for k in (1, 2):
            ls = "-" if k == 1 else "--"
            if not split:
                ax.step(*series(d, k, "all", mi), where="mid", color=COLOR[k], ls=ls, lw=2.6,
                        label=KLABEL[k], zorder=3)
                continue
            ax.step(*series(d, k, "initial", mi), where="mid", color=COLOR[k], ls=ls, lw=2.6,
                    label=KLABEL[k] + ", 12 initial nodes", zorder=3)
            ax.step(*series(d, k, "joining", mi), where="mid", color=COLOR[k], ls=":", lw=1.8,
                    label=KLABEL[k] + ", 2 joining nodes", zorder=2)
        ax.set_title(title, fontsize=15, pad=36)
        ax.set_ylim(*((40, 101) if split else (75, 101)))
        if split:   # k >= 2 on the joining nodes is 0 at round 11 (one booster, one vote)
            ax.annotate("$k\\geq2$, joining nodes:\n0 at round 11", xy=(11, 40.5),
                        xytext=(13.2, 46), fontsize=10, color=COLOR[2],
                        arrowprops=dict(arrowstyle="->", color=COLOR[2], lw=1))
        ax.grid(alpha=0.22, zorder=0)
        ax.set_xlim(0.5, last + 0.5)
        ax.set_xlabel("round")
        for a, b, label, _ in PHASES:
            ax.text((a + b) / 2, 1.01, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10, color="#5f5d56")
    axes[0].set_ylabel("%")
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cfcdc4",
                   fontsize=11 if split else 12)
    fig.tight_layout()
    out = OUT + ("_split" if split else "")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("wrote", out + ".png / .pdf")


if __name__ == "__main__":
    main("--split" in sys.argv[1:])
