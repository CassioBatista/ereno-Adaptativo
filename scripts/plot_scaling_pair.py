#!/usr/bin/env python3
"""Scalability and communication cost, N = 10 -> 100, as one two-panel figure (Section 4.5).

  (a) detection vs N under OR: recall and F1-score (gossip and federation coincide under OR)
  (b) communication and storage vs N: total bytes of a gossip diffusion from a cold start,
      and per-node pool footprint

Replaces the four panels of the former Figures 8-9 (scripts/plot_scalability_pairs.py in
the v2 line): the per-attack breakdown at N=98 is summarised in the text, and the k sweep
at N=98 is dropped (its best k was read on the test set). Size and fonts match
scripts/plot_switching_scenario.py.

Reads results/escalabilidade.csv (written by scripts/vote_comm_analysis.py).
  python scripts/plot_scaling_pair.py
Out: results/scaling_pair.{pdf,png}
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "escalabilidade.csv")
OUT = os.path.join(HERE, "results", "scaling_pair")

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5, "legend.fontsize": 12})
BOX = dict(boxstyle="round,pad=0.35", fc="#f7f6f2", ec="#cfcdc4")


def main():
    with open(CSV, encoding="utf-8") as fh:
        d = list(csv.DictReader(fh))
    N = [int(r["N"]) for r in d]
    col = lambda name: [float(r[name]) for r in d]  # noqa: E731
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.6))

    # OR only: the k >= 2 columns of this file score the gossip pool built from a cold start
    # (about 2N - 1 boosters, warm-start variants included), not the federated pool
    a.plot(N, col("OR_rec_GL"), "-o", color="#1f9e6e", lw=2.4, ms=6, label="recall")
    a.plot(N, col("OR_F1_GL"), "-s", color="#2a78d6", lw=2.4, ms=6, label="F1-score")
    a.set_xlabel("number of nodes $N$")
    a.set_ylabel("%")
    a.set_ylim(89.5, 100.8)
    a.grid(alpha=0.22)
    a.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#cfcdc4")
    a.set_title("(a) detection, OR fusion", fontsize=15)

    cold = [v / 1e6 for v in col("GL_comm_bytes")]
    a_kb = sum(col("avg_booster_KB")) / len(d)
    b.plot(N, cold, "-o", color="#b87318", lw=2.4, ms=6, label="gossip diffusion from a cold start")
    b.plot(N, [v / 1e6 for v in col("GL_bytes")], "-s", color="#2a78d6", lw=2.4, ms=6,
           label="per-node pool")
    b.set_yscale("log")
    b.set_ylim(0.05, 3000)
    b.set_xlabel("number of nodes $N$")
    b.set_ylabel("MB (log scale)")
    b.grid(alpha=0.22, which="both")
    b.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cfcdc4")
    b.text(0.97, 0.06, f"one booster $\\approx$ {a_kb:.0f} KB", transform=b.transAxes,
           ha="right", va="bottom", fontsize=12, color="#5f5d56", bbox=BOX)
    b.set_title("(b) communication and storage", fontsize=15)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print("wrote", OUT + ".png / .pdf")


if __name__ == "__main__":
    main()
