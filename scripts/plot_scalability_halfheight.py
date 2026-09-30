#!/usr/bin/env python3
"""Paper 1 (JPDC) Figs. 6-9 regenerated at HALF HEIGHT (same width, data and style).

  Fig. 6  Detection quality vs. number of peers   <- results/escalabilidade.csv
  Fig. 7  Per-attack breakdown at N=98             <- results/por_ataque_n98.csv
  Fig. 8  Corroboration Pareto sweep at N=98       <- results/kvary_n98.csv
  Fig. 9  Communication and storage vs. N          <- results/escalabilidade.csv

Only the vertical extent changes: legends go to single rows and annotations are
moved to empty regions so nothing overlaps the data at the reduced height.
Out: results/{escalabilidade,por_ataque_n98,kvary_n98,escalabilidade_comunicacao}_half.{png,pdf}
Run: ~/venv-ereno314/bin/python scripts/plot_scalability_halfheight.py
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

R = "results/"
BOX = dict(boxstyle="round,pad=0.35", fc="#f4f2ee", ec="0.7")


def rows(name):
    with open(R + name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    fig.savefig(R + name + "_half.png", dpi=200, bbox_inches="tight")
    fig.savefig(R + name + "_half.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[half] ->", R + name + "_half.{png,pdf}")


def fig6():
    d = rows("escalabilidade.csv")
    N = [int(r["N"]) for r in d]
    rec = [float(r["OR_rec_GL"]) for r in d]
    or_f1 = [float(r["OR_F1_GL"]) for r in d]
    k2_f1 = [float(r["k2_F1_GL"]) for r in d]
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.plot(N, rec, "-o", color="#2ca02c", lw=1.8, ms=4, label="Recall (OR = $k\\geq2$, GL = FL)")
    ax.plot(N, or_f1, "-s", color="#1f77b4", lw=1.8, ms=4, label="OR F1 (GL = FL)")
    ax.plot(N, k2_f1, "--^", color="#e08214", lw=1.8, ms=5, label="$k\\geq2$ F1")
    ax.set_xlabel("Number of peers (N)")
    ax.set_ylabel("Detection metric (%)", fontsize=9)
    ax.set_title("Recall invariant; F1 erodes only under over-partitioning", fontsize=10.5)
    ax.set_ylim(90, 100.6)
    ax.set_yticks([90, 92, 94, 96, 98, 100])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.78), ncol=3, fontsize=8, framealpha=0.92)
    ax.annotate("OR F1 erosion at N=100 (weak-data specialists;\n"
                "higher $k$ on the seeded pool recovers it)",
                xy=(100, or_f1[-1]), xytext=(22, 90.45), ha="left", va="bottom",
                fontsize=7.5, color="#b2182b", style="italic",
                arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1))
    fig.tight_layout()
    save(fig, "escalabilidade")


def fig7():
    d = rows("por_ataque_n98.csv")
    label = {"masquerade_fake_faul": "masq. fault", "masquerade_fake_norm": "masq. normal",
             "poisoned_high_rate": "poisoned", "inverse_replay": "inv. replay",
             "random_replay": "rand. replay", "injection": "injection", "high_StNum": "high StNum"}
    names = [label.get(r["attack"], r["attack"]) for r in d]
    pos = [float(r["pos_per_spec"]) for r in d]
    fp = [float(r["FP_rate_specs"]) for r in d]
    rec_min = min(float(r["recall_union"]) for r in d)
    x = np.arange(len(d))
    blue, red = "#3b8fde", "#d9483b"

    fig, ax = plt.subplots(figsize=(10.4, 2.9))
    b = ax.bar(x, pos, width=0.8, color="#61a4e2", label="positives / specialist (train)")
    ax.set_yscale("log")
    ax.set_ylim(950, 3000)
    ax.set_yticks([1000, 2000])
    ax.set_yticklabels(["$10^3$", "$2\\times10^3$"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("positives per\nspecialist (log)", color=blue, fontsize=9.5)
    ax.tick_params(axis="y", colors=blue)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_xlabel("attack class  (fewer data  →  more data per specialist)", fontsize=10)

    ax2 = ax.twinx()
    (ln,) = ax2.plot(x, fp, "-o", color=red, lw=2.2, ms=6, label="specialists' FP rate on benign (%)")
    ax2.set_ylim(-0.02, 0.58)
    ax2.set_ylabel("FP rate of the attack's\nspecialists (%)", color=red, fontsize=9.5)
    ax2.tick_params(axis="y", colors=red)
    ax2.legend([b, ln], [b.get_label(), ln.get_label()], loc="upper center", ncol=2,
               fontsize=8.5, framealpha=0.95)
    ax.set_title("N=98: data-poor specialists (masquerade) drive the false positives\n"
                 f"the precision cost is upstream of fusion; union recall stays ~100% "
                 f"(min {rec_min:.1f}%)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "por_ataque_n98")


def fig8():
    d = rows("kvary_n98.csv")
    k = [int(r["k"]) for r in d]
    f1 = [float(r["F1"]) for r in d]
    rec = [float(r["Recall"]) for r in d]
    prec = [float(r["Precision"]) for r in d]
    fig, ax = plt.subplots(figsize=(8.2, 2.4))
    ax.plot(k, f1, "-o", color="#3a8ee0", lw=2, ms=5, label="F1")
    ax.plot(k, rec, "-s", color="#1f9e6e", lw=2, ms=5, label="Recall")
    ax.plot(k, prec, "-^", color="#b87318", lw=2, ms=5, label="Precision")
    ax.set_xticks(k)
    ax.set_ylim(83.5, 101.2)
    ax.set_yticks([84, 88, 92, 96, 100])
    ax.set_xlabel("corroboration threshold $k$  (N=98, 14 specialists/attack)")
    ax.set_ylabel("metric (%)")
    ax.set_title("N=98 (14 specialists/attack): precision-recall frontier over $k$", fontsize=10.5)
    ax.grid(True, alpha=0.3)
    ax.text(0.16, 0.06,
            f"$k=1$ (OR) → $k=14$:   F1 {f1[0]:.1f} → {f1[-1]:.1f}  ·  "
            f"Prec {prec[0]:.1f} → {prec[-1]:.1f}\n"
            f"Recall ≥ {min(rec):.1f} throughout",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="0.25", bbox=BOX)
    ax.legend(loc="lower right", ncol=3, fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    save(fig, "kvary_n98")


def fig9():
    d = rows("escalabilidade.csv")
    N = [int(r["N"]) for r in d]
    cold = [float(r["GL_comm_bytes"]) / 1e6 for r in d]
    node = [float(r["GL_bytes"]) / 1e6 for r in d]
    booster_kb = float(np.mean([float(r["avg_booster_KB"]) for r in d]))
    cold50 = cold[N.index(50)]
    fig, ax = plt.subplots(figsize=(6.3, 2.2))
    ax.plot(N, cold, "-o", color="#b87318", lw=2, ms=5, label="cold-diffusion total (MB)")
    ax.plot(N, node, "-s", color="#3a8ee0", lw=2, ms=5, label="per-node footprint (MB)")
    ax.set_yscale("log")
    ax.set_ylim(0.15, 4000)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("Number of peers (N)", fontsize=9.5)
    ax.set_ylabel("communication /\nstorage (MB, log)", fontsize=9)
    ax.set_title("Compact boosters: MB-scale; seeded switch pays ~0", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95)
    ax.text(0.97, 0.47,
            f"booster ≈ {booster_kb:.0f} KB (constant)\n"
            f"cold-diffusion @ N=50: {cold50:.0f} MB\n"
            "vs P2P-secure baseline ∼196 Gb",
            transform=ax.transAxes, ha="right", va="center", fontsize=7, color="0.45", bbox=BOX)
    fig.tight_layout()
    save(fig, "escalabilidade_comunicacao")


if __name__ == "__main__":
    fig6(); fig7(); fig8(); fig9()
