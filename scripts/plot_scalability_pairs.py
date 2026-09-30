#!/usr/bin/env python3
"""Paper 1 (JPDC) Figs. 6-9 as SIDE-BY-SIDE pairs, drawn at print width.

  pair A:  Fig. 6 (detection vs N)       | Fig. 7 (per-attack at N=98)
  pair B:  Fig. 8 (corroboration sweep)  | Fig. 9 (communication & storage vs N)

Each pair is 7.2 in wide (~3.5 in per panel), so fonts print at their true size
when the figure is placed at \\textwidth. Same data and styling as the originals;
legends/annotations compacted to fit the narrower panels. Each panel is also
exported on its own (3.6 in wide) for a two-minipage layout with separate captions.

Out: results/fig6_7_pair.{png,pdf}, results/fig8_9_pair.{png,pdf},
     results/fig{6,7,8,9}_panel.{png,pdf}
Run: ~/venv-ereno314/bin/python scripts/plot_scalability_pairs.py
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

R = "results/"
BOX = dict(boxstyle="round,pad=0.3", fc="#f4f2ee", ec="0.7")
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5})


def rows(name):
    with open(R + name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def p6(ax, t=""):
    d = rows("escalabilidade.csv")
    N = [int(r["N"]) for r in d]
    or_f1 = [float(r["OR_F1_GL"]) for r in d]
    ax.plot(N, [float(r["OR_rec_GL"]) for r in d], "-o", color="#2ca02c", lw=1.5, ms=3.5,
            label="Recall (OR = $k\\geq2$, GL = FL)")
    ax.plot(N, or_f1, "-s", color="#1f77b4", lw=1.5, ms=3.5, label="OR F1 (GL = FL)")
    ax.plot(N, [float(r["k2_F1_GL"]) for r in d], "--^", color="#e08214", lw=1.5, ms=4,
            label="$k\\geq2$ F1")
    ax.set_xlabel("Number of peers (N)")
    ax.set_ylabel("Detection metric (%)")
    ax.set_title(t + "Recall invariant; F1 erodes only under over-partitioning", fontsize=7.5)
    ax.set_ylim(90, 100.6)
    ax.set_yticks([90, 92, 94, 96, 98, 100])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.79), ncol=2, framealpha=0.92,
              handlelength=1.8, columnspacing=1.0)
    ax.annotate("OR F1 erosion at N=100\n(weak-data specialists; higher $k$ recovers it)",
                xy=(100, or_f1[-1]), xytext=(13, 90.35), ha="left", va="bottom",
                fontsize=6, color="#b2182b", style="italic",
                arrowprops=dict(arrowstyle="->", color="#b2182b", lw=0.9))


def p7(ax, t=""):
    d = rows("por_ataque_n98.csv")
    label = {"masquerade_fake_faul": "masq. fault", "masquerade_fake_norm": "masq. normal",
             "poisoned_high_rate": "poisoned", "inverse_replay": "inv. replay",
             "random_replay": "rand. replay", "injection": "injection", "high_StNum": "high StNum"}
    x = np.arange(len(d))
    blue, red = "#3b8fde", "#d9483b"
    b = ax.bar(x, [float(r["pos_per_spec"]) for r in d], width=0.8, color="#61a4e2",
               label="positives / specialist (train)")
    ax.set_yscale("log")
    ax.set_ylim(950, 3200)
    ax.set_yticks([1000, 2000])
    ax.set_yticklabels(["$10^3$", "$2\\times10^3$"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("positives per specialist (log)", color=blue)
    ax.tick_params(axis="y", colors=blue)
    ax.set_xticks(x)
    ax.set_xticklabels([label.get(r["attack"], r["attack"]) for r in d],
                       rotation=30, ha="right", fontsize=6.5)
    ax.set_xlabel("attack class  (fewer data → more data per specialist)")
    ax2 = ax.twinx()
    (ln,) = ax2.plot(x, [float(r["FP_rate_specs"]) for r in d], "-o", color=red, lw=1.7, ms=4,
                     label="specialists' FP rate on benign (%)")
    ax2.set_ylim(-0.02, 0.62)
    ax2.set_ylabel("FP rate of the attack's\nspecialists (%)", color=red)
    ax2.tick_params(axis="y", colors=red, labelsize=7)
    ax2.legend([b, ln], [b.get_label(), ln.get_label()], loc="upper center", ncol=1,
               framealpha=0.95)
    rec_min = min(float(r["recall_union"]) for r in d)
    ax.set_title(t + "N=98: data-poor specialists (masquerade)\n"
                 f"drive the FPs; union recall ~100% (min {rec_min:.1f}%)", fontsize=7.5)


def p8(ax, t=""):
    d = rows("kvary_n98.csv")
    k = [int(r["k"]) for r in d]
    f1 = [float(r["F1"]) for r in d]
    rec = [float(r["Recall"]) for r in d]
    prec = [float(r["Precision"]) for r in d]
    ax.plot(k, f1, "-o", color="#3a8ee0", lw=1.6, ms=3.5, label="F1")
    ax.plot(k, rec, "-s", color="#1f9e6e", lw=1.6, ms=3.5, label="Recall")
    ax.plot(k, prec, "-^", color="#b87318", lw=1.6, ms=3.5, label="Precision")
    ax.set_xticks(k)
    ax.set_ylim(83.5, 101.2)
    ax.set_yticks([84, 88, 92, 96, 100])
    ax.set_xlabel("corroboration threshold $k$ (14 specialists/class)")
    ax.set_ylabel("metric (%)")
    ax.set_title(t + "Precision–recall vs. $k$ ($N{=}98$)", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.text(0.13, 0.05,
            f"$k=1$ (OR) → $k=14$:\nF1 {f1[0]:.1f} → {f1[-1]:.1f}  ·  "
            f"Prec {prec[0]:.1f} → {prec[-1]:.1f}\nRecall ≥ {min(rec):.1f} throughout",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color="0.25", bbox=BOX)
    ax.legend(loc="lower right", ncol=1, framealpha=0.95)


def p9(ax, t=""):
    d = rows("escalabilidade.csv")
    N = [int(r["N"]) for r in d]
    cold = [float(r["GL_comm_bytes"]) / 1e6 for r in d]
    booster_kb = float(np.mean([float(r["avg_booster_KB"]) for r in d]))
    ax.plot(N, cold, "-o", color="#b87318", lw=1.6, ms=3.5, label="cold-diffusion total (MB)")
    ax.plot(N, [float(r["GL_bytes"]) / 1e6 for r in d], "-s", color="#3a8ee0", lw=1.6, ms=3.5,
            label="per-node footprint (MB)")
    ax.set_yscale("log")
    ax.set_ylim(0.15, 4000)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("Number of peers (N)")
    ax.set_ylabel("communication / storage (MB, log)")
    ax.set_title(t + "Compact boosters: MB-scale; seeded switch pays ~0", fontsize=7.5)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.text(0.97, 0.47,
            f"booster ≈ {booster_kb:.0f} KB (constant)\n"
            f"cold-diffusion @ N=50: {cold[N.index(50)]:.0f} MB\n"
            "vs P2P-secure baseline ∼196 Gb",
            transform=ax.transAxes, ha="right", va="center", fontsize=5.8, color="0.45", bbox=BOX)


def save(fig, name):
    fig.savefig(R + name + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(R + name + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("[pairs] ->", R + name + ".{png,pdf}")


def pair(fa, fb, name, height, ratios=(1, 1)):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, height), gridspec_kw={"width_ratios": ratios})
    fa(a, "(a) "); fb(b, "(b) ")
    fig.tight_layout(w_pad=1.6)
    save(fig, name)


def single(f, name, height):
    fig, ax = plt.subplots(figsize=(3.6, height))
    f(ax)
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    pair(p6, p7, "fig6_7_pair", 2.75)
    pair(p8, p9, "fig8_9_pair", 2.45)
    single(p6, "fig6_panel", 2.75); single(p7, "fig7_panel", 2.75)
    single(p8, "fig8_panel", 2.45); single(p9, "fig9_panel", 2.45)
