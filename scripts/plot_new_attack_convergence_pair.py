#!/usr/bin/env python3
"""Side-by-side version of the two new-attack convergence figures, for a single LaTeX
figure at full text width: (a) one new specialist, (b) with the second specialist at r16.
Reads the CSVs written by scripts/new_attack_convergence.py (same data, same styling;
fonts sized for half-width panels, one shared legend, no in-plot titles -- the
sub-captions carry them).

  python scripts/plot_new_attack_convergence_pair.py
Out: results/new_attack_convergence_pair.{pdf,png}
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INJECT_ROUND, SECOND_ROUND = 11, 16


def load(path):
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def panel(ax, rows, two_source, tag):
    R = [r["round"] for r in rows]
    base = rows[0]["fl_k1"]
    ax.axvspan(1, INJECT_ROUND, color="0.85", alpha=0.5, lw=0)
    ax.text(1.5, 58, f"zero-day baseline\n(no expert:\ncross-firing ~{base:.1f}%)",
            fontsize=8, color="0.35", va="center")
    ax.axvline(INJECT_ROUND, color="0.4", ls=":", lw=1)
    ax.text(INJECT_ROUND - 1.2, 86, f"novel expert\n@node 0, r{INJECT_ROUND}",
            fontsize=8, color="0.35", ha="right", va="center")
    if two_source:
        ax.axvline(SECOND_ROUND, color="#6a3d9a", ls=":", lw=1)
        ax.annotate(f"2nd source\n@node 6, r{SECOND_ROUND}", xy=(SECOND_ROUND, 9),
                    xytext=(24, 18), fontsize=8, color="#6a3d9a", va="center",
                    arrowprops={"arrowstyle": "->", "color": "#6a3d9a", "lw": 0.8})
    ax.plot(R, [r["fl_k1"] for r in rows], "-o", color="#b2182b", lw=2.0, ms=3.5,
            label=r"FL, $k\geq1$")
    ax.plot(R, [r["gl_k1"] for r in rows], "-o", color="#1b7837", lw=2.0, ms=3.5,
            label=r"GL, $k\geq1$")
    ax.plot(R, [r["fl_k2"] for r in rows], "--s", color="#d6604d", lw=1.7, ms=3,
            label=r"FL, $k\geq2$")
    ax.plot(R, [r["gl_k2"] for r in rows], "--^", color="#74c476", lw=1.7, ms=3,
            label=r"GL, $k\geq2$")
    ax.set_xlabel("round", fontsize=10)
    ax.set_ylim(-3, 104)
    ax.set_xlim(0, R[-1] + 1)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=9)
    ax.text(0.01, 1.02, tag, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom")


def main():
    r1 = load("results/new_attack_convergence_1src.csv")
    r2 = load("results/new_attack_convergence_2src.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    panel(a, r1, False, "(a)")
    panel(b, r2, True, "(b)")
    a.set_ylabel("recall on the novel attack (%)", fontsize=10)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"results/new_attack_convergence_pair.{ext}", dpi=300, bbox_inches="tight")
    print("results/new_attack_convergence_pair.{pdf,png}")


if __name__ == "__main__":
    main()
