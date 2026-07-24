#!/usr/bin/env python3
"""Curvas de convergencia dos 4 runs adaptativos com encolhimento fino de nos.

FL->GL e GL->FL, 90 rounds, encolhe 1 cliente por vez (10->3), para k>=1 e k>=2.
Le results/conv_adapt_shrink1.csv (gerado dos logs adapt_{flgl,glfl}_{k1,k2}.log)
e gera results/conv_adapt_shrink1_metricas.png (paineis F1 / Recall / FPR).

Uso: python scripts/plot_conv_adapt_shrink1.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "conv_adapt_shrink1.csv")
OUT = os.path.join(HERE, "results", "conv_adapt_shrink1_metricas.png")

# run -> (rotulo, cor, estilo de linha)
STYLE = {
    "flgl_k1": ("FL->GL  k>=1", "#2a78d6", "-"),
    "glfl_k1": ("GL->FL  k>=1", "#2a78d6", "--"),
    "flgl_k2": ("FL->GL  k>=2", "#eb6834", "-"),
    "glfl_k2": ("GL->FL  k>=2", "#eb6834", "--"),
}

# fronteiras de fase: comuta em r10->11; encolhe a cada 10 rounds
SWITCH = 10.5
SHRINK = [20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5]
NODES = {25: 9, 35: 8, 45: 7, 55: 6, 65: 5, 75: 4, 85: 3}


def load():
    data = {r: {"round": [], "f1": [], "recall": [], "fpr": []} for r in STYLE}
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            d = data[row["run"]]
            d["round"].append(int(row["round"]))
            d["f1"].append(float(row["f1"]))
            d["recall"].append(float(row["recall"]))
            d["fpr"].append(float(row["fpr"]))
    return data


def main():
    data = load()
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    panels = [("f1", "F1-score (%)", (45, 101)),
              ("recall", "Recall (%)", (30, 101)),
              ("fpr", "FPR (%)", (-0.05, None))]

    for ax, (key, ylab, ylim) in zip(axes, panels):
        for run, (lab, col, ls) in STYLE.items():
            d = data[run]
            ax.plot(d["round"], d[key], color=col, linestyle=ls,
                    linewidth=2, label=lab)
        ax.axvline(SWITCH, color="#555", lw=1.2, ls=":")
        for x in SHRINK:
            ax.axvline(x, color="#bbb", lw=0.8, ls=":")
        ax.set_ylabel(ylab)
        if ylim[1] is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=ylim[0])
        ax.grid(alpha=0.25)

    axes[0].legend(loc="lower left", ncol=2, fontsize=9)
    axes[0].set_title("Adaptacao com encolhimento fino de nos (10->3) x k-de-n  "
                      "|  comuta em r10; -1 no a cada 10 rounds", fontsize=11)
    # anota o nº de nós no topo
    for x, n in NODES.items():
        axes[0].annotate(f"{n}", (x, 99.5), ha="center", fontsize=8, color="#777")
    axes[-1].set_xlabel("round")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
