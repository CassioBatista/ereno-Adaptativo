#!/usr/bin/env python3
"""F1 e Recall x numero de nos (encolhimento 10->3) para os 4 runs adaptativos.

Le results/conv_adapt_shrink1.csv, extrai o bloco de cada contagem de nos
(pos-comutacao: r20=10, r30=9, ..., r90=3) e gera
results/shrink_por_no.png (paineis F1 e Recall x nos).

Uso: python scripts/plot_shrink_por_no.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "conv_adapt_shrink1.csv")
OUT = os.path.join(HERE, "results", "shrink_por_no.png")

# round representativo -> nº de nós (pos-comutacao)
ROUND2NODES = {20: 10, 30: 9, 40: 8, 50: 7, 60: 6, 70: 5, 80: 4, 90: 3}
STYLE = {
    "flgl_k1": ("FL->GL  k>=1", "#2a78d6", "-"),
    "glfl_k1": ("GL->FL  k>=1", "#2a78d6", "--"),
    "flgl_k2": ("FL->GL  k>=2", "#eb6834", "-"),
    "glfl_k2": ("GL->FL  k>=2", "#eb6834", "--"),
}


def load():
    data = {r: {} for r in STYLE}   # run -> {nodes: (f1, recall)}
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rnd = int(row["round"])
            if rnd in ROUND2NODES and row["run"] in data:
                data[row["run"]][ROUND2NODES[rnd]] = (
                    float(row["f1"]), float(row["recall"]))
    return data


def main():
    data = load()
    nodes = sorted(ROUND2NODES.values(), reverse=True)   # 10..3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, idx, ylab, ylim in [(ax1, 0, "F1-score (%)", (45, 101)),
                                (ax2, 1, "Recall (%)", (30, 101))]:
        for run, (lab, col, ls) in STYLE.items():
            ys = [data[run][n][idx] for n in nodes]
            ax.plot(nodes, ys, color=col, linestyle=ls, linewidth=2,
                    marker="o", markersize=5, label=lab)
        ax.set_xlabel("nº de nós")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.invert_xaxis()   # 10 -> 3 (encolhimento da esquerda p/ direita)
        ax.grid(alpha=0.25)
    ax1.legend(loc="lower left", ncol=2, fontsize=9)
    ax1.set_title("Encolhimento 10->3 nos  |  FL<->GL x k-de-n", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
