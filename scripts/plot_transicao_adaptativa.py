#!/usr/bin/env python3
"""IDS adaptativo: comutacao de arquitetura (r11) sob encolhimento 10->3 nos.
Versao por-round, legivel (ingles, fontes grandes, sem titulo interno).

Le results/conv_adapt_shrink1.csv (4 runs: FL->GL e GL->FL, k=1 e k>=2) e gera
results/transicao_adaptativa.{png,pdf}. Contador de nos ATIVOS no topo: 10->3
(1 no a cada 10 rounds). Sem titulo -> vai no \\caption{} do LaTeX.

Uso: python3 scripts/plot_transicao_adaptativa.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "conv_adapt_shrink1.csv")
OUT = os.path.join(HERE, "results", "transicao_adaptativa")

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 16.5, "axes.titlesize": 15.5,
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "legend.fontsize": 12.5,
})

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d84a3a"
# run -> (rotulo, cor, estilo, largura)  | solido = FL->GL (transparente)
STYLE = {
    "flgl_k1": ("FL$\\to$GL,  k$\\geq$1",  BLUE,   "-",  3.2),
    "flgl_k2": ("FL$\\to$GL,  k$\\geq$2",  ORANGE, "-",  3.2),
    "glfl_k1": ("GL$\\to$FL,  k$\\geq$1",  BLUE,   "--", 2.0),
    "glfl_k2": ("GL$\\to$FL,  k$\\geq$2",  ORANGE, "--", 2.0),
}
SWITCH = 10.5
SHRINK = [20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5]
# encolhimento de 1 no a cada 10 rounds: 10 -> 3 nos (rotulo no meio de cada faixa)
NODES = {15: 10, 25: 9, 35: 8, 45: 7, 55: 6, 65: 5, 75: 4, 85: 3}


def load():
    d = {r: {} for r in STYLE}
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["run"] in d:
                d[row["run"]][int(row["round"])] = float(row["f1"])
    return {r: [d[r][i] for i in range(1, 91)] for r in STYLE}


def main():
    data = load()
    x = list(range(1, 91))
    fig, ax = plt.subplots(figsize=(12.4, 6.4))

    # fase inicial (from-scratch) sombreada
    ax.axvspan(0.5, SWITCH, color="#ecebe6", alpha=0.7, zorder=0)
    ax.text(5.5, 98.6, "cold\nstart", ha="center", va="center",
            fontsize=11, color="#8a8880", style="italic")

    # limites de encolhimento + contador de nos ATIVOS no topo (10 -> 3)
    for r in SHRINK:
        ax.axvline(r, color="#d3d1c8", lw=0.9, ls=":", zorder=1)
    ax.text(1.0, 102.4, "active nodes:", ha="left", va="center",
            fontsize=11.5, color="#8a8880")
    for xr, n in NODES.items():
        ax.text(xr, 102.4, f"{n}", ha="center", va="center",
                fontsize=11.5, color="#8a8880")

    # curvas
    for run, (lab, col, ls, lw) in STYLE.items():
        ax.plot(x, data[run], color=col, ls=ls, lw=lw, label=lab, zorder=3,
                solid_capstyle="round")

    # ENFASE: a comutacao de arquitetura (r11)
    ax.axvline(SWITCH, color=RED, lw=2.4, zorder=4, alpha=0.9)
    ax.annotate("architecture switch (r11)", xy=(SWITCH, 66), xytext=(23, 63),
                fontsize=13, fontweight="bold", color="#b5301f",
                arrowprops=dict(arrowstyle="->", color="#b5301f", lw=1.8))

    # contraste (2 anotacoes, sem sobreposicao)
    ax.annotate("FL$\\to$GL: transparent switch\nF1 stays flat as nodes drop 10$\\to$3",
                xy=(50, 95.74), xytext=(42, 73), fontsize=12.5, color="#185fa5",
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#185fa5", lw=1.6))
    ax.annotate("GL$\\to$FL: cold start, then\ndegrades under node loss",
                xy=(85, 53.24), xytext=(66, 66), fontsize=12.5, color="#993c1d",
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#993c1d", lw=1.6))

    ax.set_xlabel("round")
    ax.set_ylabel("F1-score (%)")
    ax.set_ylim(44, 104)
    ax.set_xlim(0.5, 90.5)
    ax.grid(alpha=0.22, zorder=0)
    ax.legend(loc="lower left", ncol=2, framealpha=0.96, edgecolor="#b8b6ad")
    # sem titulo interno -> vai no \caption{} do LaTeX
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print("wrote", OUT + ".png / .pdf")


if __name__ == "__main__":
    main()
