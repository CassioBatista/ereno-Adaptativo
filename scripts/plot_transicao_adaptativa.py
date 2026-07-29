#!/usr/bin/env python3
"""Enfase na COMUTACAO de arquitetura do IDS adaptativo (FL<->GL em r11).

Le results/conv_adapt_shrink1.csv e gera results/transicao_adaptativa.png:
F1 x round com o momento da comutacao destacado (a alteracao adaptativa),
mais o contraste transparente (FL->GL) vs disruptivo (GL->FL) e o encolhimento.

Uso: python scripts/plot_transicao_adaptativa.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "conv_adapt_shrink1.csv")
OUT = os.path.join(HERE, "results", "transicao_adaptativa.png")

STYLE = {
    "flgl_k1": ("FL->GL  k>=1", "#2a78d6", "-"),
    "glfl_k1": ("GL->FL  k>=1", "#2a78d6", "--"),
    "flgl_k2": ("FL->GL  k>=2", "#eb6834", "-"),
    "glfl_k2": ("GL->FL  k>=2", "#eb6834", "--"),
}
SWITCH = 10.5
SHRINK = [20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5]
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
    fig, ax = plt.subplots(figsize=(13, 6.5))

    # --- fase inicial (from-scratch) sombreada ---
    ax.axvspan(0.5, SWITCH, color="#ecebe6", alpha=0.7, zorder=0)
    ax.text(5.5, 101, "fase inicial\n(from-scratch)",
            ha="center", va="top", fontsize=8, color="#7a786f")

    # --- limites de encolhimento (tênues) + nº de nós (na base, longe da comutação) ---
    for r in SHRINK:
        ax.axvline(r, color="#cfcdc4", lw=0.8, ls=":", zorder=1)
    ax.text(6, 46.3, "nº de nós:", ha="left", va="center",
            fontsize=8, color="#9a988f")
    for xr, n in NODES.items():
        ax.text(xr, 46.3, f"{n}", ha="center", va="center",
                fontsize=8, color="#9a988f")

    # --- curvas ---
    for run, (lab, col, ls) in STYLE.items():
        ax.plot(x, data[run], color=col, ls=ls, lw=2.2, label=lab, zorder=3)

    # --- ENFASE: a comutacao de arquitetura (a alteracao adaptativa) ---
    ax.axvspan(SWITCH - 0.5, SWITCH + 0.5, color="#d84a3a", alpha=0.18, zorder=2)
    ax.axvline(SWITCH, color="#d84a3a", lw=2.6, zorder=4)
    ax.annotate("COMUTAÇÃO DE ARQUITETURA (r11)\nFL ↔ GL — a adaptação",
                xy=(SWITCH, 62), xytext=(21, 60),
                fontsize=11, fontweight="bold", color="#b5301f",
                arrowprops=dict(arrowstyle="->", color="#b5301f", lw=1.6))

    # contraste no ponto da comutacao
    ax.annotate("FL→GL: comutação TRANSPARENTE\n(F1 mantém 95,74)",
                xy=(12, 95.74), xytext=(30, 98.7), fontsize=9, color="#185fa5",
                arrowprops=dict(arrowstyle="->", color="#185fa5", lw=1.2))
    ax.annotate("GL→FL k≥2: sweet-spot 96,38…", xy=(10, 96.38), xytext=(2, 90),
                fontsize=9, color="#993c1d",
                arrowprops=dict(arrowstyle="->", color="#993c1d", lw=1.2))
    ax.annotate("…DESPENCA a 87,63\nao comutar p/ federado", xy=(11.3, 87.63),
                xytext=(15, 74), fontsize=9, color="#993c1d",
                arrowprops=dict(arrowstyle="->", color="#993c1d", lw=1.2))
    ax.annotate("k≥2 + federado + 3 nós\n= crater (53,24)", xy=(85, 53.24),
                xytext=(60, 60), fontsize=9, color="#993c1d",
                arrowprops=dict(arrowstyle="->", color="#993c1d", lw=1.2))

    ax.set_xlabel("round")
    ax.set_ylabel("F1-score (%)")
    ax.set_ylim(44, 103)
    ax.set_xlim(0.5, 90.5)
    ax.grid(alpha=0.22, zorder=0)
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.95)
    ax.set_title("IDS adaptativo: a COMUTAÇÃO de arquitetura (r11) e seu efeito "
                 "sob encolhimento", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
