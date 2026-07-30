#!/usr/bin/env python3
"""Resiliencia do FL k>=2 com especialistas redundantes -> GL k>=2.

Compara, sob encolhimento da rede (14->3 nos), tres configuracoes do FL->GL
(todas planas pela transparencia da semeadura FL->GL):
  - k>=2 com 14 nos (2 sensores/ataque)  -> 95.98 (recall ~100), o melhor
  - k>=1 (OR)                            -> 95.74 (referencia §7)
  - k>=2 sem redundancia (1 sensor/ataque) -> 87.63 (recall 78%, §7)

Gera results/resiliencia_k2_redund.png.
Uso: python scripts/plot_resiliencia_k2_redund.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "resiliencia_k2_redund.png")

# Niveis planos (o FL->GL semeia a uniao em todo no => transparente ao churn).
# k>=2 redundante medido 14->3 (teste completo 95.98); OR e k>=2-sem-redund
# do §7 (planos, deploys de 10 nos) como referencia.
NODES = [14, 12, 10, 8, 7, 6, 5, 3]
LEVELS = {
    "k>=2, 14 nos (2/ataque)":        (95.98, "#1d9e75", "-",  3),
    "k>=1 (OR)":                      (95.74, "#2a78d6", "-",  2),
    "k>=2, sem redundancia (1/ataque)": (87.63, "#eb6834", "--", 2),
}


def main():
    fig, ax = plt.subplots(figsize=(9, 5))
    for lab, (val, col, ls, lw) in LEVELS.items():
        ax.plot(NODES, [val] * len(NODES), color=col, ls=ls, lw=lw,
                marker="o", markersize=5, label=lab)
    ax.set_xlabel("nº de nós (encolhendo →)")
    ax.set_ylabel("F1-score (%)")
    ax.set_ylim(80, 100)
    ax.invert_xaxis()   # 14 -> 3 da esquerda para a direita
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Resiliência a churn: FL k≥2 com especialistas redundantes → GL k≥2",
                 fontsize=11)
    ax.annotate("recall ~100%", xy=(7, 95.98), xytext=(9, 92.5), fontsize=8,
                color="#0f6e56", arrowprops=dict(arrowstyle="->", color="#0f6e56"))
    ax.annotate("recall 78%", xy=(7, 87.63), xytext=(9, 84), fontsize=8,
                color="#993c1d", arrowprops=dict(arrowstyle="->", color="#993c1d"))
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
