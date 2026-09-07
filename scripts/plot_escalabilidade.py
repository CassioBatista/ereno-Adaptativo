#!/usr/bin/env python3
"""Figura do teste de ESCALABILIDADE (sweep de peers N=10..100).

Reconstrói a figura a partir de results/escalabilidade.csv. Três séries:
  * Recall (OR = k>=2, GL = FL)   -> OR_rec_GL  (invariante ~100%)
  * OR F1 (GL = FL)               -> OR_F1_GL
  * k>=2 F1                       -> k2_F1_GL   (pool difundido do GL)
Mensagem: recall invariante; F1 só erode em over-partitioning (N=100), onde os
especialistas ficam com pouco dado; k>=2 sobre o pool difundido recupera parte.

NOTA: versão SEM a faixa "realistic substation range" (removida a pedido — o
corte era um sombreado interpretativo sem justificativa no código).

Saída: results/escalabilidade.{png,pdf}
Rodar: ~/venv-ereno314/bin/python scripts/plot_escalabilidade.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/escalabilidade.csv"
OUT = "results/escalabilidade"


def main():
    if not os.path.exists(CSV):
        sys.exit(f"[plot] ausente: {CSV}")
    N, recall, or_f1, k2_f1 = [], [], [], []
    with open(CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            N.append(int(row["N"]))
            recall.append(float(row["OR_rec_GL"]))
            or_f1.append(float(row["OR_F1_GL"]))
            k2_f1.append(float(row["k2_F1_GL"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.plot(N, recall, "-o", color="#2ca02c", lw=1.8, ms=5,
            label="Recall (OR = $k\\geq2$, GL = FL)")
    ax.plot(N, or_f1, "-s", color="#1f77b4", lw=1.8, ms=5,
            label="OR F1 (GL = FL)")
    ax.plot(N, k2_f1, "--^", color="#e08214", lw=1.8, ms=6,
            label="$k\\geq2$ F1")

    ax.set_xlabel("Number of peers (N)")
    ax.set_ylabel("Detection metric (%)")
    ax.set_title("Recall invariant; F1 erodes only under over-partitioning")
    ax.set_ylim(90, 100.6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    # anotação da erosão em N=100 (mantida)
    ax.annotate(
        "OR F1 erosion at N=100\n(weak-data specialists;\n"
        "higher $k$ on the seeded pool recovers it)",
        xy=(100, or_f1[-1]), xytext=(58, 92.4),
        fontsize=8, color="#b2182b", style="italic",
        arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1),
    )

    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=150)
    fig.savefig(OUT + ".pdf")
    print("[plot] ->", OUT + ".png / .pdf")


if __name__ == "__main__":
    main()
