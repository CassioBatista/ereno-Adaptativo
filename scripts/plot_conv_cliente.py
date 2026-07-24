"""Plota a convergencia por-cliente (le results/conv_por_cliente.csv).

GL: 10 curvas por-no que sobem (difusao) + media. FL: 10 linhas planas
(especialistas parciais) + agregado (uniao OR). Mostra que o gossip da a CADA
no o que o FL so da ao servidor.
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/conv_por_cliente.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "results/conv_por_cliente.png"

series = defaultdict(lambda: ([], []))
with open(src) as fh:
    for row in csv.DictReader(fh):
        r, v = int(row["round"]), float(row["recall"])
        series[(row["serie"], row["tipo"])][0].append(r)
        series[(row["serie"], row["tipo"])][1].append(v)

fig, ax = plt.subplots(figsize=(9, 5.2))
for (name, tipo), (xs, ys) in series.items():
    if tipo == "gl_node":
        ax.plot(xs, ys, color="#85b7eb", lw=1, alpha=0.8, zorder=2)
    elif tipo == "fl_node":
        ax.plot(xs, ys, color="#f0997b", lw=1, alpha=0.7, ls="--", zorder=1)
ax.plot(*series[("GL_agg", "gl_agg")], color="#185fa5", lw=2.8,
        label="GL — média dos nós (difusão)", zorder=4)
ax.plot(*series[("FL_agg", "fl_agg")], color="#0f6e56", lw=2.8,
        label="FL — agregado (união OR)", zorder=4)
ax.plot([], [], color="#85b7eb", lw=1, label="GL — cada nó (10)")
ax.plot([], [], color="#f0997b", lw=1, ls="--", label="FL — cada especialista (10)")

ax.set_xlabel("round de gossip"); ax.set_ylabel("Recall no teste global (%)")
ax.set_title("Convergência por-nó: GL difunde até cada nó ≈ rede inteira; "
             "FL deixa cada especialista parcial")
ax.set_ylim(0, 102); ax.grid(True, color="#e1e0d9", lw=0.6)
ax.legend(fontsize=8.5, loc="center right")
fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"[plot] salvo {out}")
