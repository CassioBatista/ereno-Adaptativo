#!/usr/bin/env python3
"""Slide-ready headline: adaptive (FL->GL, retained union) holds detection under
node loss while static FL collapses. Source: results/redund_k2_fulltest.csv (k>=2,
N=14 down to 3, two specialists/attack). Out: results/headline_shrink.{png,pdf}."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, fl_f1, gl_f1, fl_rec, gl_rec = [], [], [], [], []
with open("results/redund_k2_fulltest.csv") as fh:
    for r in csv.DictReader(fh):
        if int(r["k"]) == 2:
            N.append(int(r["N"]))
            fl_f1.append(float(r["FL_F1"]));  gl_f1.append(float(r["GL_F1"]))
            fl_rec.append(float(r["FL_Recall"])); gl_rec.append(float(r["GL_Recall"]))

# ordena por N decrescente (14 -> 3, como a rede encolhe)
order = sorted(range(len(N)), key=lambda i: -N[i])
N      = [N[i] for i in order]
fl_f1  = [fl_f1[i] for i in order];  gl_f1 = [gl_f1[i] for i in order]

plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.plot(N, gl_f1, "-o", color="#1b7837", lw=2.6, ms=8,
        label="Adaptive (FL$\\rightarrow$GL, retained union)")
ax.plot(N, fl_f1, "--s", color="#b2182b", lw=2.4, ms=8,
        label="Static FL (no switch)")

ax.annotate(f"{gl_f1[-1]:.1f}", (N[-1], gl_f1[-1]), textcoords="offset points",
            xytext=(6, 8), color="#1b7837", fontweight="bold")
ax.annotate(f"{fl_f1[-1]:.1f}", (N[-1], fl_f1[-1]), textcoords="offset points",
            xytext=(6, -18), color="#b2182b", fontweight="bold")

ax.set_xlabel("Number of active nodes (network shrinks $\\rightarrow$)")
ax.set_ylabel("F1-score (%)")
ax.set_title("Under node loss: adaptive stays flat; static FL collapses\n"
             "(ERENO, $N{=}14$, 2 specialists/attack, $k\\geq2$)", fontsize=14)
ax.set_xticks(N)
ax.invert_xaxis()                      # 14 na esquerda, 3 na direita
ax.set_ylim(48, 100)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", fontsize=13)
fig.tight_layout()
fig.savefig("results/headline_shrink.png", dpi=180)
fig.savefig("results/headline_shrink.pdf")
print("delta @N=3:  adaptive %.2f  vs  static %.2f  (gap %.1f pts)"
      % (gl_f1[-1], fl_f1[-1], gl_f1[-1]-fl_f1[-1]))
print("ok")
