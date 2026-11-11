#!/usr/bin/env python3
"""ReSIDS architecture (Paper 2, Fig. resids_architecture).

Left: one node -- Disaster-FD monitor -> trust level -> agent -> learning mode
(FL / GL / local) -> held specialists M_i -> OR fusion -> alert.
Right: the system -- aggregator, ring overlay, FL and GL exchanges, and the probes
of one monitor (aggregator 50, neighbours 15 + 15).
Differs from Paper 1 Fig. 1 (global FL<->GL switch, k-of-n): control is per node,
driven by the trust level, with a third mode (local).
Out: results/figs/resids_architecture.{pdf,png}
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

GREY = dict(fc="#ececec", ec="#7a7a7a")
BLUE = dict(fc="#d6e4f5", ec="#3b5b92")
GREEN = dict(fc="#d9efdd", ec="#3a7d4f")
ORANGE = dict(fc="#f8e6cf", ec="#b0651c")
LAV = dict(fc="#eadcf5", ec="#6a3d9a")
C_FL, C_GL, C_PR, C_AL = "#3b5b92", "#3a7d4f", "#6a3d9a", "#c0392b"

fig, ax = plt.subplots(figsize=(10.4, 5.75))
ax.set_xlim(0, 12.4); ax.set_ylim(0, 6.85); ax.set_aspect("equal"); ax.axis("off")


def box(x, y, w, h, st, title, sub=None, tfs=11, sfs=8.6, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=st["fc"], ec=st["ec"], lw=lw))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h - 0.27, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=st["ec"])
        ax.text(cx, y + (h - 0.45) / 2, sub, ha="center", va="center", fontsize=sfs,
                color="0.2", linespacing=1.35)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=st["ec"])


def arrow(p0, p1, color="0.35", ls="-", lw=1.6, rad=0.0, ms=13):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, ls=ls, shrinkA=1, shrinkB=1,
                                 connectionstyle=f"arc3,rad={rad}"))


# ── left: one node ──────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.1, 0.1), 6.45, 6.5, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="none", ec="0.6", lw=1.2, ls=(0, (5, 3))))
ax.text(0.3, 6.38, "Node $i$", ha="left", va="center", fontsize=12, fontweight="bold",
        color="0.2")
ax.text(1.42, 6.38, "(specialist of attack type $i$)", ha="left", va="center", fontsize=9,
        color="0.4")

box(0.3, 4.75, 2.55, 1.3, LAV, "Disaster-FD monitor $i$",
    "probes, with weights:\nthe FL aggregator ($w_A$),\neach neighbour specialist ($w_j$)",
    tfs=10.5, sfs=8.2)
box(3.75, 4.75, 2.6, 1.3, ORANGE, "Agent $i$",
    "FL if TL $\\geq \\theta_{\\mathrm{FL}}$;  GL if TL $\\geq \\theta_{\\mathrm{GL}}$;\n"
    "local otherwise\nsettle $S$;  no TL for $F$ $\\to$ local",
    sfs=8.2)
arrow((2.87, 5.4), (3.73, 5.4), color=C_PR, lw=1.8)
ax.text(3.3, 5.62, "TL", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_PR)
ax.text(3.3, 5.13, "loopback\nHTTP/CoAP", ha="center", va="center", fontsize=6.8,
        color="0.3", linespacing=1.1)

# mode selector
ax.add_patch(FancyBboxPatch((0.3, 3.05), 6.05, 1.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc="#f6f6f6", ec="0.7", lw=1.1, ls=(0, (4, 3))))
ax.text(0.45, 4.07, "learning mode (one at a time)", ha="left", va="center", fontsize=8.2,
        color="0.35", style="italic")
box(0.5, 3.17, 1.75, 0.75, BLUE, "FL", "aggregator + specialists", tfs=10.5, sfs=7.6)
box(2.45, 3.17, 1.75, 0.75, GREEN, "GL", "specialists only", tfs=10.5, sfs=7.6)
box(4.4, 3.17, 1.75, 0.75, GREY, "Local", "no exchange", tfs=10.5, sfs=7.8)
arrow((5.05, 4.73), (5.05, 4.27))
ax.text(5.15, 4.5, "selects", ha="left", va="center", fontsize=7.8, color="0.35")

# specialists
box(0.3, 1.55, 1.95, 1.0, GREY, "$b_i$", "own specialist,\nlocal traffic", tfs=11, sfs=7.8)
box(2.75, 1.55, 3.6, 1.0, GREY, "Held specialists $\\mathcal{M}_i$",
    "union deduplicated by content,\nat most one per node", tfs=10, sfs=7.8)
arrow((2.27, 2.05), (2.73, 2.05))
arrow((3.33, 3.03), (3.9, 2.57))
ax.text(3.75, 2.83, "specialists of\nother nodes", ha="left", va="center", fontsize=7.2,
        color="0.35", linespacing=1.1)
arrow((1.3, 2.57), (1.38, 3.03))
ax.text(1.45, 2.8, "$b_i$ shared", ha="left", va="center", fontsize=7.2, color="0.35")

# detection
box(1.55, 0.3, 3.4, 0.9, ORANGE, "OR fusion",
    "alert if $b(x) \\geq 0.5$ for some $b \\in \\mathcal{M}_i$", tfs=10.5, sfs=7.8)
ax.text(0.95, 0.75, "message $x$", ha="center", va="center", fontsize=8.5, color="0.3")
arrow((1.0, 0.55), (1.53, 0.6), color="0.35")
arrow((4.55, 1.53), (4.1, 1.22))
arrow((4.97, 0.75), (5.55, 0.75), color=C_AL, lw=2.0)
ax.text(5.95, 0.75, "ALERT", ha="center", va="center", fontsize=10, fontweight="bold",
        color=C_AL)

# ── right: the system, one view per sharing mode ───────────────────────────────
ax.text(6.95, 6.38, "System", ha="left", va="center", fontsize=12, fontweight="bold",
        color="0.2")
n, R, rn, cx = 8, 0.95, 0.21, 10.15
ang = [math.pi / 2 + 2 * math.pi * k / n for k in range(n)]      # node i at the top


def nodes(cy, labels, nbrs=()):
    pos = [(cx + R * math.cos(a), cy + R * math.sin(a)) for a in ang]
    for k, (x, y) in enumerate(pos):
        st = ORANGE if k == 0 else GREEN if k in nbrs else GREY
        ax.add_patch(Circle((x, y), rn, fc=st["fc"], ec=st["ec"], lw=1.4, zorder=3))
        ax.text(x, y, labels.get(k, ""), ha="center", va="center", fontsize=7.4, zorder=4,
                color="0.15")
        # its own Disaster-FD monitor + agent (badge, outward)
        bx, by = cx + (R + 0.3) * math.cos(ang[k]), cy + (R + 0.3) * math.sin(ang[k])
        ax.add_patch(Circle((bx, by), 0.1, fc=LAV["fc"], ec=LAV["ec"], lw=1.2, zorder=5))
    return pos


def side_label(y, title, sub, col):
    ax.text(6.95, y + 0.2, title, ha="left", va="center", fontsize=11, fontweight="bold",
            color=col)
    ax.text(6.95, y - 0.2, sub, ha="left", va="center", fontsize=8, color="0.3",
            linespacing=1.25)


# FL: aggregator + specialists (star)
cy_fl = 5.0
pos = nodes(cy_fl, {0: "$i$"})
for x, y in pos:
    ax.add_line(Line2D([cx, x], [cy_fl, y], color=C_FL, lw=1.4, zorder=1))
box(cx - 0.55, cy_fl - 0.22, 1.1, 0.44, BLUE, "Aggregator", tfs=8.2, lw=1.4)
side_label(cy_fl, "FL", "aggregator\n+ specialists", C_FL)

# GL: specialists only, on a generic connected overlay graph (ring in the evaluation)
cy_gl = 1.85
nbrs_i = (1, 3, 6)                                   # N(i): neighbours of node i (k = 0)
edges = [(0, 1), (0, 3), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1),
         (2, 5), (4, 7)]
pos = nodes(cy_gl, {0: "$i$"}, nbrs=nbrs_i)
for a, b in edges:
    (x0, y0), (x1, y1) = pos[a], pos[b]
    ax.add_line(Line2D([x0, x1], [y0, y1], color=C_GL, lw=2.4 if 0 in (a, b) else 1.4,
                       alpha=1.0 if 0 in (a, b) else 0.55, zorder=2))
ax.text(6.95, cy_gl + 0.75, "GL", ha="left", va="center", fontsize=11, fontweight="bold",
        color=C_GL)
ax.text(6.95, cy_gl + 0.4, "specialists only,\noverlay graph\n(a ring in the\nevaluation)",
        ha="left", va="top", fontsize=8, color="0.3", linespacing=1.25)
ax.add_patch(Circle((7.05, cy_gl - 0.95), 0.1, fc=GREEN["fc"], ec=GREEN["ec"], lw=1.3))
ax.text(7.22, cy_gl - 0.95, "$\\mathcal{N}(i)$, neighbours of $i$", ha="left", va="center",
        fontsize=7.6, color="0.25")

ax.add_line(Line2D([6.95, 12.25], [3.45, 3.45], color="0.8", lw=0.8, ls=(0, (4, 3))))
ax.add_patch(Circle((7.08, 0.2), 0.1, fc=LAV["fc"], ec=LAV["ec"], lw=1.2))
ax.text(7.25, 0.2, "Disaster-FD monitor + agent, one per specialist (no monitor at the "
        "aggregator)", ha="left", va="center", fontsize=7.4, color="0.25")
fig.tight_layout(pad=0.2)
os.makedirs("results/figs", exist_ok=True)
fig.savefig("results/figs/resids_architecture.pdf")
fig.savefig("results/figs/resids_architecture.png", dpi=170)
print("ok -> results/figs/resids_architecture.{pdf,png}")
