#!/usr/bin/env python3
"""Mini flowchart of the GL->FL recovery ("recover carefully") decision, per node
per round: full participation restored -> stability >= dwell -> cooldown elapsed ->
admission test (agg == TrustedUnion, v3) -> quorum -> COMMIT GL->FL; any 'no'
keeps the node in GL until the next round. Out: results/gl_fl_flow.{png,pdf}."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

fig, ax = plt.subplots(figsize=(9.5, 11))
ax.set_xlim(0, 112); ax.set_ylim(0, 100); ax.axis("off")
SX = 40  # spine x

def box(cx, cy, w, h, text, fc="white", ec="black", fs=10.5, lw=1.4):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.4",
                       fc=fc, ec=ec, lw=lw, mutation_scale=8, zorder=3)
    p._cx, p._cy, p._w, p._h = cx, cy, w, h
    ax.add_patch(p); ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)
    return p

def diamond(cx, cy, w, h, text, fc="#eef3fb", fs=9.5):
    pts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    p = Polygon(pts, closed=True, fc=fc, ec="black", lw=1.4, zorder=3)
    p._cx, p._cy, p._w, p._h = cx, cy, w, h
    ax.add_patch(p); ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)
    return p

def down(a, b, label=None):
    ax.add_patch(FancyArrowPatch((a._cx, a._cy-a._h/2), (b._cx, b._cy+b._h/2),
                 arrowstyle="-|>", mutation_scale=14, lw=1.4, zorder=2))
    if label: ax.text(a._cx+3, (a._cy-a._h/2+b._cy+b._h/2)/2, label, fontsize=8.5,
                      ha="left", va="center", color="#1b7837", zorder=5)

def toright(a, rail_x, label="no"):
    ax.add_patch(FancyArrowPatch((a._cx+a._w/2, a._cy), (rail_x, a._cy),
                 arrowstyle="-|>", mutation_scale=12, lw=1.2, ls="--", color="#b2182b", zorder=2))
    ax.text(a._cx+a._w/2+3, a._cy+1.6, label, fontsize=8, color="#b2182b", ha="left", zorder=5)

# --- right rail: remain in GL ---
railx = 84
ax.add_patch(FancyBboxPatch((railx, 12), 24, 66, boxstyle="round,pad=0.4",
             fc="#fbeeee", ec="#b2182b", lw=1.4, ls="--", zorder=1))
ax.text(railx+12, 45, "remain in GL\n(re-evaluate\nnext round)", ha="center", va="center",
        fontsize=10.5, color="#7d1a1a", zorder=4)

# --- spine ---
start = box(SX, 92, 40, 7, "In GL mode  (each round, per node)", fc="#eafaf0")
d1 = diamond(SX, 80, 34, 12, "full participation\nrestored?\n(all $N$ back, no new fail)")
inc = box(SX, 67, 26, 7, "stability counter ++")
d2 = diamond(SX, 56, 30, 11, "stable $\\geq$ dwell?")
d3 = diamond(SX, 44, 30, 11, "cooldown elapsed?")
d4 = diamond(SX, 31, 36, 13, "admission test OK?\n(agg == TrustedUnion)\n[v3]")
d5 = diamond(SX, 18, 34, 11, "quorum?\n($\\geq q$ vote FL)")
commit = box(SX, 7, 34, 7, "COMMIT  GL$\\rightarrow$FL  (recovery)", fc="#cdebd6", fs=10.5, lw=1.6)

down(start, d1)
down(d1, inc, "yes"); toright(d1, railx, "no: reset stability")
down(inc, d2)
down(d2, d3, "yes"); toright(d2, railx)
down(d3, d4, "yes"); toright(d3, railx)
down(d4, d5, "yes"); toright(d4, railx, "no: reject server")
down(d5, commit, "yes"); toright(d5, railx)

# loop back: rail -> start (next round)
ax.add_patch(FancyArrowPatch((railx+12, 78), (SX+20, 92), arrowstyle="-|>",
             mutation_scale=13, lw=1.2, ls="--", color="0.5",
             connectionstyle="arc3,rad=0.3", zorder=2))
ax.text(80, 87, "next round", fontsize=8.5, color="0.4", ha="right", zorder=5)

ax.text(SX, 99, "GL $\\rightarrow$ FL recovery (“recover carefully”)",
        ha="center", va="top", fontsize=13, fontweight="bold")

fig.tight_layout()
fig.savefig("results/gl_fl_flow.pdf")
fig.savefig("results/gl_fl_flow.png", dpi=175)
print("ok")
