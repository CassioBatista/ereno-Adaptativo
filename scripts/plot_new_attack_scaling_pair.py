#!/usr/bin/env python3
"""Side-by-side version of the two new-attack scaling figures, for a single LaTeX figure
at full text width: (a) detection latency vs N, (b) network recall ramps per N.
Same computation as scripts/new_attack_scaling.py (GLow head-election diffusion of one
expert on a ring; recall levels measured in the N=12 run); fonts sized for half-width
panels, no in-plot titles -- the sub-captions carry them.

  python scripts/plot_new_attack_scaling_pair.py
Out: results/new_attack_scaling_pair.{pdf,png}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.topology import build_topology  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NLIST = [10, 20, 50, 100]
CEILING, BASELINE, SRC, MAXR = 99.64, 3.81, 0, 400
COLORS = ["#1b7837", "#2166ac", "#e08214", "#b2182b"]


def gl_diffusion(n):
    topo = build_topology("ring", n)
    reached, counts, r = {SRC}, [1], 1
    while len(reached) < n and r <= MAXR:
        head = (r - 1) % n
        if head not in reached and any(nb in reached for nb in topo.neighbors(head)):
            reached.add(head)
        counts.append(len(reached))
        r += 1
    return counts


def main():
    lat, lat90, ramps = {}, {}, {}
    for n in NLIST:
        c = gl_diffusion(n)
        ramps[n] = [BASELINE + (CEILING - BASELINE) * (x / n) for x in c]
        lat[n] = next(r for r, x in enumerate(c) if x >= n)
        lat90[n] = next(r for r, x in enumerate(c) if x >= 0.9 * n)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    # (a) latency vs N
    a.plot(NLIST, [lat[n] for n in NLIST], "-o", color="#1b7837", lw=2.0, ms=6,
           label="GL: whole network")
    a.plot(NLIST, [lat90[n] for n in NLIST], "--^", color="#74c476", lw=1.7, ms=5,
           label="GL: 90% of nodes")
    a.plot(NLIST, [1] * len(NLIST), "-s", color="#b2182b", lw=2.0, ms=6,
           label="FL (~1 round)")
    a.plot(NLIST, NLIST, ":", color="0.55", lw=1.1, label="$y=N$")
    a.set_xlabel("number of peers $N$", fontsize=10)
    a.set_ylabel("new-attack detection latency (rounds)", fontsize=10)
    a.legend(loc="upper left", fontsize=8.5)
    # (b) recall ramps
    for n, col in zip(NLIST, COLORS):
        b.plot(range(len(ramps[n])), ramps[n], "-", color=col, lw=1.9, label=f"GL, $N={n}$")
    b.plot([0, 1], [BASELINE, CEILING], "-o", color="0.15", lw=2.2, ms=4, label="FL, any $N$")
    b.axhline(CEILING, color="0.6", ls=":", lw=1)
    b.set_xlabel("rounds since the new expert is ready (at one node)", fontsize=10)
    b.set_ylabel("network recall on the new attack (%)", fontsize=10)
    b.set_xlim(0, max(lat.values()) * 1.02)
    b.set_ylim(-3, 103)
    b.legend(loc="lower right", fontsize=8.5)
    for ax, tag in ((a, "(a)"), (b, "(b)")):
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.text(0.01, 1.02, tag, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"results/new_attack_scaling_pair.{ext}", dpi=300, bbox_inches="tight")
    print({n: (lat[n], lat90[n]) for n in NLIST})


if __name__ == "__main__":
    main()
