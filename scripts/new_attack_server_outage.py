#!/usr/bin/env python3
"""Dissemination of a new specialist with and without the server: FL-only, GL-only, switching.

What retention cannot do is spread NEW knowledge. This quantifies the role of each mode for
a specialist that becomes ready at one node (node 0, round 11), in the setting of
scripts/new_attack_convergence.py (N = 12, ring, k >= 1), when the server is unavailable for
D rounds from the round in which the specialist is ready.
Policies:
  FL-only    disseminates in the first round in which the server is available, never before;
  GL-only    GLow head-election diffusion on the ring, whatever the state of the server;
  switching  federated while the server is available, gossip while it is not (ReSIDS).
Every node RETAINS what it has received, so detection of the known attacks is unaffected in
all three; only the new attack is at stake.

Panel (a): network recall on the new attack per round for one outage (D = 8).
Panel (b): exposure, in node-rounds without the specialist, for every D from 0 to 16, so
that the result does not depend on the outage chosen for panel (a).

Like new_attack_convergence.py, this is a model of the dissemination with measured recalls:
the two recall levels (without and with the new specialist) are read from
results/new_attack_convergence_1src.csv, which that script measured with real boosters on
the full ERENO test set; the propagation follows the same rules (FL: every node in the
round; GL: the elected head joins if a ring neighbour already holds the specialist).

  python scripts/new_attack_server_outage.py
Out: results/new_attack_server_outage.{csv,txt,pdf,png}
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "results", "new_attack_convergence_1src.csv")
OUT = os.path.join(HERE, "results", "new_attack_server_outage")

N, READY, ROUNDS = 12, 11, 40
D_SHOWN, D_MAX = 8, 16                            # outage of panel (a); sweep of panel (b)
RED, GREEN, BLUE = "#b2182b", "#1b7837", "#2a78d6"
POL = {"fl": ("FL-only", RED, "-", 2.4), "gl": ("GL-only", GREEN, "-", 2.4),
       "switch": ("switching (FL when the server is available, GL otherwise)", BLUE, "--", 3.2)}

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14.5, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5, "legend.fontsize": 13})


def levels():
    rows = list(csv.DictReader(open(SRC)))
    return float(rows[0]["fl_k1"]), float(rows[-1]["fl_k1"])     # without / with the specialist


def gl_step(reached, r):
    """GLow head election (round-robin): the head joins if a ring neighbour holds the item."""
    head = (r - 1) % N
    if head not in reached and ((head - 1) % N in reached or (head + 1) % N in reached):
        reached.add(head)


def run(policy, D):
    """Nodes holding the specialist per round; the server is unavailable in rounds
    READY .. READY + D - 1 (and from round 6, before the specialist exists)."""
    up = lambda r: not (6 <= r <= READY + D - 1) if D > 0 else True
    reached, out = set(), []
    for r in range(1, ROUNDS + 1):
        if r >= READY:
            reached.add(0)                         # the node that trained it
            if policy == "fl" or (policy == "switch" and up(r)):
                if up(r):                          # the server delivers it to every node
                    reached = set(range(N))        # (same round, as in new_attack_convergence.py)
            elif r > READY:                        # gossip: one head election per round
                gl_step(reached, r)
        out.append(len(reached))
    return out


def exposure(cov):
    return sum(N - c for r, c in enumerate(cov, 1) if r >= READY)


def main():
    lo, hi = levels()
    lines = [f"N = {N}, ring; new specialist ready at node 0 in round {READY}; k >= 1",
             f"recall on the new attack: {lo:.2f}% without the specialist, {hi:.2f}% with it",
             "exposure = node-rounds without the specialist, from the round it is ready",
             "", "  D (rounds without server) | FL-only | GL-only | switching | all nodes hold it at round: FL / GL / switching"]
    sweep, rows = {p: [] for p in POL}, []
    for D in range(0, D_MAX + 1):
        cov = {p: run(p, D) for p in POL}
        full = {p: next((r for r, c in enumerate(cov[p], 1) if c == N), None) for p in POL}
        for p in POL:
            sweep[p].append(exposure(cov[p]))
            for r, c in enumerate(cov[p], 1):
                rows.append([D, p, r, c, round((c * hi + (N - c) * lo) / N, 4)])
        lines.append(f"  {D:>26} | {sweep['fl'][-1]:>7} | {sweep['gl'][-1]:>7} | {sweep['switch'][-1]:>9} |"
                     f" {full['fl']} / {full['gl']} / {full['switch']}")
    worse = [D for D in range(D_MAX + 1) if sweep["switch"][D] > min(sweep["fl"][D], sweep["gl"][D])]
    lines.append(f"\nswitching above the better of FL-only and GL-only for D in: {worse or 'none'}")
    txt = "\n".join(lines) + "\n"
    open(OUT + ".txt", "w").write(txt)
    print(txt)
    with open(OUT + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outage_rounds", "policy", "round", "nodes_with_specialist", "network_recall"])
        w.writerows(rows)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    x = list(range(1, 35))
    a.axvspan(5.5, READY + D_SHOWN - 0.5, color="#f1ddd9", alpha=0.75, lw=0, zorder=0)
    a.text((6 + READY + D_SHOWN - 1) / 2, 78, "server\nunavailable", ha="center", va="center",
           fontsize=11.5, color="#9a4a3c")
    a.axvline(READY, color="0.45", ls=":", lw=1.2)
    for p, (lab, col, ls, lw) in POL.items():
        cov = run(p, D_SHOWN)[:34]
        a.plot(x, [(c * hi + (N - c) * lo) / N for c in cov], color=col, ls=ls, lw=lw, label=lab,
               zorder=4 if p == "switch" else 3)
    a.set_xlabel("round")
    a.set_ylabel("recall on the new attack (%)")
    a.set_xlim(1, 34)
    a.set_ylim(-3, 103)
    for p, (lab, col, ls, lw) in POL.items():
        b.plot(range(D_MAX + 1), sweep[p], color=col, ls=ls, lw=lw, zorder=4 if p == "switch" else 3)
    b.axvline(D_SHOWN, color="0.45", ls=":", lw=1.2)
    b.set_xlabel("rounds without the server after the specialist is ready")
    b.set_ylabel("exposure (node-rounds)")
    b.set_xlim(0, D_MAX)
    b.set_ylim(0, None)
    for ax, tag in ((a, "(a)"), (b, "(b)")):
        ax.text(0.02, 0.9, tag, transform=ax.transAxes, fontsize=14, fontweight="bold")
        ax.grid(alpha=0.25)
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print("wrote", OUT + ".{csv,txt,pdf,png}")


if __name__ == "__main__":
    main()
