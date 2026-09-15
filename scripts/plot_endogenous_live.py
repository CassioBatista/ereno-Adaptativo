#!/usr/bin/env python3
"""Endogenous latency-aware curve from a LIVE decentralized run (B1 Gaps 1+2).

Unlike the B2 composition (scripts/plot_latency_aware.py, which glued measured F1
levels to a modeled latency), this parses the per-round F1 emitted by an actual
Flower run of the DECENTRALIZED manager (ROUND;<r>;<mode>;f1=..;recall=..;fpr=..)
so the FL->GL switch round is where the live decentralized detection actually fired.

Reads results/dist_adapt_n14_k2_decentralized.log (override via argv[1]).
Out: results/endogenous_live_end2end.{png,pdf} + .csv
"""
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else "results/dist_adapt_n14_k2_decentralized.log"
ROUND_RE = re.compile(r"^ROUND;(\d+);(\w+);f1=([\d.]+);recall=([\d.]+);fpr=([\d.]+)")
FAULT_RE = re.compile(r"\[DistArch\] round=(\d+).*active=\[")     # rounds w/ a fault
COMMIT_RE = re.compile(r"\[DistArch\] COMMIT (\w+) @round=(\d+)")

rows = []
faults, commits = [], []
with open(LOG) as fh:
    for line in fh:
        m = ROUND_RE.match(line)
        if m:
            rows.append({"round": int(m.group(1)), "mode": m.group(2),
                         "f1": float(m.group(3)), "recall": float(m.group(4)),
                         "fpr": float(m.group(5))})
        c = COMMIT_RE.search(line)
        if c:
            commits.append((c.group(1), int(c.group(2))))

if not rows:
    sys.exit(f"No ROUND; lines in {LOG} — is the run finished? (rows=0)")

rows.sort(key=lambda r: r["round"])
R = [r["round"] for r in rows]
F1 = [r["f1"] for r in rows]
modes = [r["mode"] for r in rows]

# endogenous switch = first round whose mode is gossip
switch = next((r["round"] for r in rows if r["mode"] == "gossip"), None)
# fault rounds from the config are 10/16/22 (first drop = 10); detect from log commits
fail_round = 10

os.makedirs("results", exist_ok=True)
with open("results/endogenous_live_end2end.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["round", "mode", "f1", "recall", "fpr"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

fig, ax = plt.subplots(figsize=(9.2, 5.2))
# split the curve by mode so FL (degraded) and GL (recovered) read distinctly
fl_pts = [(r["round"], r["f1"]) for r in rows if r["mode"] == "federated"]
gl_pts = [(r["round"], r["f1"]) for r in rows if r["mode"] == "gossip"]
ax.plot(R, F1, "-", color="0.6", lw=1.2, zorder=1)
if fl_pts:
    ax.plot(*zip(*fl_pts), "s", color="#b2182b", ms=6, label="federated (FL) round")
if gl_pts:
    ax.plot(*zip(*gl_pts), "o", color="#1b7837", ms=6, label="gossip (GL) round")

if switch is not None:
    ax.axvspan(fail_round, switch, color="#f0ad4e", alpha=0.18)
    ax.axvline(fail_round, color="0.4", ls=":", lw=1)
    ax.axvline(switch, color="#2166ac", ls="-", lw=1.4, alpha=0.85)
    ax.annotate(f"endogenous switch\nFL$\\rightarrow$GL @round {switch}",
                (switch, min(F1) + 0.6), xytext=(switch + 0.4, min(F1) + 0.6),
                fontsize=9, color="#2166ac", va="center")
    ax.annotate(f"decentralized detection window = {switch - fail_round} rounds\n"
                f"(node loss @{fail_round} $\\to$ live commit @{switch})",
                ((fail_round + switch) / 2, max(F1) - (max(F1) - min(F1)) * 0.35),
                ha="center", fontsize=8.5, color="#8a5a00")
    ax.annotate("node loss (14$\\to$13)", (fail_round + 0.2, max(F1) - 0.15),
                fontsize=8.5, ha="left", va="top", color="0.35")

ax.set_xlabel("round"); ax.set_ylabel("F1-score (%)")
ax.set_title("Endogenous latency-aware end-to-end (LIVE decentralized run):\n"
             "the FL$\\rightarrow$GL switch round is where the decentralized "
             "detection actually fired ($N{=}14$, $k\\geq2$, ring)", fontsize=11.5)
ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=9.5)
fig.tight_layout()
fig.savefig("results/endogenous_live_end2end.png", dpi=175)
fig.savefig("results/endogenous_live_end2end.pdf")
print(f"rounds parsed: {len(rows)}  modes: FL={modes.count('federated')} GL={modes.count('gossip')}")
print(f"endogenous switch @round {switch}  (fault @{fail_round} -> window {None if switch is None else switch-fail_round})")
print(f"commits: {commits}")
print("ok -> results/endogenous_live_end2end.{png,pdf,csv}")
