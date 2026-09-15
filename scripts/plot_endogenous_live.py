#!/usr/bin/env python3
"""Endogenous latency-aware curve from a LIVE decentralized run (B1 Gaps 1+2).

Unlike the B2 composition (scripts/plot_latency_aware.py, which glued measured F1
levels to a modeled latency), this parses the per-round F1 emitted by an actual
Flower run of the DECENTRALIZED manager (ROUND;<r>;<mode>;f1=..;recall=..;fpr=..)
so every FL<->GL switch round is where the live decentralized detection actually
fired. Handles both the fail-only run and the fail+recovery run (GL->FL).

Reads results/dist_adapt_n14_k2_decentralized.log (override via argv[1]).
Out: results/<stem>_end2end.{png,pdf} + .csv  (stem from the log name)
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
COMMIT_RE = re.compile(r"\[DistArch\] COMMIT (\w+) @round=(\d+)")
ACTIVE_RE = re.compile(r"\[DistArch\] round=(\d+).*active=\[")   # partial participation

rows, commits, partial_rounds = [], [], []
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
        a = ACTIVE_RE.search(line)
        if a:
            partial_rounds.append(int(a.group(1)))

if not rows:
    sys.exit(f"No ROUND; lines in {LOG} — is the run finished? (rows=0)")

rows.sort(key=lambda r: r["round"])
R = [r["round"] for r in rows]
F1 = [r["f1"] for r in rows]

# fault round = first round whose F1 drops below the healthy baseline
base = F1[0]
fail_round = next((r["round"] for r in rows if r["f1"] < base - 1e-6), None)
# node-return round (recovery runs): first full-participation round after a partial spell
node_return = None
if partial_rounds:
    last_partial = max(partial_rounds)
    node_return = last_partial + 1 if last_partial < max(R) else None
# mode transitions from the per-round labels
trans = [(rows[i]["round"], rows[i - 1]["mode"], rows[i]["mode"])
         for i in range(1, len(rows)) if rows[i]["mode"] != rows[i - 1]["mode"]]
sw_fl_gl = next((r for r, a, b in trans if a == "federated" and b == "gossip"), None)
sw_gl_fl = next((r for r, a, b in trans if a == "gossip" and b == "federated"), None)

stem = os.path.splitext(os.path.basename(LOG))[0].replace("dist_adapt_n14_k2_", "")
out = f"results/{stem}_end2end" if stem else "results/endogenous_live_end2end"
if "decentralized" in LOG and "recovery" in LOG:
    out = "results/endogenous_live_recovery_end2end"
elif "decentralized" in LOG:
    out = "results/endogenous_live_end2end"
os.makedirs("results", exist_ok=True)
with open(out + ".csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["round", "mode", "f1", "recall", "fpr"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

fig, ax = plt.subplots(figsize=(9.6, 5.3))
fl_pts = [(r["round"], r["f1"]) for r in rows if r["mode"] == "federated"]
gl_pts = [(r["round"], r["f1"]) for r in rows if r["mode"] == "gossip"]
ax.plot(R, F1, "-", color="0.6", lw=1.2, zorder=1)
if fl_pts:
    ax.plot(*zip(*fl_pts), "s", color="#b2182b", ms=6, label="federated (FL) round")
if gl_pts:
    ax.plot(*zip(*gl_pts), "o", color="#1b7837", ms=6, label="gossip (GL) round")

lo, hi = min(F1), max(F1)
# degraded window: fault -> FL->GL switch (FL runs degraded until the switch)
if fail_round is not None and sw_fl_gl is not None:
    ax.axvspan(fail_round, sw_fl_gl, color="#f0ad4e", alpha=0.18)
    ax.axvline(fail_round, color="0.4", ls=":", lw=1)
    ax.annotate("node loss", (fail_round + 0.2, hi - 0.1), fontsize=8.5,
                ha="left", va="top", color="0.35")
    ax.axvline(sw_fl_gl, color="#2166ac", ls="-", lw=1.4, alpha=0.85)
    ax.annotate(f"FL$\\rightarrow$GL @r{sw_fl_gl}\n(detect window {sw_fl_gl - fail_round})",
                (sw_fl_gl + 0.3, lo + (hi - lo) * 0.10), fontsize=8.5, color="#2166ac")
# recovery: node returns -> GL->FL switch (dwell + federated-vote diffusion)
if node_return is not None:
    ax.axvline(node_return, color="0.4", ls=":", lw=1)
    ax.annotate("node returns", (node_return + 0.2, hi - 0.1), fontsize=8.5,
                ha="left", va="top", color="0.35")
if sw_gl_fl is not None:
    ax.axvspan((node_return or sw_gl_fl), sw_gl_fl, color="#7fbf7b", alpha=0.14)
    ax.axvline(sw_gl_fl, color="#1b7837", ls="-", lw=1.4, alpha=0.85)
    ax.annotate(f"GL$\\rightarrow$FL recovery @r{sw_gl_fl}\n(transparent: F1 stays "
                f"{F1[-1]:.1f})", (sw_gl_fl + 0.3, lo + (hi - lo) * 0.45),
                fontsize=8.5, color="#1b7837")

ax.set_xlabel("round"); ax.set_ylabel("F1-score (%)")
ttl = ("Endogenous latency-aware end-to-end (LIVE decentralized run):\n"
       "every FL$\\leftrightarrow$GL switch fired from decentralized detection "
       "($N{=}14$, $k\\geq2$, ring)")
ax.set_title(ttl, fontsize=11.5)
ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9.5)
fig.tight_layout()
fig.savefig(out + ".png", dpi=175)
fig.savefig(out + ".pdf")
print(f"rounds={len(rows)}  fault@{fail_round}  FL->GL@{sw_fl_gl}  "
      f"node_return@{node_return}  GL->FL@{sw_gl_fl}")
print(f"commits: {commits}")
print(f"ok -> {out}.{{png,pdf,csv}}")
