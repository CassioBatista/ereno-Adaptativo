#!/usr/bin/env python3
"""False-positive baseline from stream A (benign-only traffic).

Stream A contains no attacks, so EVERY intrusion_detected in it is a false alarm. This
quantifies what the monitor must filter: how often a benign window fires, with what
volume (n_flags) and what corroboration (k_votes), and what an escalation threshold
would cost and save. Compares against stream B (windows that do contain attacks).

Out: results/fp_baseline_stream_a.{txt,png,pdf}
"""
import json
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, B = "results/events_A_availability.jsonl", "results/events_B_intrusion.jsonl"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


a = [e for e in load(A) if e["type"] == "intrusion_detected"]
b = [e for e in load(B) if e["type"] == "intrusion_detected"]
A_WIN = 724                      # benign-only windows in the stream (from two_driver_events)
fa = np.array([e["n_flags"] for e in a])
fb = np.array([e["n_flags"] for e in b])
ka = np.array([e["k_votes"] for e in a])
kb = np.array([e["k_votes"] for e in b])

L = []
L.append("FALSE-POSITIVE BASELINE — stream A (benign-only traffic, no attacks present)")
L.append("")
L.append(f"benign-only windows      : {A_WIN}")
L.append(f"windows that fired       : {len(a)}  ({100*len(a)/A_WIN:.1f}%)")
L.append(f"windows silent           : {A_WIN-len(a)}  ({100*(A_WIN-len(a))/A_WIN:.1f}%)")
L.append("")
L.append("n_flags (alarm volume) per fired window")
L.append("                      FALSE (stream A)     TRUE (stream B)")
for lbl, f in [("count", None), ("min", np.min), ("median", np.median),
               ("mean", np.mean), ("p90", lambda x: np.percentile(x, 90)), ("max", np.max)]:
    va = len(fa) if f is None else f(fa)
    vb = len(fb) if f is None else f(fb)
    L.append(f"  {lbl:<8}            {va:12.1f}       {vb:12.1f}")
L.append("")
L.append("k_votes (corroboration) per fired window")
L.append(f"  median              {np.median(ka):12.1f}       {np.median(kb):12.1f}")
L.append(f"  max                 {np.max(ka):12.1f}       {np.max(kb):12.1f}")
L.append("")
L.append("Threshold sweep: escalate only when n_flags >= T")
L.append("   T    A: false alarms kept   B: true alarms kept   suppressed FP")
for T in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500):
    keep_a = int((fa >= T).sum())
    keep_b = int((fb >= T).sum())
    L.append(f"{T:5d} {keep_a:12d} ({100*keep_a/len(fa):5.1f}%) {keep_b:12d} "
             f"({100*keep_b/len(fb):5.1f}%) {len(fa)-keep_a:12d}")
L.append("")
lo, hi = max(fa.min(), fb.min()), min(fa.max(), fb.max())
n_a_in, n_b_in = int(((fa >= lo) & (fa <= hi)).sum()), int(((fb >= lo) & (fb <= hi)).sum())
L.append(f"overlapping n_flags range: [{lo}, {hi}]  -> {n_a_in} false ({100*n_a_in/len(fa):.0f}%) "
         f"and {n_b_in} true ({100*n_b_in/len(fb):.0f}%) alarms fall inside it")
T0 = int(fa.max()) + 1
keep = int((fb >= T0).sum())
L.append(f"threshold that removes EVERY false alarm here: T = {T0} "
         f"(max false volume {int(fa.max())}) -> keeps {keep}/{len(fb)} "
         f"({100*keep/len(fb):.1f}%) of the true ones")
L.append(f"true alarms that NO volume threshold can save: {int((fb < fa.max()).sum())} "
         f"(their volume is below the worst false positive)")
L.append("")
L.append("CAVEAT: these numbers are specific to this window size (1 s ~ 4.7k samples) and")
L.append("to this traffic. With ~0.6% FPR, a full window is expected to produce ~28 false")
L.append("flags, which is the scale seen here (median 56). Larger windows raise the false")
L.append("volume and push the threshold up; shorter windows lower both.")
txt = "\n".join(L)
print(txt)
os.makedirs("results", exist_ok=True)
open("results/fp_baseline_stream_a.txt", "w", encoding="utf-8").write(txt + "\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))
bins = np.logspace(0, np.log10(max(fa.max(), fb.max()) + 1), 28)
ax1.hist(fa, bins=bins, color="#b2182b", alpha=0.75, label=f"FALSE — stream A ({len(fa)})")
ax1.hist(fb, bins=bins, color="#2166ac", alpha=0.6, label=f"TRUE — stream B ({len(fb)})")
ax1.set_xscale("log")
ax1.set_xlabel("n_flags (alarm volume, log)")
ax1.set_ylabel("windows")
ax1.set_title("Alarm volume: benign-only vs attack windows", fontsize=10.5)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

Ts = np.unique(np.concatenate([np.arange(1, 50), np.logspace(np.log10(50), 3, 40).astype(int)]))
ax2.plot(Ts, [100 * (fb >= t).sum() / len(fb) for t in Ts], color="#2166ac", lw=2,
         label="true alarms kept (stream B)")
ax2.plot(Ts, [100 * (fa >= t).sum() / len(fa) for t in Ts], color="#b2182b", lw=2,
         label="false alarms kept (stream A)")
ax2.axvline(10, color="0.4", ls=":", lw=1.5)
ax2.text(10.5, 50, "T = 10", fontsize=9, color="0.35")
ax2.set_xscale("log")
ax2.set_xlabel("escalation threshold T  (escalate if n_flags >= T)")
ax2.set_ylabel("% of alarms kept")
ax2.set_title("Threshold trade-off", fontsize=10.5)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
fig.suptitle("False-positive baseline: what the monitor must filter", fontsize=12.5,
             fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("results/fp_baseline_stream_a.png", dpi=170, bbox_inches="tight")
fig.savefig("results/fp_baseline_stream_a.pdf", bbox_inches="tight")
print("\n-> results/fp_baseline_stream_a.{txt,png,pdf}")
