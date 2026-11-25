#!/usr/bin/env python3
"""Numbers of the API section on false positives and the window (sec:api-fp), recomputed on the
evaluation-v1 detector: ERENO, the 14 specialists of step 1, OR at 0.5 (results/.cache/
ev1_step1_ereno_s<seed>.npz), test split in time order. Replaces the earlier values, computed
with fusion k >= 2 on the model of the scenario streams (scripts/window_prevalence_sweep.py,
scripts/cadence_sweep.py).

1. Load of false positives, 1-s windows: attack-free windows with at least one flagged sample;
   median flagged samples in those windows and in the attack windows.
2. Cadence: the network-wide rule (n_f >= T OR fraction > f; margin 1.25 over the largest
   value in the attack-free windows) calibrated ONCE at 1 s on all attack-free windows, then
   held fixed while the window is 0.25, 0.5, 2, 5 and 10 s, as in cadence_sweep.py.
   (The out-of-sample rates of the 1-s window are those of scripts/ev1_windows.py.)

  python scripts/ev1_windows_cadence.py [--seed 42]
Out: results/ev1_windows_cadence_ereno_s<seed>.txt
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev1_windows as E  # noqa: E402

TICKS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


def main(seed):
    z = np.load(os.path.join(E.ROOT, f"results/.cache/ev1_step1_ereno_s{seed}.npz"), allow_pickle=True)
    y, nc = z["y"].astype(np.int64), int(z["nc"])
    t = E.times(E.DOMS["ereno"])
    o = np.argsort(t, kind="stable")
    t, fired, y = t[o], z["Ps"][o] >= 0.5, y[o]
    W = {}
    for s in TICKS:
        w = np.floor((t - t[0]) / s).astype(np.int64)
        W[s] = E.window_table(w, fired, y, nc, [])
    w1 = W[1.0]
    B, A = ~w1["atk"], w1["atk"]
    fpw = B & (w1["nf"] >= 1)
    lines = [f"ERENO, OR at 0.5 over the 14 specialists of step 1, seed {seed}, 1-s windows",
             f"attack-free windows with at least one flagged sample: {int(fpw.sum())} of {int(B.sum())} "
             f"({100 * fpw.mean() / B.mean():.1f} %)",
             f"median flagged samples: {np.median(w1['nf'][fpw]):.0f} in those windows; "
             f"{np.median(w1['nf'][A]):.0f} in the {int(A.sum())} attack windows", ""]
    T = int(math.ceil(E.MARGIN * w1["nf"][B].max())) + 1
    f = E.MARGIN * float(w1["frac"][B].max())
    lines.append(f"network-wide rule calibrated once at 1 s on the {int(B.sum())} attack-free windows: "
                 f"T = {T}, f = {f:.4f}; then held fixed")
    lines.append(f"{'window':>8} {'attack windows alarmed':>26} {'attack-free windows alarmed':>30}")
    for s in TICKS:
        q = W[s]
        al = (q["nf"] >= T) | (q["frac"] > f)
        a, b = q["atk"], ~q["atk"]
        lines.append(f"{s:>7g}s {int(al[a].sum()):>8}/{int(a.sum()):<6} ({100 * al[a].mean():5.1f} %) "
                     f"{int(al[b].sum()):>10}/{int(b.sum()):<6} ({100 * al[b].mean():5.2f} %)")
    # out of sample: blocked-5 on the 1-s windows (as scripts/ev1_windows.py); each block is a
    # time interval; thresholds calibrated on the attack-free 1-s windows of the other four
    # blocks are applied to the windows of every length that start inside the block
    lines += ["", "out of sample, blocked-5: thresholds calibrated on the attack-free 1-s windows of the "
                  "other four blocks, applied to the windows of each length that start in the block"]
    w1id = np.unique(np.floor((t - t[0]) / 1.0).astype(np.int64))          # occupied 1-s windows
    folds = np.array_split(np.arange(len(w1id)), E.BLOCKS)
    edges = [float(w1id[f[0]]) for f in folds] + [np.inf]                   # block starts, s from t[0]
    th = []
    for k, held in enumerate(folds):
        cal = np.setdiff1d(np.arange(len(w1id)), held)
        b = cal[~w1["atk"][cal]]
        th.append((int(math.ceil(E.MARGIN * w1["nf"][b].max())) + 1, E.MARGIN * float(w1["frac"][b].max())))
    lines.append("  thresholds per block: " + "; ".join(f"{k}: T={a}, f={c:.4f}" for k, (a, c) in enumerate(th)))
    lines.append(f"{'window':>8} {'attack windows alarmed':>26} {'attack-free windows alarmed':>30}")
    for s in TICKS:
        q = W[s]
        w = np.floor((t - t[0]) / s).astype(np.int64)
        start = np.unique(w) * s                                            # window start, s from t[0]
        blk = np.searchsorted(edges, start, side="right") - 1
        T_ = np.array([th[k][0] for k in blk]); f_ = np.array([th[k][1] for k in blk])
        al = (q["nf"] >= T_) | (q["frac"] > f_)
        a, b = q["atk"], ~q["atk"]
        lines.append(f"{s:>7g}s {int(al[a].sum()):>8}/{int(a.sum()):<6} ({100 * al[a].mean():5.1f} %) "
                     f"{int(al[b].sum()):>10}/{int(b.sum()):<6} ({100 * al[b].mean():5.2f} %)")
    out = os.path.join(E.ROOT, f"results/ev1_windows_cadence_ereno_s{seed}.txt")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
