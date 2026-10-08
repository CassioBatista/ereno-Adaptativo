#!/usr/bin/env python3
"""Profile ablation on the corrected CICIDS2017 (RQ5): the same data and specialists under
two profiles, so that only the profile changes.

  it-flow          time-based local tick, attribution = source address (reference)
  it-flow-blind    count-based local tick (one tick every M flows, in arrival order, as
                   for a release without timestamps), attribution = none

Per-flow detection does not depend on the profile (features are the same), and is reported
once. What the profile changes is (a) the window the triage counts over and (b) whether an
alarm can name an emitter. The triage is the policy used everywhere else: network-wide
volume/fraction (margin x1.25) OR per-specialist counts (T_c = max(ceil(1.25 * max benign)
+ 1, 2)), calibrated on Monday's first 70 % (benign only). Count windows of M flows are
compared with 10-s time windows, M spanning the mean number of flows per 10 s of the test.
With attribution none, the alarm carries no source, so no isolation can follow.

  python scripts/_cicids_profile_ablation.py
Out: results/cicids2017c_profile_ablation.txt
"""
import io
import math
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402
from _cicids_source_triage import load  # noqa: E402

FLOOR = 2


def window_ids(secs, mode, size):
    """Window index of each row: floor(t / size) for time, rank // size for count."""
    if mode == "time":
        return np.floor(secs / size).astype(np.int64)
    order = np.argsort(secs, kind="stable")
    w = np.empty(len(secs), np.int64)
    w[order] = np.arange(len(secs)) // size
    return w


def counts(w, fused, V, ncat):
    per = np.column_stack([V[:, [c, c + ncat]].max(1) for c in range(ncat)]) * fused[:, None]
    d = pd.DataFrame(per)
    d["_f"], d["_n"] = fused, 1
    return d.groupby(w).sum()


def triage(mode, size, secs, cat, calib, test, fc, ft, Vc, Vt, ncat):
    wc = counts(window_ids(secs[calib], mode, size), fc, Vc, ncat)
    wt_ids = window_ids(secs[test], mode, size)
    wt = counts(wt_ids, ft, Vt, ncat)
    atk = pd.Series((cat[test] != "Benign").astype(int)).groupby(wt_ids).sum().reindex(wt.index).values > 0
    T = int(math.ceil(1.25 * wc["_f"].max())) + 1
    f = 1.25 * (wc["_f"] / wc["_n"]).max()
    Tc = np.maximum(np.ceil(1.25 * wc[list(range(ncat))].max().values) + 1, FLOOR)
    glob = ((wt["_f"] >= T) | ((wt["_f"] / wt["_n"]) > f)).values
    spec = (wt[list(range(ncat))].values >= Tc).any(1)
    comb = glob | spec
    dur = pd.Series(secs[test]).groupby(wt_ids).agg(lambda s: s.max() - s.min()).median()
    return {"attack windows": int(atk.sum()), "benign windows": int((~atk).sum()),
            "median duration (s)": round(float(dur), 2),
            "network-wide": (100 * glob[atk].mean(), int(glob[~atk].sum()), 100 * glob[~atk].mean()),
            "OR per specialist": (100 * comb[atk].mean(), int(comb[~atk].sum()), 100 * comb[~atk].mean())}


def main():
    X, cat, secs, src, srcip, train, calib, test = load()
    boosters, _ = C.train_specialists(X[train], cat[train], C.CATS_C)
    ncat = len(C.CATS_C)
    Vc, Vt = C.votes(boosters, X[calib]), C.votes(boosters, X[test])
    fc = (Vc.sum(1) >= 2).astype(np.int8)
    ft = (Vt.sum(1) >= 2).astype(np.int8)
    g = C.stats(ft.astype(bool), cat[test] != "Benign")
    print(f"per-flow detection (identical under both profiles): recall {g['recall']:.2f}%, "
          f"FPR {g['FPR']:.3f}%, F1 {g['F1']:.2f}")
    per10 = len(secs[test]) / max(1, len(np.unique(np.floor(secs[test] / 10))))
    print(f"mean test flows per occupied 10-s window: {per10:.0f}")
    rows = [("it-flow (time tick, 10 s)", "time", 10)]
    for m in (100, 300, 1000, 3000):
        rows.append((f"it-flow-blind (count tick, M={m})", "count", m))
    print(f"\n{'profile':<34} {'attack win':>10} {'benign win':>10} {'median s':>9} | "
          f"{'network-wide':>24} | {'network-wide OR per specialist':>32}")
    for name, mode, size in rows:
        r = triage(mode, size, secs, cat, calib, test, fc, ft, Vc, Vt, ncat)
        nw, cb = r["network-wide"], r["OR per specialist"]
        print(f"{name:<34} {r['attack windows']:>10,} {r['benign windows']:>10,} "
              f"{r['median duration (s)']:>9} | {nw[0]:5.1f}% FA {nw[1]:>4} ({nw[2]:.3f}%) | "
              f"{cb[0]:5.1f}% FA {cb[1]:>4} ({cb[2]:.3f}%)")
    print("\nattribution: it-flow names the source address of every alarm; it-flow-blind "
          "names none, so intrusion-driven isolation cannot be exercised under it.")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/cicids2017c_profile_ablation.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
