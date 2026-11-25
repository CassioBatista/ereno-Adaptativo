#!/usr/bin/env python3
"""Window triage on the UNSW IoT attack traces (label set unsw-iot-3), with the policy used
for ERENO, the corrected CICIDS2017 and Electra: network-wide volume/fraction rule (margin
x1.25) OR per-specialist counts (T_c = max(ceil(1.25 * max benign) + 1, 2)).

Same specialists and split as the protocol of inspect_unsw_iot.py (6 specialists; attacks
split per category with intervals kept whole, normal minutes split at 70 % of each
device's time). That split is not one global time cut, so windows are built from the rows
of one side only: thresholds are calibrated on windows of training normal minutes and
applied to windows of test minutes. A window counts the fused alerts of all devices whose
minutes fall in it; one device contributes at most one row per minute, so a single-device
attack reaches the per-specialist floor of 2 only in windows of 2 minutes or more.

  python scripts/_unsw_window_triage.py
Out: results/unsw_window_triage.txt
"""
import io
import math
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402
import inspect_unsw_iot as U  # noqa: E402

WLENS = (1, 5, 10)  # minutes
FLOOR = 2


def split(X, A):
    names = sorted(X["_cat"].unique())
    tr = set()
    for c in names:
        if c == "Normal":
            continue
        iv = A[A["cat"] == c].sort_values("start")["iid"].values
        tr |= set(iv[: int(round(0.7 * len(iv)))])
    tcut = {d: g["_ts"].quantile(0.7) for d, g in X.groupby("_dev")}
    return np.where(X["_cat"].values == "Normal", X["_ts"].values < X["_dev"].map(tcut).values,
                    np.isin(X["_iid"].values, list(tr)))


def specialists(Xa, cat, is_tr, cats):
    Xtr, ctr = Xa[is_tr], cat[is_tr]
    N = 2 * len(cats)
    ben = np.where(ctr == "Normal")[0]
    shares = np.array_split(np.random.default_rng(C.SEED).permutation(ben), N)
    rng2 = np.random.default_rng(C.SEED)
    halves = {c: np.array_split(rng2.permutation(np.where(ctr == c)[0]), 2) for c in cats}
    models = []
    for i in range(N):
        c = cats[i % len(cats)]
        pos, neg = halves[c][i // len(cats)], shares[i]
        rows = np.r_[pos, neg]
        prm = dict(C.XGB); prm["scale_pos_weight"] = len(neg) / max(len(pos), 1)
        models.append(xgb.train(prm, xgb.DMatrix(Xtr[rows], label=np.r_[np.ones(len(pos)), np.zeros(len(neg))]),
                                num_boost_round=10))
    return np.column_stack([(m.predict(xgb.DMatrix(Xa)) >= 0.5).astype(np.int8) for m in models])


def windows(minute, fused, V, cats, wlen):
    w = np.floor(minute / wlen).astype(np.int64)
    ncat = len(cats)
    per = np.column_stack([V[:, [c, c + ncat]].max(1) for c in range(ncat)]) * fused[:, None]
    d = pd.DataFrame(per, columns=cats)
    d["_n"] = 1
    d["_f"] = fused
    return d.groupby(w).sum(), w


def main():
    X, A = U.load()
    feats = [c for c in X.columns if not c.startswith("_")]
    is_tr = split(X, A)
    keep = np.isin(X["_cat"].values, U.CATS3 + ["Normal"])
    X, is_tr = X[keep].reset_index(drop=True), is_tr[keep]
    cats = U.CATS3
    cat = X["_cat"].values
    V = specialists(X[feats].to_numpy(np.float32), cat, is_tr, cats)
    fused = (V.sum(1) >= 2).astype(np.int8)
    atk = cat != "Normal"
    g = C.stats(fused[~is_tr].astype(bool), atk[~is_tr])
    print(f"per-minute check (test, GL k>=2): recall {g['recall']:.2f}%  FPR {g['FPR']:.4f}% (FP {g['FP']})")
    minute = np.floor(X["_ts"].values / 60)
    iid = X["_iid"].values

    cal_rows = is_tr & ~atk
    ev_rows = ~is_tr
    for wlen in WLENS:
        cal, _ = windows(minute[cal_rows], fused[cal_rows], V[cal_rows], cats, wlen)
        ev, wid = windows(minute[ev_rows], fused[ev_rows], V[ev_rows], cats, wlen)
        sub = pd.DataFrame({"w": wid, "atk": atk[ev_rows], "cat": cat[ev_rows], "iid": iid[ev_rows]})
        ev["atk"] = sub.groupby("w")["atk"].any().reindex(ev.index).values
        ev["cls"] = sub[sub["atk"]].groupby("w")["cat"].agg(lambda s: s.mode().iloc[0]).reindex(ev.index).fillna("Normal").values
        T = int(math.ceil(1.25 * cal["_f"].max())) + 1
        f = 1.25 * (cal["_f"] / cal["_n"]).max()
        Tc = np.maximum(np.ceil(1.25 * cal[cats].max().values) + 1, FLOOR)
        glob = ((ev["_f"] >= T) | ((ev["_f"] / ev["_n"]) > f)).values
        spec = (ev[cats].values >= Tc).any(1)
        a = ev["atk"].values
        days = (~a).sum() * wlen / 1440
        print(f"\n=== window {wlen} min: calibration {len(cal):,} windows of training normal minutes; "
              f"test {int(a.sum()):,} attack / {int((~a).sum()):,} benign windows ({days:.1f} days)")
        print(f"  thresholds: T={T}, f={f:.4f}; per specialist "
              + ", ".join(f"{c}={int(t)}" for c, t in zip(cats, Tc))
              + f"; max fused alerts in a calibration window {int(cal['_f'].max())}")
        for name, r in (("network-wide (ERENO policy)", glob), ("per specialist", spec),
                        ("network-wide OR per specialist", glob | spec)):
            print(f"  {name:<31} attack windows {int(r[a].sum())}/{int(a.sum())} ({100 * r[a].mean():.1f}%)   "
                  f"benign false alarms {int(r[~a].sum())}/{int((~a).sum())} ({100 * r[~a].mean():.3f}%; "
                  f"{r[~a].sum() / days:.2f} per day)")
        comb = pd.Series(glob | spec, index=ev.index)
        print("  per category (OR rule):",
              {c: f"{100 * comb[(ev['cls'] == c).values].mean():.0f}% of {int((ev['cls'] == c).sum())}" for c in cats})
        s = sub[sub["atk"]]
        hit = comb.reindex(s["w"].values).values
        det = pd.Series(hit, index=s.index).groupby(s["iid"].values).max()
        print(f"  attack intervals with an alarmed window (OR rule): {int(det.sum())}/{len(det)}")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/unsw_window_triage.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
