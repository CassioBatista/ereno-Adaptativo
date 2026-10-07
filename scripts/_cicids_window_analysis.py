#!/usr/bin/env python3
"""Why does the ERENO window policy catch only ~27 % of the attack windows of the corrected
CICIDS2017 (cross_domain_results.py)? Reuses its split and specialists; reports, per
window length, the n_flags threshold curve (TPR vs FPR on test windows; calibration on
Monday) and the per-category window recall at the margin threshold.
Out: results/cicids2017c_window_analysis.txt
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


def main():
    import zipfile
    z = zipfile.ZipFile(C.D + "cicids2017_improved/CICIDS2017_improved.zip")
    df = pd.concat([pd.read_csv(z.open(f"{d}.csv"), low_memory=False,
                                dtype={c: "str" for c in C.TEXT}).assign(_day=d)
                    for d in ["monday", "tuesday", "wednesday", "thursday", "friday"]],
                   ignore_index=True)
    df["_cat"] = df["Label"].map(C.cat_cicids)
    df = df[df["_cat"] != "drop"].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="mixed")
    feats = [c for c in df.columns if c not in C.TEXT + C.IDENT + C.FINGERPRINT + ["_day", "_cat", "_ts"]]
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float32").to_numpy()
    X[~np.isfinite(X)] = np.nan
    test = np.zeros(len(df), bool)
    for d, g in df.groupby("_day"):
        if d == "monday":
            continue
        for c, gc in g.groupby("_cat"):
            if c != "Benign":
                t70, tmax = gc["_ts"].quantile(0.7), gc["_ts"].max()
                test |= ((df["_day"] == d) & (df["_ts"] >= t70) & (df["_ts"] <= tmax)).to_numpy()
    mon = (df["_day"] == "monday").to_numpy()
    t70m = df.loc[mon, "_ts"].quantile(0.7)
    calib = mon & (df["_ts"] < t70m).to_numpy()
    test |= mon & (df["_ts"] >= t70m).to_numpy()
    train = ~test & ~calib
    cat = df["_cat"].to_numpy()
    boosters, _ = C.train_specialists(X[train], cat[train], C.CATS_C)
    secs = (df["_ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    k = 2
    fc = (C.votes(boosters, X[calib]).sum(1) >= k).astype(int)
    ft = (C.votes(boosters, X[test]).sum(1) >= k).astype(int)
    cte = cat[test]
    for wlen in (1, 10, 60):
        tc = secs[calib] / wlen
        o = np.argsort(tc, kind="stable")
        _, nfc, _, _ = C.windows(tc[o], fc[o], np.zeros(len(o), bool))
        tt = secs[test] / wlen
        o = np.argsort(tt, kind="stable")
        w = np.floor(tt[o]).astype(np.int64)
        uw, start = np.unique(w, return_index=True)
        end = np.r_[start[1:], len(w)]
        cs = np.r_[0, np.cumsum(ft[o])]
        nf = cs[end] - cs[start]
        cats_o = cte[o]
        wcat = []
        for s, e in zip(start, end):
            cc = cats_o[s:e]
            a = cc[cc != "Benign"]
            wcat.append(pd.Series(a).mode().iloc[0] if len(a) else "Benign")
        wcat = np.array(wcat)
        atk = wcat != "Benign"
        print(f"\n=== window {wlen}s: calibration windows={len(nfc):,} (benign); test: "
              f"{int(atk.sum()):,} attack / {int((~atk).sum()):,} benign")
        print(f"  calibration benign n_flags: max={nfc.max()}, 99.9%={np.quantile(nfc, .999):.0f}, "
              f"99%={np.quantile(nfc, .99):.0f}, share with >=1 flag={100 * (nfc >= 1).mean():.2f}%")
        print(f"  {'T':>4} {'cal FPR':>8} {'test TPR':>9} {'test FPR':>9}")
        for T in (1, 2, 3, 5, 10, int(math.ceil(nfc.max() * 1.25)) + 1):
            print(f"  {T:>4} {100 * (nfc >= T).mean():7.3f}% {100 * (nf[atk] >= T).mean():8.1f}% "
                  f"{100 * (nf[~atk] >= T).mean():8.3f}%")
        Tm = int(math.ceil(nfc.max() * 1.25)) + 1
        print(f"  per category at T={Tm} (margin policy) and at T=1:")
        for c in C.CATS_C:
            m = wcat == c
            if m.any():
                print(f"    {c:<13} windows={int(m.sum()):>5}  median n_flags={np.median(nf[m]):6.0f}  "
                      f"TPR@T={100 * (nf[m] >= Tm).mean():5.1f}%  TPR@1={100 * (nf[m] >= 1).mean():5.1f}%")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/cicids2017c_window_analysis.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
