#!/usr/bin/env python3
"""ReSIDS evaluation protocol on X-IIoTID: the decisive test after the audit
(scripts/inspect_xiiotid.py), which found consistent labels but host statistics acting as a
session clock and normal traffic that loses separability on a time-ordered split.

Label set: the seven class2 attack types with enough records (RDOS, Reconnaissance,
Weaponization, Lateral movement, Exfiltration, Tampering, C&C) -> N = 14; Exploitation
(1,133) and crypto-ransomware (458, timestamps missing) are removed from the stream.
Features: flow fields only (no host statistics, alerts, logs, addresses, ports or time).
Protocol as cross_domain_results.py. Split: time-ordered per class (normal included),
first 70 % train, last 30 % test.

  python scripts/cross_domain_xiiotid.py
Out: results/cross_domain_xiiotid.txt
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402
from inspect_xiiotid import groups  # noqa: E402

CATS = ["RDOS", "Reconnaissance", "Weaponization", "Lateral _movement", "Exfiltration",
        "Tampering", "C&C"]


def main():
    df = pd.read_csv(os.path.expanduser("~/datasets/xiiotid/xiiotid.csv"), low_memory=False)
    df = df[df["class2"].isin(CATS + ["Normal"])].reset_index(drop=True)
    flow, _, _ = groups(df.columns)
    X = df[flow].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.factorize(X[c])[0]
    X = X.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    cat = df["class2"].values
    ts = pd.to_numeric(df["Timestamp"], errors="coerce").fillna(0).values
    tr = np.zeros(len(df), bool)
    for c in CATS + ["Normal"]:
        ii = np.where(cat == c)[0]
        ii = ii[np.argsort(ts[ii], kind="stable")]
        tr[ii[: int(0.7 * len(ii))]] = True
    print(f"flow features ({len(flow)}): {flow}")
    print("train:", pd.Series(cat[tr]).value_counts().to_dict())
    print("test: ", pd.Series(cat[~tr]).value_counts().to_dict())
    rng = np.random.default_rng(C.SEED)
    Xtr, ctr = X[tr], cat[tr]
    ben = np.where(ctr == "Normal")[0]
    shares = np.array_split(rng.permutation(ben), 14)
    halves = {c: np.array_split(rng.permutation(np.where(ctr == c)[0]), 2) for c in CATS}
    models = []
    for i in range(14):
        c = CATS[i % 7]
        pos, neg = halves[c][i // 7], shares[i]
        rows = np.r_[pos, neg]
        p = dict(C.XGB); p["scale_pos_weight"] = len(neg) / max(len(pos), 1)
        models.append(xgb.train(p, xgb.DMatrix(Xtr[rows], label=np.r_[np.ones(len(pos)), np.zeros(len(neg))]),
                                num_boost_round=10))
    Xte, cte = X[~tr], cat[~tr]
    V = np.column_stack([(m.predict(xgb.DMatrix(Xte)) >= 0.5).astype(np.int8) for m in models])
    atk = cte != "Normal"
    for k in (1, 2):
        g = C.stats(V.sum(1) >= k, atk)
        print(f"GL (union of 14) k>={k}: recall {g['recall']:.2f}%  FPR {g['FPR']:.3f}%  "
              f"F1 {g['F1']:.2f}  F1@13.75 {g['F1@13.75']:.2f}")
    f2 = V.sum(1) >= 2
    print("per category recall (GL k>=2):", {c: round(100 * f2[cte == c].mean(), 1) for c in CATS})
    for nn in (7, 3):
        g = C.stats(V[:, :nn].sum(1) >= 2, atk)
        print(f"FL k>=2 with {nn} nodes: recall {g['recall']:.1f}%  FPR {g['FPR']:.3f}%")
    print("specialists firing on false positives:",
          {CATS[i % 7][:8] + f"#{i // 7}": int(V[~atk & f2, i].sum()) for i in range(14)})


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/cross_domain_xiiotid.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
