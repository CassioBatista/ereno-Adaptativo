#!/usr/bin/env python3
"""Follow-up to the shortcut probe of inspect_unsw_iot.py: on the time-ordered split with
attack intervals kept whole, ping of death and Smurf fall to F1 0.00 and 0.52. Are they
confused with each other (both ICMP) or with normal traffic?

Same sample, split and probe as inspect_unsw_iot.py (counters only), plus:
  (a) the confusion matrix of the multiclass probe;
  (b) the probe with the two ICMP categories merged;
  (c) one binary probe per category against normal traffic (other attacks removed).

  python scripts/_unsw_probe_confusion.py
Out: results/unsw_probe_confusion.txt
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspect_unsw_iot as U  # noqa: E402

PRM = {"max_depth": 6, "eta": 0.3, "tree_method": "hist", "nthread": 4, "seed": 42}


def multiclass(Xf, y, n, tr, te):
    m = xgb.train({**PRM, "objective": "multi:softmax", "num_class": n},
                  xgb.DMatrix(Xf.iloc[tr], label=y[tr]), num_boost_round=100)
    return m.predict(xgb.DMatrix(Xf.iloc[te])).astype(int)


def main():
    X, A = U.load()
    feats = [c for c in X.columns if not c.startswith("_")]
    y, names = pd.factorize(X["_cat"], sort=True)
    Xf = X[feats]
    rng = np.random.RandomState(42)
    norm = np.where(X["_cat"].values == "Normal")[0]
    idx = np.sort(np.r_[np.where(X["_cat"].values != "Normal")[0], rng.choice(norm, 30_000, replace=False)])
    tr = set()
    for c in names:
        if c == "Normal":
            continue
        iv = A[A["cat"] == c].sort_values("start")["iid"].values
        tr |= set(iv[: int(round(0.7 * len(iv)))])
    tcut = {d: g["_ts"].quantile(0.7) for d, g in X.groupby("_dev")}
    is_tr = np.where(X["_cat"].values == "Normal", X["_ts"].values < X["_dev"].map(tcut).values,
                     np.isin(X["_iid"].values, list(tr)))
    ta, tb = idx[is_tr[idx]], idx[~is_tr[idx]]

    print("=== (a) multiclass probe, counters, time split with intervals whole ===")
    p = multiclass(Xf, y, len(names), ta, tb)
    per = f1_score(y[tb], p, average=None, labels=range(len(names)), zero_division=0)
    print(f"macro-F1={per.mean():.3f}  " + "  ".join(f"{k[:9]}={v:.2f}" for k, v in zip(names, per)))
    cm = pd.DataFrame(confusion_matrix(y[tb], p, labels=range(len(names))),
                      index=[f"true {k}" for k in names], columns=[k[:9] for k in names])
    print(cm.to_string())
    for c in ("PingOfDeath", "Smurf"):
        row = cm.loc[f"true {c}"]
        print(f"  {c}: {row.sum()} test minutes -> " + ", ".join(f"{k} {v}" for k, v in row.items() if v))

    print("\n=== (b) ping of death and Smurf merged into one ICMP category ===")
    cat2 = X["_cat"].replace({"PingOfDeath": "Icmp", "Smurf": "Icmp"})
    y2, n2 = pd.factorize(cat2, sort=True)
    p2 = multiclass(Xf, y2, len(n2), ta, tb)
    per2 = f1_score(y2[tb], p2, average=None, labels=range(len(n2)), zero_division=0)
    print(f"macro-F1={per2.mean():.3f}  " + "  ".join(f"{k[:9]}={v:.2f}" for k, v in zip(n2, per2)))

    print("\n=== (c) binary probe per category against normal (other attacks removed) ===")
    cats = X["_cat"].values
    for c in U.CATS:
        tr_c = ta[np.isin(cats[ta], [c, "Normal"])]
        te_c = tb[np.isin(cats[tb], [c, "Normal"])]
        yb = (cats == c).astype(int)
        m = xgb.train({**PRM, "objective": "binary:logistic"},
                      xgb.DMatrix(Xf.iloc[tr_c], label=yb[tr_c]), num_boost_round=100)
        pb = (m.predict(xgb.DMatrix(Xf.iloc[te_c])) >= 0.5).astype(int)
        tp = int(((pb == 1) & (yb[te_c] == 1)).sum())
        fp = int(((pb == 1) & (yb[te_c] == 0)).sum())
        print(f"  {c:<18} test minutes {int(yb[te_c].sum()):>4}  recall {100 * tp / max(yb[te_c].sum(), 1):5.1f}%  "
              f"FP {fp:>4} of {int((yb[te_c] == 0).sum()):,} normal  F1={f1_score(yb[te_c], pb, zero_division=0):.2f}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    out = buf.getvalue()
    print(out)
    os.makedirs("results", exist_ok=True)
    open("results/unsw_probe_confusion.txt", "w").write(out)
