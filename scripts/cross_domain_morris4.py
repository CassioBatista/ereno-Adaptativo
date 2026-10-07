#!/usr/bin/env python3
"""ReSIDS evaluation protocol on Morris-4 (new gas pipeline, 2015): the decisive test after
the audit (scripts/inspect_morris_ics.py), which found real timestamps with attacks
interleaved throughout, but 27% of the normal packets sharing their feature vector with
attack packets.

Protocol as cross_domain_results.py: 14 XGBoost specialists (two per category, depth 4,
eta 0.1, 10 rounds, scale_pos_weight, seed 42), k-of-n fusion, GL = retained union, FL with
fewer nodes. Features: Modbus and process fields, without time and crc rate (the probe
showed both act as capture artefacts). Missing values stay NaN. Split: global time order,
first 70 % train, last 30 % test (attacks are interleaved, so every category is in both).

  python scripts/cross_domain_morris4.py
Out: results/cross_domain_morris4.txt
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
from inspect_morris_ics import NAMES, load_arff  # noqa: E402

CATS = [NAMES[k] for k in range(1, 8)]


def main():
    df = load_arff(os.path.expanduser("~/datasets/morris_ics/IanArffDataset.arff"))
    cat = df["categorized result"].astype(int).map(NAMES).values
    df = df.sort_values("time", kind="stable")
    cat = cat[df.index.values]
    feats = [c for c in df.columns if c not in ("time", "crc rate", "binary result",
                                                "categorized result", "specific result")]
    X = df[feats].to_numpy(np.float32)
    n = len(X); cut = int(0.7 * n)
    tr = np.arange(n) < cut
    print(f"features ({len(feats)}): {feats}")
    print("train:", pd.Series(cat[tr]).value_counts().to_dict())
    print("test: ", pd.Series(cat[~tr]).value_counts().to_dict())
    rng = np.random.default_rng(C.SEED)
    Xtr, ctr = X[tr], cat[tr]
    ben = np.where(ctr == "normal")[0]
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
    atk = cte != "normal"
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
          {CATS[i % 7] + f"#{i // 7}": int(V[~atk & f2, i].sum()) for i in range(14)})


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/cross_domain_morris4.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
