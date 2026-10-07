#!/usr/bin/env python3
"""ReSIDS evaluation protocol on Electra Modbus: the decisive test after the audit
(scripts/inspect_electra.py). The audit found two kinds of labels. MITM_UNALTERED,
REPLAY_ATTACK and most READ_ATTACK packets have the content of normal packets and are told
apart only by the MitM host's MAC, so their label marks the path, not the packet. The
other four types (recognition, write, force error, response modification) are separable by
content and never share a vector with normal traffic.

Variants (rows of the excluded types are removed from the stream):
  clean4  RECOGNITION, WRITE, FORCE_ERROR, RESPONSE          -> N = 8
  clean5  the four above + READ_ATTACK                       -> N = 10
Protocol as cross_domain_results.py: two XGBoost specialists per category (depth 4, eta 0.1,
10 rounds, scale_pos_weight, seed 42), k-of-n fusion, GL = retained union, FL with fewer
nodes. Features: packet content only (request, fc, error, address, data); MACs, IPs and
time feed the profile, never the features. Split: global time order, first 70 % train,
last 30 % test (attacks are interleaved over the whole capture).

  python scripts/cross_domain_electra.py
Out: results/cross_domain_electra.txt
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

CONTENT = ["request", "fc", "error", "address", "data"]
CLEAN4 = ["RECOGNITION_ATTACK", "WRITE_ATTACK", "FORCE_ERROR_ATTACK", "RESPONSE_ATTACK"]
VARIANTS = {"clean4": CLEAN4, "clean5": CLEAN4 + ["READ_ATTACK"]}


def run(df, cats):
    d = df[df["label"].isin(cats + ["NORMAL"])].sort_values("Time", kind="stable")
    X = d[CONTENT].to_numpy(np.float32)
    lab = d["label"].astype(str).values
    n = len(X); cut = int(0.7 * n); tr = np.arange(n) < cut
    print("train:", pd.Series(lab[tr]).value_counts().to_dict())
    print("test: ", pd.Series(lab[~tr]).value_counts().to_dict())
    N = 2 * len(cats)
    rng = np.random.default_rng(C.SEED)
    Xtr, ctr = X[tr], lab[tr]
    ben = np.where(ctr == "NORMAL")[0]
    shares = np.array_split(rng.permutation(ben), N)
    halves = {c: np.array_split(rng.permutation(np.where(ctr == c)[0]), 2) for c in cats}
    models = []
    for i in range(N):
        c = cats[i % len(cats)]
        pos, neg = halves[c][i // len(cats)], shares[i]
        rows = np.r_[pos, neg]
        p = dict(C.XGB); p["scale_pos_weight"] = len(neg) / max(len(pos), 1)
        models.append(xgb.train(p, xgb.DMatrix(Xtr[rows], label=np.r_[np.ones(len(pos)), np.zeros(len(neg))]),
                                num_boost_round=10))
    Xte, cte = X[~tr], lab[~tr]
    V = np.column_stack([(m.predict(xgb.DMatrix(Xte)) >= 0.5).astype(np.int8) for m in models])
    atk = cte != "NORMAL"
    for k in (1, 2):
        g = C.stats(V.sum(1) >= k, atk)
        print(f"GL (union of {N}) k>={k}: recall {g['recall']:.2f}%  FPR {g['FPR']:.4f}% (FP {g['FP']})  "
              f"F1 {g['F1']:.2f}  F1@13.75 {g['F1@13.75']:.2f}")
    f2 = V.sum(1) >= 2
    print("per category recall (GL k>=2):", {c: round(100 * f2[cte == c].mean(), 1) for c in cats})
    for nn in sorted({len(cats), 3}, reverse=True):
        g = C.stats(V[:, :nn].sum(1) >= 2, atk)
        print(f"FL k>=2 with {nn} nodes: recall {g['recall']:.1f}%  FPR {g['FPR']:.4f}%  "
              f"per category {({c: round(100 * (V[:, :nn].sum(1) >= 2)[cte == c].mean(), 1) for c in cats})}")


def main():
    df = pd.read_csv(os.path.expanduser("~/datasets/electra/electra_modbus.csv"),
                     usecols=["Time"] + CONTENT + ["label"], dtype={"label": "category"})
    for name, cats in VARIANTS.items():
        print(f"\n===== {name}: {cats} =====")
        run(df, cats)


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/cross_domain_electra.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
