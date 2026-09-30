#!/usr/bin/env python3
"""Diagnostic: can the IDS keep up with the SV cadence (~0.21 ms per sample)?

Scores a pool of boosters over a batch and reports per-sample cost, against the
0.21 ms budget measured on the ERENO trace. Also reports single-row latency, which
is dominated by per-call overhead and is NOT how a deployment should run.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
import python.util as util

FEAT = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
POOL = 20          # diffused booster pool (~2N-1 at N=10)
BUDGET_MS = 0.214  # measured median inter-arrival on the merged trace

X, y, cv = util.load_arff("all_in_one_ereno_train.csv")
nc = util.normal_class
Xf = util.filter_features(X, FEAT)
rng = np.random.default_rng(0)
a = np.where(y == 5)[0]
b = rng.permutation(np.where(y == nc)[0])[:41667]
idx = np.concatenate([a, b])
lab = (y[idx] != nc).astype(int)
p = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "nthread": 4,
     "scale_pos_weight": (lab == 0).sum() / lab.sum()}
bst = xgb.train(p, xgb.DMatrix(Xf[idx], label=lab), num_boost_round=10)
pool = [bst] * POOL

for n in (1, 100, 1000, 10000, 100000):
    d = xgb.DMatrix(Xf[:n])
    reps = 50 if n <= 1000 else 5
    t0 = time.perf_counter()
    for _ in range(reps):
        for m in pool:
            m.predict(d)
    per_sample_ms = (time.perf_counter() - t0) / reps / n * 1000
    print("batch=%7d  %8.4f ms/sample (pool of %d)  -> %s budget %.3f ms  [%.0fx]"
          % (n, per_sample_ms, POOL,
             "WITHIN" if per_sample_ms <= BUDGET_MS else "OVER  ", BUDGET_MS,
             BUDGET_MS / per_sample_ms if per_sample_ms else 0))
print("\nSV cadence 0.214 ms/sample = %.0f samples/s sustained" % (1000 / BUDGET_MS))
