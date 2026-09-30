#!/usr/bin/env python3
"""Diagnostic: how long does a node wait to buffer the first 30/100/1000 samples of
each ERENO attack, measured on the trace timestamps (F1), from the attack's first
occurrence. Feeds the learning-latency part of the round-time estimate."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util

X, y, cv = util.load_arff("all_in_one_ereno_train.csv")
t = X[:, 0]
print("%22s %8s %9s %9s %9s %10s" % ("attack", "n", "t@30", "t@100", "t@1000", "burst/s"))
for c in range(len(cv)):
    if c == util.normal_class:
        continue
    ta = np.sort(t[y == c])
    if not len(ta):
        continue
    t0 = ta[0]

    def w(k):
        return (ta[k - 1] - t0) if len(ta) >= k else float("nan")
    d = np.diff(ta)
    d = d[d > 0]
    burst = 1 / np.median(d) if len(d) else 0
    print("%22s %8d %9.2f %9.2f %9.2f %10.0f" % (cv[c], len(ta), w(30), w(100), w(1000), burst))
print("\n(t@k = seconds from the attack's first message until k messages are buffered)")
