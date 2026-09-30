#!/usr/bin/env python3
"""Diagnostic (not a paper artifact): is the ERENO all_in_one trace one stream or
overlapping per-scenario captures? Prints per-class time span/rate, timestamp
multiplicity and benign inter-arrival."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util

X, y, cv = util.load_arff("all_in_one_ereno_train.csv")
t = X[:, 0]                      # F1 = Time
hdr = ("class", "n", "t_min", "t_max", "span", "rate/s")
print("%22s %10s %9s %9s %8s %8s" % hdr)
for c in range(len(cv)):
    m = y == c
    if not m.any():
        continue
    tc = t[m]
    span = float(tc.max() - tc.min())
    print("%22s %10d %9.1f %9.1f %8.1f %8.1f"
          % (cv[c], m.sum(), tc.min(), tc.max(), span, m.sum() / span if span else 0))

u = np.unique(t)
print("\ndistinct timestamps: %d   records per timestamp: %.1f" % (len(u), len(t) / len(u)))
chg = int((np.diff(y) != 0).sum())
print("class changes along the file: %d  (contiguous blocks = %d)" % (chg, chg + 1))

tb = np.sort(t[y == util.normal_class])
d = np.diff(tb)
d = d[d > 0]
print("benign: n=%d  median gap=%.3f ms -> %.0f msg/s (within benign stream)"
      % (len(tb), np.median(d) * 1000, 1 / np.median(d)))
for c in range(len(cv)):
    if c == util.normal_class:
        continue
    ta = np.sort(t[y == c])
    da = np.diff(ta)
    da = da[da > 0]
    if len(da):
        print("  %22s median gap=%8.3f ms -> %7.2f msg/s" % (cv[c], np.median(da) * 1000, 1 / np.median(da)))
