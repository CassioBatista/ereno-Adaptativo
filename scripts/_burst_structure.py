#!/usr/bin/env python3
"""Diagnostic: burst structure of the ERENO trace.

Within a burst every record is one SV sample apart (~0.21 ms). Bursts are separated by
idle gaps. This separates "active" time (K x 0.21 ms) from wall-clock time (which
includes the idle gaps), for the merged stream and per class.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util

GAP = 0.005          # > 5 ms between consecutive records = new burst
X, y, cv = util.load_arff("all_in_one_ereno_train.csv")
t = X[:, 0]
nc = util.normal_class


def report(name, ts):
    ts = np.sort(ts)
    d = np.diff(ts)
    small = d[(d > 0) & (d <= GAP)]
    big = d[d > GAP]
    active = float(small.sum())
    wall = float(ts[-1] - ts[0])
    print("%22s n=%8d wall=%7.0fs active=%7.1fs (%4.1f%%) bursts=%5d "
          "median_gap=%.3fms idle_median=%6.2fs"
          % (name, len(ts), wall, active, 100 * active / wall if wall else 0,
             len(big) + 1, np.median(small) * 1000 if len(small) else 0,
             np.median(big) if len(big) else 0))


report("ALL (merged)", t)
report("normal", t[y == nc])
for c in range(len(cv)):
    if c != nc:
        report(cv[c], t[y == c])
print("\nGAP threshold = %g s; 'active' = time spent inside bursts (sum of small gaps)" % GAP)
