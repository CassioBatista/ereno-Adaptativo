#!/usr/bin/env python3
"""Diagnostic: for the N=14 experiment (2 specialists per attack, benign_cap=500k),
how long would each client wait to observe its OWN training quota on the wire?

Per client: K_attack = class_count/2, K_benign = 500000/14. Attack and benign arrive
concurrently, so the wait is the LARGER of the two (not their sum). Times are measured
on the trace timestamps (F1), from each class's first message.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util

NODES = 14
BENIGN_CAP = 500_000

X, y, cv = util.load_arff("all_in_one_ereno_train.csv")
t = X[:, 0]
nc = util.normal_class

tb = np.sort(t[y == nc])
kb = BENIGN_CAP // NODES
tb_wait = float(tb[kb - 1] - tb[0]) if len(tb) >= kb else float("nan")
print("benign per client: K=%d -> %.1f s (of %d available, span %.0f s)"
      % (kb, tb_wait, len(tb), tb[-1] - tb[0]))

print("\n%22s %9s %10s %10s %10s" % ("attack", "K/client", "t_attack", "t_benign", "wait=max"))
for c in range(len(cv)):
    if c == nc:
        continue
    ta = np.sort(t[y == c])
    k = len(ta) // 2                      # 2 specialists per attack at N=14
    ta_wait = float(ta[k - 1] - ta[0])
    print("%22s %9d %10.1f %10.1f %10.1f"
          % (cv[c], k, ta_wait, tb_wait, max(ta_wait, tb_wait)))
print("\n(t_attack: seconds to observe K attack samples; t_benign: to observe %d benign;"
      " both accumulate in parallel)" % kb)
