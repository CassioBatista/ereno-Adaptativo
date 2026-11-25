"""Window alarms by the number of attack samples in the attack window (primary windows; same
rule and blocked-5 calibration as scripts/ev1_windows.py). Read-only on the step-1 caches."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo/scripts"))
import ev1_windows as E

BINS = [(1, 1), (2, 9), (10, 99), (100, 999), (1000, 10 ** 9)]
for d, size in (("ereno", 1.0), ("cicids", 10.0), ("morris4", 100.0)):
    dom = E.DOMS[d]
    z = np.load(os.path.join(E.ROOT, f"results/.cache/ev1_step1_{d}_s42.npz"), allow_pickle=True)
    y, nc = z["y"].astype(np.int64), int(z["nc"])
    t = E.times(dom)
    o = np.argsort(t, kind="stable")
    t, fired, y = t[o], z["Ps"][o] >= 0.5, y[o]
    w = np.floor((t - t[0]) / size).astype(np.int64)
    W = E.window_table(w, fired, y, nc, [])
    starts = np.flatnonzero(np.r_[True, w[1:] != w[:-1]])
    na = np.add.reduceat((y != nc).astype(np.int64), starts)
    net, full, log, scored = E.evaluate(W)
    A = W["atk"] & scored
    print(f"== {dom['name']} {size:g}-s: attack windows {int(A.sum())}; by attack samples in the window: "
          "windows / alarmed (full rule)")
    for lo, hi in BINS:
        m = A & (na >= lo) & (na <= hi)
        if m.any():
            print(f"   {lo:>5}-{'' if hi > 10 ** 8 else hi:<5} {int(m.sum()):>5}  {int(full[m].sum()):>5}  "
                  f"({100 * full[m].mean():5.1f} %)")
    # single-sample sensitivity: smallest per-specialist threshold in each fold
    print("   T_s floor is 2: a window whose only flagged attack sample is one cannot alarm through S_s; "
          f"T (volume) per fold {[g['T'] for g in log]}")
