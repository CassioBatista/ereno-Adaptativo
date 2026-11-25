"""Out-of-sample cadence (scripts/ev1_windows_cadence.py): are the single false alarms at 1, 2,
5 and 10 s the same stretch of traffic? Prints each false-alarm window (interval from the start
of the trace, block, n, n_f, fraction, which rule fired) and whether it contains the 1-s one.
Read-only."""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo/scripts"))
import ev1_windows as E

z = np.load(os.path.join(E.ROOT, "results/.cache/ev1_step1_ereno_s42.npz"), allow_pickle=True)
y, nc = z["y"].astype(np.int64), int(z["nc"])
t = E.times(E.DOMS["ereno"]); o = np.argsort(t, kind="stable")
t, fired, y = t[o], z["Ps"][o] >= 0.5, y[o]
rel = t - t[0]
w1 = np.floor(rel).astype(np.int64)
W1 = E.window_table(w1, fired, y, nc, [])
ids1 = np.unique(w1)
folds = np.array_split(np.arange(len(ids1)), E.BLOCKS)
edges = [float(ids1[f[0]]) for f in folds] + [np.inf]
th = []
for held in folds:
    cal = np.setdiff1d(np.arange(len(ids1)), held)
    b = cal[~W1["atk"][cal]]
    th.append((int(math.ceil(E.MARGIN * W1["nf"][b].max())) + 1, E.MARGIN * float(W1["frac"][b].max())))
fa = {}
for s in (0.5, 1.0, 2.0, 5.0, 10.0):
    w = np.floor(rel / s).astype(np.int64)
    q = E.window_table(w, fired, y, nc, [])
    start = np.unique(w) * s
    blk = np.searchsorted(edges, start, side="right") - 1
    T_ = np.array([th[k][0] for k in blk]); f_ = np.array([th[k][1] for k in blk])
    vol, fra = q["nf"] >= T_, q["frac"] > f_
    bad = np.flatnonzero((vol | fra) & ~q["atk"])
    fa[s] = [(start[i], start[i] + s) for i in bad]
    if s == 0.5:
        print(f"0.5 s: {len(bad)} false alarms (not listed)")
        continue
    for i in bad:
        print(f"{s:>4g} s: [{start[i]:8.2f}, {start[i] + s:8.2f}) s, block {blk[i]}, n {q['n'][i]}, n_f {q['nf'][i]}, "
              f"fraction {q['frac'][i]:.4f}, rule {'volume' if vol[i] else ''}{' fraction' if fra[i] else ''} "
              f"(T={T_[i]}, f={f_[i]:.4f})")
a1 = fa[1.0][0]
for s in (2.0, 5.0, 10.0):
    print(f"{s:g}-s false alarm contains the 1-s one: {all(lo <= a1[0] and a1[1] <= hi for lo, hi in fa[s])}")
