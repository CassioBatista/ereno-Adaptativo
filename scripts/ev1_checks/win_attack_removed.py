"""r = 0: every attack sample removed; the windows that held attacks keep only their benign
samples. How many of them still alarm (rule and blocked-5 of section 6, calibrated on the
windows that were attack-free)? Measures alarms driven by benign-labelled samples around the
attacks. Read-only."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo/scripts"))
import ev1_windows as E
for d, size in (("ereno", 1.0), ("cicids", 10.0), ("morris4", 100.0)):
    z = np.load(os.path.join(E.ROOT, f"results/.cache/ev1_step1_{d}_s42.npz"), allow_pickle=True)
    y, nc = z["y"].astype(np.int64), int(z["nc"])
    t = E.times(E.DOMS[d]); o = np.argsort(t, kind="stable")
    t, fired, y = t[o], z["Ps"][o] >= 0.5, y[o]
    w = np.floor((t - t[0]) / size).astype(np.int64)
    W = E.window_table(w, fired, y, nc, [])
    was_atk = W["atk"].copy()
    b = y == nc                                       # drop attack samples
    wb = w[b]
    Wb = E.window_table(wb, fired[b], y[b], nc, [])
    ids_all = np.unique(w); ids_b = np.unique(wb)
    was = was_atk[np.searchsorted(ids_all, ids_b)]   # window held attacks before removal
    Wb["atk"] = was                                   # keep them out of calibration, as before
    _, full, _, scored = E.evaluate(Wb)
    m = was & scored
    print(f"{E.DOMS[d]['name']}: windows that held attacks, attack samples removed: {int(m.sum())}; "
          f"still alarmed {int(full[m].sum())} ({100 * full[m].mean():.1f} %)")
