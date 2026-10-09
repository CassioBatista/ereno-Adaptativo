#!/usr/bin/env python3
"""Fault containment of k >= 2 vs OR on ERENO (cached probabilities of k_fusion_audit.py).

Paper 1 states that under redundancy the case for k >= 2 is fault containment: no single
specialist can raise an alarm alone. This measures it. One booster of the 14 (N=14, two
per attack, tau = 0.5) is replaced by a faulty one, for every booster in turn:
  stuck-1        fires on every sample (crashed into "attack", corrupted model)
  noise p        fires at random on a fraction p of the samples
  stuck-0        never fires (silent / crashed into "normal")
and, for the limit of the claim, BOTH boosters of one attack stuck-1 (a fault shared by the
pair, e.g. poisoned training data of that class).

  python scripts/k_fusion_fault_containment.py
Out: results/k_fusion_fault_containment.txt
"""
import io
import os
from contextlib import redirect_stdout

import numpy as np

CACHE = "results/.cache/ereno_probs_n14_k2audit.npz"


def main():
    z = np.load(CACHE)
    P, y, nc, spec, cv = z["P"], z["y"], int(z["nc"]), z["spec"], list(z["cv"])
    atk = y != nc
    V = P >= 0.5
    classes = sorted(set(spec.tolist()))
    rng = np.random.default_rng(7)

    def m(votes, k):
        pred = votes >= k
        tp = int((pred & atk).sum()); fp = int((pred & ~atk).sum())
        rec = tp / atk.sum(); fpr = fp / (~atk).sum(); prec = tp / max(tp + fp, 1)
        return 100 * 2 * prec * rec / max(prec + rec, 1e-12), 100 * rec, 100 * fpr

    base = V.sum(1).astype(np.int16)
    print("ERENO full test, N=14 (2 specialists per attack), tau=0.5")
    for k, nm in ((1, "OR"), (2, "k>=2")):
        f1, r, fpr = m(base, k)
        print(f"  healthy {nm:<5} F1 {f1:6.2f}  recall {r:6.2f}  FPR {fpr:7.3f}%")

    def faulted(i, col):
        return base - V[:, i] + col

    def report(title, cols_by_booster):
        print(f"\n=== {title} (one booster faulty; min / mean / max over the 14) ===")
        for k, nm in ((1, "OR"), (2, "k>=2")):
            res = np.array([m(faulted(i, c), k) for i, c in cols_by_booster])
            print(f"  {nm:<5} F1 {res[:, 0].min():6.2f} / {res[:, 0].mean():6.2f} / {res[:, 0].max():6.2f}   "
                  f"recall {res[:, 1].min():6.2f} / {res[:, 1].mean():6.2f} / {res[:, 1].max():6.2f}   "
                  f"FPR {res[:, 2].min():7.3f} / {res[:, 2].mean():7.3f} / {res[:, 2].max():7.3f}%")
        return

    n = len(y)
    report("stuck-1: fires on every sample", [(i, np.ones(n, np.int16)) for i in range(14)])
    for p in (0.10, 0.01, 0.001):
        report(f"noise: fires at random on {100 * p:g}% of samples",
               [(i, (rng.random(n) < p).astype(np.int16)) for i in range(14)])
    report("stuck-0: never fires (silent)", [(i, np.zeros(n, np.int16)) for i in range(14)])

    # per-attack view of stuck-0 under k>=2: which class is vetoed
    print("\n  stuck-0 under k>=2, recall of the attack whose booster went silent:")
    for i in range(14):
        c = int(spec[i])
        votes = faulted(i, np.zeros(n, np.int16))
        print(f"    booster {i:2d} ({cv[c][:22]:<22}) recall of its attack "
              f"{100 * (votes >= 2)[y == c].mean():6.2f}%  (healthy {100 * (base >= 2)[y == c].mean():6.2f}%; "
              f"OR {100 * (votes >= 1)[y == c].mean():6.2f}%)")

    print("\n=== limit: BOTH boosters of one attack stuck-1 (fault shared by the pair) ===")
    for c in classes:
        idx = np.where(spec == c)[0]
        votes = base - V[:, idx].sum(1) + 2
        f1, r, fpr = m(votes, 2)
        print(f"  {cv[c][:22]:<22} k>=2: FPR {fpr:7.3f}%  (F1 {f1:6.2f})")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/k_fusion_fault_containment.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
