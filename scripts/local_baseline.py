#!/usr/bin/env python3
"""Local baseline: what each node detects alone, with its own specialist only.

Third reference level for Table 7 (N = 10, attack-specialist partition, GRASP-24, seed 42,
full ERENO test), between which the shared union is read:
  local      each node (a substation with local, partial knowledge) uses only the specialist
             it trained;
  shared     the union of the ten specialists, OR fusion (FL = GL), no data pooled;
  pooled     the centralized model of Table 7, which needs the data of all nodes in one place.
The union row must reproduce the headline of reproduce.py (F1 95.72, 17,881 FP).

Reuses the training/scoring of scripts/k_fusion_audit.py (lean ARFF reader) with N = 10.
  python scripts/local_baseline.py
Out: results/local_baseline.txt; models and probabilities under results/.cache/
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import k_fusion_audit as kf

kf.N = 10
kf.MODELS = "results/.cache/local_baseline_models"
kf.CACHE = "results/.cache/ereno_probs_n10_local.npz"


def metrics(pred, atk):
    tp = int((pred & atk).sum()); fp = int((pred & ~atk).sum())
    rec = tp / atk.sum(); fpr = fp / (~atk).sum(); prec = tp / max(tp + fp, 1)
    return 100 * 2 * prec * rec / max(prec + rec, 1e-12), 100 * rec, 100 * prec, 100 * fpr, fp


def main():
    if not os.path.exists(kf.CACHE):
        kf.phase_train()
        kf.phase_score()
    z = np.load(kf.CACHE)
    P, y, nc, spec, cv = z["P"], z["y"], int(z["nc"]), z["spec"], list(z["cv"])
    atk = y != nc
    V = P >= 0.5
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"ERENO full test: {len(y):,} samples, {int(atk.sum()):,} attack, {int((~atk).sum()):,} benign")
        print(f"N = {kf.N} nodes, one specialist each, threshold 0.5\n")
        print("node | attack of its specialist | F1 | recall (all attacks) | recall (own attack) | precision | FPR % | #FP")
        rows = []
        for i in range(kf.N):
            f1, rec, prec, fpr, fp = metrics(V[:, i], atk)
            own = 100 * V[y == spec[i], i].mean()
            rows.append((f1, rec, prec, fpr, fp))
            print(f"  {i:>2} | {cv[spec[i]]:<24} | {f1:6.2f} | {rec:6.2f} | {own:6.2f} | {prec:6.2f} | {fpr:.3f} | {fp:,}")
        R = np.array(rows)
        print(f"\nlocal (one node alone): F1 {R[:, 0].min():.2f}-{R[:, 0].max():.2f} (median {np.median(R[:, 0]):.2f}); "
              f"recall {R[:, 1].min():.2f}-{R[:, 1].max():.2f} (median {np.median(R[:, 1]):.2f}); "
              f"FPR {R[:, 3].min():.3f}-{R[:, 3].max():.3f}%")
        f1, rec, prec, fpr, fp = metrics(V.sum(1) >= 1, atk)
        print(f"shared (union of the {kf.N}, OR): F1 {f1:.2f}; recall {rec:.2f}; precision {prec:.2f}; "
              f"FPR {fpr:.3f}%; #FP {fp:,}")
        print("pooled (centralized, Table 7): F1 99.23; recall 99.96; precision 98.52; FPR 0.110%; #FP 3,017")
    open("results/local_baseline.txt", "w").write(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
