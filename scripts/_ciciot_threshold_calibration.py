#!/usr/bin/env python3
"""Can CICIoT2023 reach an acceptable false-positive rate? Per-specialist thresholds
calibrated on held-out benign (never used for training) to a per-specialist FPR budget,
instead of the fixed 0.5. Training benign is split 70/30: 70 % trains, 30 % calibrates.
Variants: benign share vs all benign per specialist; protocol capacity vs 100 rounds/depth 6.
Out: results/ciciot2023_threshold_calibration.txt
"""
import glob
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_ciciot2023 import category  # noqa: E402

D = os.path.expanduser("~/datasets/ciciot2023/")
CATS = ["DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]
BUDGETS = [None, 0.01, 0.005, 0.001]          # per-specialist benign FPR on calibration


def main():
    parts = sorted(glob.glob(D + "part-*.csv"))
    tr = pd.concat([pd.read_csv(p) for p in parts[:7]], ignore_index=True)
    te = pd.concat([pd.read_csv(p) for p in parts[7:]], ignore_index=True)
    lab = tr.columns[-1]
    feats = [c for c in tr.columns if c not in (lab, "IAT")]
    Xtr, Xte = tr[feats].to_numpy(np.float32), te[feats].to_numpy(np.float32)
    ctr, cte = tr[lab].map(category).to_numpy(), te[lab].map(category).to_numpy()
    rng = np.random.default_rng(42)
    ben = rng.permutation(np.where(ctr == "Benign")[0])
    cal = ben[: int(0.3 * len(ben))]
    ben_tr = ben[int(0.3 * len(ben)):]
    print(f"benign: train {len(ben_tr):,}, calibration {len(cal):,}, test {int((cte == 'Benign').sum()):,}")
    atk = cte != "Benign"
    for bmode, rounds, depth in (("share", 10, 4), ("all", 10, 4), ("all", 100, 6)):
        shares = np.array_split(ben_tr, 14)
        P_cal, P_te, names = [], [], []
        for i in range(14):
            c = CATS[i % 7]
            idx = np.random.default_rng(42 + i).permutation(np.where(ctr == c)[0])
            pos = np.array_split(idx, 2)[i // 7]
            neg = shares[i] if bmode == "share" else ben_tr
            rows = np.r_[pos, neg]
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            p = {"max_depth": depth, "eta": 0.1, "objective": "binary:logistic", "seed": 42,
                 "nthread": 4, "scale_pos_weight": len(neg) / len(pos)}
            b = xgb.train(p, xgb.DMatrix(Xtr[rows], label=y), num_boost_round=rounds)
            P_cal.append(b.predict(xgb.DMatrix(Xtr[cal])))
            P_te.append(b.predict(xgb.DMatrix(Xte)))
            names.append(c)
        print(f"\n=== benign {bmode}, {rounds} rounds, depth {depth}")
        print(f"{'budget':>8} {'k':>2} {'recall':>7} {'FPR':>7} {'F1@13.75':>9}  per-category recall (k>=2)")
        for bud in BUDGETS:
            V = []
            for pc, pt in zip(P_cal, P_te):
                thr = 0.5 if bud is None else max(0.5, float(np.quantile(pc, 1 - bud)))
                V.append((pt >= thr).astype(np.int8))
            V = np.column_stack(V)
            for k in (1, 2):
                f = V.sum(1) >= k
                rec, fpr = f[atk].mean(), f[~atk].mean()
                prec = rec / (rec + 13.75 * fpr) if rec else 0
                f1 = 2 * prec * rec / max(prec + rec, 1e-12)
                pcr = ""
                if k == 2:
                    pcr = " ".join(f"{c[:5]}={100 * f[cte == c].mean():.1f}" for c in CATS)
                tag = "0.5" if bud is None else f"{100 * bud:.1f}%"
                print(f"{tag:>8} {k:>2} {100 * rec:7.2f} {100 * fpr:7.3f} {100 * f1:9.2f}  {pcr}")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/ciciot2023_threshold_calibration.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
