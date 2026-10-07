#!/usr/bin/env python3
"""Why is the CICIoT2023 benign FPR 23 % (cross_domain_results.py)? Same 14 specialists
and k>=2 GL fusion, varying (a) how much benign each specialist sees and (b) capacity.
Benign is 2.4 % of the release: with a 1/14 share each specialist sees ~2,800 benign rows.
Out: results/ciciot2023_benign_sensitivity.txt
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


def run(Xtr, ctr, Xte, cte, benign_mode, rounds, depth, attack_cap=None):
    rng = np.random.default_rng(42)
    ben = np.where(ctr == "Benign")[0]
    shares = np.array_split(rng.permutation(ben), 14)
    V = []
    for i in range(14):
        c = CATS[i % 7]
        idx = rng.permutation(np.where(ctr == c)[0])
        pos = np.array_split(idx, 2)[i // 7]
        if attack_cap:
            pos = pos[:attack_cap]
        neg = shares[i] if benign_mode == "share" else ben
        rows = np.r_[pos, neg]
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        p = {"max_depth": depth, "eta": 0.1, "objective": "binary:logistic", "seed": 42,
             "nthread": 4, "scale_pos_weight": len(neg) / len(pos)}
        b = xgb.train(p, xgb.DMatrix(Xtr[rows], label=y), num_boost_round=rounds)
        V.append((b.predict(xgb.DMatrix(Xte)) >= 0.5).astype(np.int8))
    V = np.column_stack(V)
    atk = cte != "Benign"
    out = {}
    for k in (1, 2):
        f = V.sum(1) >= k
        rec, fpr = f[atk].mean(), f[~atk].mean()
        prec = rec / (rec + 13.75 * fpr) if rec else 0
        out[k] = (100 * rec, 100 * fpr, 100 * 2 * prec * rec / (prec + rec))
    single = [100 * V[~atk, i].mean() for i in range(14)]
    return out, single


def main():
    parts = sorted(glob.glob(D + "part-*.csv"))
    tr = pd.concat([pd.read_csv(p) for p in parts[:7]], ignore_index=True)
    te = pd.concat([pd.read_csv(p) for p in parts[7:]], ignore_index=True)
    lab = tr.columns[-1]
    feats = [c for c in tr.columns if c not in (lab, "IAT")]
    Xtr, Xte = tr[feats].to_numpy(np.float32), te[feats].to_numpy(np.float32)
    ctr, cte = tr[lab].map(category).to_numpy(), te[lab].map(category).to_numpy()
    print(f"benign train={int((ctr == 'Benign').sum()):,} test={int((cte == 'Benign').sum()):,}")
    print(f"{'variant':<52} {'k':>2} {'recall':>7} {'FPR':>7} {'F1@13.75':>9}")
    variants = [("benign 1/14 share, 10 rounds, depth 4 (protocol)", "share", 10, 4, None),
                ("benign 1/14 share, 100 rounds, depth 6", "share", 100, 6, None),
                ("all benign per specialist, 10 rounds, depth 4", "all", 10, 4, None),
                ("all benign per specialist, 100 rounds, depth 6", "all", 100, 6, None),
                ("all benign, attack capped at 20k, 100 rounds, depth 6", "all", 100, 6, 20000)]
    for name, bm, r, dpt, cap in variants:
        out, single = run(Xtr, ctr, Xte, cte, bm, r, dpt, cap)
        for k, (rec, fpr, f1) in out.items():
            print(f"{name:<52} {k:>2} {rec:7.2f} {fpr:7.2f} {f1:9.2f}")
        print(f"{'':<52}    per-specialist benign FPR: "
              + " ".join(f"{CATS[i % 7][:4]}{v:.1f}" for i, v in enumerate(single)))


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/ciciot2023_benign_sensitivity.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
