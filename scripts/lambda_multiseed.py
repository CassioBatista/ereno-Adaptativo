#!/usr/bin/env python3
"""Estudo estatistico do lambda: cada conjunto combinado (por-ataque(lambda) U
global-15) avaliado em 5 seeds -> media +/- IC95. Responde "lambda=0.05 degrada
vs lambda=0?" com significancia (o gap vs a variacao de seed). Carrega train+test
UMA vez; para cada lambda filtra as features e re-particiona/treina por seed.
NAO roda GRASP. Saida: results/lambda_multiseed.csv
Rodar: ~/venv-ereno314/bin/python scripts/lambda_multiseed.py
"""
import math
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "scripts"))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import python.util as util
from fd.dataset import _attack_per_client
from lambda_combined_eval import build_combined, LAMBDAS

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
SEEDS = [42, 7, 13, 21, 99]
N, CAP = 10, 500000
T95 = 2.776  # t(0.975, df=4)


def main():
    sets = build_combined()
    print("[lms] conjuntos:", {lam: len(sets[lam][1]) for lam in LAMBDAS}, flush=True)
    print("[lms] carregando train (uma vez)...", flush=True)
    X_all, y_all, _ = util.load_arff(f"{DATASET}.csv")
    nc = util.normal_class
    print("[lms] carregando teste (uma vez)...", flush=True)
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    normal_idx = np.where(y_all == nc)[0]

    def train_pool(Xtr, seed):
        rng = np.random.default_rng(seed)
        drop = rng.permutation(normal_idx)[CAP:]
        keep = np.ones(len(y_all), dtype=bool); keep[drop] = False
        Xf, y = Xtr[keep], y_all[keep]
        pool = []
        for idx in _attack_per_client(Xf, y, N, rng):
            Xc, yc = Xf[idx], y[idx]
            Xct, _, yct, _ = train_test_split(Xc, yc, test_size=0.2, random_state=seed)
            yy = (yct != nc).astype(int)
            p, q = int(yy.sum()), int((yy == 0).sum())
            pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
                  "seed": seed, "nthread": 12}
            if p and q:
                pr["scale_pos_weight"] = q / p
            pool.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10,
                                  verbose_eval=False))
        return pool

    rows = []  # (lambda, n, seed, k, f1, fpr, fp)
    for lam in LAMBDAS:
        comb = sets[lam][1]
        cols = [i - 1 for i in sorted(comb)]
        Xtr = X_all[:, cols]
        dte = xgb.DMatrix(Xte_raw[:, cols])
        for seed in SEEDS:
            pool = train_pool(Xtr, seed)
            votes = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16)
                                     for b in pool]).sum(1)
            for k in (1, 2):
                pred = votes >= k
                vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
                rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
                prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                rows.append((lam, len(comb), seed, k, f1, fpr, fp))
            print(f"[lms] lambda={lam} ({len(comb)}f) seed={seed}: "
                  f"OR F1={rows[-2][4]:.2f} | k2 F1={rows[-1][4]:.2f}", flush=True)
        del Xtr, dte
        with open("results/lambda_multiseed.csv", "w") as fh:
            fh.write("lambda,n_features,seed,k,F1,FPR,FP\n")
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f},{r[5]:.4f},{r[6]}\n")

    print("\n===== LAMBDA x F1 (mean +/- IC95 over 5 seeds, full test) =====", flush=True)
    print(f"{'lambda':>7}{'#feat':>6}  |{'OR F1 (mean+/-CI)':>22}{'OR FPR':>16}  |"
          f"{'k2 F1 (mean+/-CI)':>22}")
    for lam in LAMBDAS:
        nf = sets[lam][1].__len__()
        def stat(k, col):
            a = np.array([r[col] for r in rows if r[0] == lam and r[3] == k])
            return a.mean(), T95 * a.std(ddof=1) / math.sqrt(len(a))
        orm, orc = stat(1, 4); ofm, ofc = stat(1, 5); k2m, k2c = stat(2, 4)
        print(f"{lam:>7}{nf:>6}  |   {orm:6.2f} +/- {orc:4.2f}      {ofm:6.3f} +/-{ofc:5.3f}  |"
              f"   {k2m:6.2f} +/- {k2c:4.2f}", flush=True)
    print("\n[lms] CSV -> results/lambda_multiseed.csv", flush=True)


if __name__ == "__main__":
    main()
