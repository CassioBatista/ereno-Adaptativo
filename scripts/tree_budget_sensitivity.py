#!/usr/bin/env python3
"""Justifica num_boost_round=10: sensibilidade ao orcamento de arvores.

Distribuido (specialist N=10, GRASP-24): num_boost_round em {5,10,20} -> OR(k>=1)
e k>=2 no teste cheio. Centralizado (1 XGBoost sobre o pool benign-capped): arvores
em {50,100,200}. Mostra (i) o alinhamento 10 boosters x 10 arvores = 100 = centralizado
100, e (ii) que a deteccao distribuida e estavel no nº de arvores (10 nao e cherry-pick).
Carrega train+test UMA vez. Saida: results/tree_budget_sensitivity.csv
Rodar: ~/venv-ereno314/bin/python scripts/tree_budget_sensitivity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import python.util as util
from fd.dataset import _attack_per_client

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
SEED, CAP, N = 42, 500000, 10
DIST_NBR = [5, 10, 20]
CENT_TREES = [50, 100, 200]


def metrics(pred, is_atk, is_ben, n_atk, n_ben):
    tp = int((pred & is_atk).sum()); fp = int((pred & is_ben).sum())
    rec = 100 * tp / max(n_atk, 1); prec = 100 * tp / max(tp + fp, 1)
    fpr = 100 * fp / max(n_ben, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return f1, rec, fpr, fp


def main():
    cols = [i - 1 for i in sorted(FEATURES)]
    print("[tb] load train...", flush=True)
    X_all, y_all, _ = util.load_arff(f"{DATASET}.csv"); nc = util.normal_class
    Xf_full = X_all[:, cols]; del X_all
    print("[tb] load test...", flush=True)
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv"); util.normal_class = nc
    dte = xgb.DMatrix(Xte_raw[:, cols]); del Xte_raw
    is_atk = (yte_m != nc); is_ben = ~is_atk
    n_atk, n_ben = int(is_atk.sum()), int(is_ben.sum())

    # pool benign-capped (uma vez): drop normal alem do CAP
    rng = np.random.default_rng(SEED)
    normal_idx = np.where(y_all == nc)[0]
    drop = rng.permutation(normal_idx)[CAP:]
    keep = np.ones(len(y_all), dtype=bool); keep[drop] = False
    Xf, y = Xf_full[keep], y_all[keep]
    splits = _attack_per_client(Xf, y, N, rng)  # same rng as the drop (matches deployed)

    rows = []
    print("\n===== DISTRIBUTED (specialist N=10) — num_boost_round sweep =====", flush=True)
    for nbr in DIST_NBR:
        pool = []
        for idx in splits:
            Xc, yc = Xf[idx], y[idx]
            Xct, _, yct, _ = train_test_split(Xc, yc, test_size=0.2, random_state=SEED)
            yy = (yct != nc).astype(int)
            p, q = int(yy.sum()), int((yy == 0).sum())
            pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
                  "seed": SEED, "nthread": 12}
            if p and q: pr["scale_pos_weight"] = q / p
            pool.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=nbr,
                                  verbose_eval=False))
        votes = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in pool]).sum(1)
        for k in (1, 2):
            f1, rec, fpr, fp = metrics(votes >= k, is_atk, is_ben, n_atk, n_ben)
            rows.append(("distributed", f"nbr={nbr},k>={k}", nbr * N, f1, rec, fpr, fp))
            print(f"  nbr={nbr:>2} (total {nbr*N} trees)  k>={k}:  F1={f1:6.2f} recall={rec:6.2f} "
                  f"FPR={fpr:6.3f} #FP={fp:,}", flush=True)

    print("\n===== CENTRALIZED (single XGBoost over the pool) — trees sweep =====", flush=True)
    yb = (y != nc).astype(int)
    p, q = int(yb.sum()), int((yb == 0).sum())
    dpool = xgb.DMatrix(Xf, label=yb)
    for nt in CENT_TREES:
        pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 12}
        if p and q: pr["scale_pos_weight"] = q / p
        bst = xgb.train(pr, dpool, num_boost_round=nt, verbose_eval=False)
        f1, rec, fpr, fp = metrics(bst.predict(dte) >= 0.5, is_atk, is_ben, n_atk, n_ben)
        rows.append(("centralized", f"trees={nt}", nt, f1, rec, fpr, fp))
        print(f"  trees={nt:>3}:  F1={f1:6.2f} recall={rec:6.2f} FPR={fpr:6.3f} #FP={fp:,}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/tree_budget_sensitivity.csv", "w") as fh:
        fh.write("arch,config,total_trees,F1,recall,FPR,FP\n")
        for a, c, t, f1, rec, fpr, fp in rows:
            fh.write(f'{a},"{c}",{t},{f1:.4f},{rec:.4f},{fpr:.4f},{fp}\n')
    print("\n[tb] CSV -> results/tree_budget_sensitivity.csv", flush=True)


if __name__ == "__main__":
    main()
