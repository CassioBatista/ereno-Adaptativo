#!/usr/bin/env python3
"""Ordem de falha importa: FL OR-recall (k=1) sob duas ordens de queda, N=10->3.
 - WORST: dropa os nos de ataque UNICO primeiro (perde cobertura de ataques rapido).
 - BEST : dropa os REDUNDANTES primeiro (mantem 1 sensor/ataque -> cobertura ate N=7).
Mostra que o F1/recall plano depende de QUAIS nos caem, nao so de quantos.
Saida: results/reducao_ordem.csv
Rodar: ~/venv-ereno314/bin/python scripts/reducao_ordem.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import python.util as util
from fd.dataset import load_and_partition

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
N, SEED = 10, 42
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 2}

# mapa no->ataque (N=10 attack partitioner): 0-1 rand, 2-3 inj, 4-5 high (redundantes);
# 6 inverse, 7 poisoned, 8 masq_normal, 9 masq_fault (unicos)
UNIQUE = [6, 7, 8, 9]
REDUND = [0, 1, 2, 3, 4, 5]
# BEST: sobrevive priorizando 1/ataque -> unicos + 1 de cada redundante + extras
BEST_KEEP = [6, 7, 8, 9, 0, 2, 4, 1, 3, 5]
# WORST: sobrevive [0..n-1] -> dropa 9,8,7,6 (unicos) primeiro
WORST_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def main():
    print("[ordem] particionando N=10 (attack)...")
    partitions, _, _ = load_and_partition(f"{DATASET}.csv", FEATURES, N, seed=SEED,
                                          partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    print("[ordem] treinando 10 especialistas...")
    fl = []
    for X_c, y_c in partitions:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        yy = (yct != nc).astype(int)
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = dict(XGB_PARAMS)
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10, verbose_eval=False))

    print("[ordem] carregando teste completo...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    dte = xgb.DMatrix(Xte)
    V = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in fl])  # 10 votos

    def stats(surv, k):
        v = V[:, surv].sum(1)
        pred = v >= k
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, fpr, fp

    rows = []
    for order, keep in (("worst", WORST_KEEP), ("best", BEST_KEEP)):
        for n_active in range(10, 2, -1):
            surv = keep[:n_active]
            for k in (1, 2):
                f1, rec, fpr, fp = stats(surv, k)
                rows.append((order, n_active, k, f1, rec, fpr, fp))

    os.makedirs("results", exist_ok=True)
    with open("results/reducao_ordem.csv", "w") as fh:
        fh.write("order,N,k,F1,Recall,FPR,FP\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    for k in (1, 2):
        print(f"\n=== FL OR/k>={k}: recall por ordem de falha ===")
        print(f"{'N':>2} | {'worst':>8} {'best':>8}")
        for n_active in range(10, 2, -1):
            w = next(r for r in rows if r[0] == "worst" and r[1] == n_active and r[2] == k)
            b = next(r for r in rows if r[0] == "best" and r[1] == n_active and r[2] == k)
            print(f"{n_active:>2} | {w[4]:>8.2f} {b[4]:>8.2f}")
    print("\n[ordem] CSV: results/reducao_ordem.csv")


if __name__ == "__main__":
    main()
