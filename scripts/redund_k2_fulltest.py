#!/usr/bin/env python3
"""Re-avaliacao do detector k>=2 REDUNDANTE (14 nos, 2/ataque) no TESTE COMPLETO,
sob encolhimento 14->3 (GL retem a uniao dos 14; FL re-agrega os presentes).
Objetivo: alinhar o LOG a tabela+figura (95,98 no teste completo, nao 96,07@200k).
Saida: results/redund_k2_fulltest.csv
Rodar: ~/venv-ereno314/bin/python scripts/redund_k2_fulltest.py
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
N, SEED = 14, 42
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 2}


def main():
    print("[full] particionando N=14 (attack -> 2/ataque)...")
    partitions, _, _ = load_and_partition(f"{DATASET}.csv", FEATURES, N, seed=SEED,
                                          partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    fl = []
    for X_c, y_c in partitions:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        yy = (yct != nc).astype(int)
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = dict(XGB_PARAMS)
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10, verbose_eval=False))
    print(f"[full] {len(fl)} especialistas (14 nos, 2/ataque)")

    print("[full] carregando teste COMPLETO (sem subamostra)...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    print(f"[full] teste completo: {len(yte_m):,} amostras ({n_atk:,} ataque / {n_ben:,} benigno)")
    dte = xgb.DMatrix(Xte)
    V = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in fl])   # 14 cols

    def stats(pred):
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100*vp/n_atk; fpr = 100*fp/n_ben
        prec = 100*vp/max(vp+fp, 1); f1 = 2*prec*rec/max(prec+rec, 1e-9)
        return f1, rec, prec, fpr, fp

    rows = []
    for n_active in range(14, 2, -1):                 # 14 -> 3
        S = list(range(n_active))
        v_fl = V[:, S].sum(1)                          # FL re-agrega presentes
        v_gl = V.sum(1)                                # GL retem os 14
        for k in (1, 2):
            f1f, rf, pcf, fprf, fpf = stats(v_fl >= k)
            f1g, rg, pcg, fprg, fpg = stats(v_gl >= k)
            rows.append((n_active, k, n_active, len(fl),
                         f1f, rf, pcf, fprf, fpf, f1g, rg, pcg, fprg, fpg))
        r2 = [r for r in rows if r[0] == n_active and r[1] == 2][0]
        print(f"[full] N={n_active:2d}  GL k>=2 (retido)= F1 {r2[9]:.2f} / rec {r2[10]:.2f} / "
              f"FPR {r2[12]:.3f} / #FP {r2[13]}")

    os.makedirs("results", exist_ok=True)
    with open("results/redund_k2_fulltest.csv", "w") as fh:
        fh.write("N,k,FL_boosters,GL_boosters,FL_F1,FL_Recall,FL_Prec,FL_FPR,FL_FP,"
                 "GL_F1,GL_Recall,GL_Prec,GL_FPR,GL_FP\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print("\n[full] CSV: results/redund_k2_fulltest.csv")


if __name__ == "__main__":
    main()
