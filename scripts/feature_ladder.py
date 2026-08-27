#!/usr/bin/env python3
"""all-58 vs GRASP: ladder de conjuntos de features no teste cheio (seed 42).

Treina o ensemble de especialistas (N=10, particao por-ataque) com quatro
conjuntos de features e mede F1/recall/prec/FPR/#FP no teste do autor (2.9M),
para OR (k>=1) e corroboracao (k>=2). Responde ao revisor: all-58 vs GRASP,
com F1 x complexidade (numero de features).

Conjuntos:
  ALL58  = todas as 58 features numericas do ERENO (inclui tempo absoluto)
  RCL55  = ALL58 sem F1/F38/F39 (marcadores de tempo absoluto; vazam posicao)
  GRASP24= conjunto do paper (uniao penalizada lambda=0.05)
  GLOBAL8= selecao global pura do GRASP (all_in_one_ereno_train.json)

Rodar: ~/venv-ereno314/bin/python scripts/feature_ladder.py
Saida: results/feature_ladder.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import python.util as util
from fd.dataset import load_and_partition

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
N, SEED = 10, 42

ALL58   = list(range(1, 59))
RCL55   = [f for f in ALL58 if f not in (1, 38, 39)]
GRASP24 = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
GLOBAL8 = [40, 41, 42, 43, 44, 54, 56, 57]
SETS = [("ALL58", ALL58), ("RCL55", RCL55), ("GRASP24", GRASP24), ("GLOBAL8", GLOBAL8)]


def train_eval(features):
    parts, _, _ = load_and_partition(f"{DATASET}.csv", features, N, seed=SEED,
                                     partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    fl = []
    for X_c, y_c in parts:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        yy = (yct != nc).astype(int)
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 12}
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10,
                            verbose_eval=False))
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, features); del Xte_raw
    is_atk = (yte_m != nc); n_ben = int((~is_atk).sum()); n_atk = int(is_atk.sum())
    P = np.column_stack([b.predict(xgb.DMatrix(Xte)) for b in fl])
    votes = (P >= 0.5).astype(np.int16).sum(1)
    out = {}
    for k in (1, 2):
        pred = votes >= k
        tp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        fpr = 100 * fp / n_ben; rec = 100 * tp / n_atk
        prec = 100 * tp / max(tp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[k] = (f1, rec, prec, fpr, fp)
    return out


def main():
    print("[ladder] all-58 vs GRASP no teste cheio (seed 42, N=10)\n")
    rows = []
    for name, feats in SETS:
        print(f"[ladder] {name} ({len(feats)} features)...")
        r = train_eval(feats)
        rows.append((name, len(feats), r))
    print("\n===== FEATURE-SET LADDER (teste cheio, seed 42) =====")
    hdr = f"{'set':<8}{'#feat':>6}  | {'OR F1':>7}{'OR FPR':>8}{'OR #FP':>9}  | {'k2 F1':>7}{'k2 FPR':>8}{'k2 #FP':>8}"
    print(hdr); print("-" * len(hdr))
    for name, nf, r in rows:
        f1a, _, _, fpra, fpa = r[1]; f1b, _, _, fprb, fpb = r[2]
        print(f"{name:<8}{nf:>6}  | {f1a:7.2f}{fpra:8.3f}{fpa:9,d}  | "
              f"{f1b:7.2f}{fprb:8.3f}{fpb:8,d}")
    os.makedirs("results", exist_ok=True)
    with open("results/feature_ladder.csv", "w") as fh:
        fh.write("set,n_features,k,f1,recall,precision,fpr,n_fp\n")
        for name, nf, r in rows:
            for k in (1, 2):
                f1, rec, prec, fpr, fp = r[k]
                fh.write(f"{name},{nf},{k},{f1:.4f},{rec:.4f},{prec:.4f},{fpr:.4f},{fp}\n")
    print("\n[ladder] CSV -> results/feature_ladder.csv")


if __name__ == "__main__":
    main()
