#!/usr/bin/env python3
"""Mostra que lambda=0.05 (24 features) nao degrada vs lambda=0 (35 features).

NAO roda GRASP. Reconstroi, dos dados reais do log per-ataque, o conjunto
COMBINADO (superset per-ataque no lambda dado UNIAO nucleo global-15) para cada
lambda, e apenas TREINA os 10 especialistas + AVALIA no teste cheio (fusao OR e
k>=2). Mesma logica do feature_ladder (segura). Assim a curva lambda x F1 usa o
MESMO objeto que o conjunto implantado (a uniao), nao a selecao global-only.

Rodar: ~/venv-ereno314/bin/python scripts/lambda_combined_eval.py
Saida: results/lambda_combined_eval.csv
"""
import ast
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import python.util as util
from fd.dataset import load_and_partition

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
N, SEED = 10, 42
LOG = "/home/cassi/grasp_por_ataque.log"
GLOBAL15 = [5, 7, 8, 11, 19, 31, 33, 40, 41, 42, 44, 45, 50, 57, 58]
LAMBDAS = [0.0, 0.05, 0.1, 0.2]


def build_combined():
    """Retorna {lambda: (superset, combined)} do log per-ataque real."""
    per = defaultdict(dict)
    cur = None
    with open(LOG, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "grasp-atk] ATAQUE" in line:
                cur = line.split("ATAQUE", 1)[1].split("(", 1)[0].strip()
            elif line.startswith("EV;") and cur:
                p = line.rstrip("\n").split(";", 3)
                if len(p) < 4:
                    continue
                try:
                    f1 = float(p[2]); feats = frozenset(ast.literal_eval(p[3]))
                except (ValueError, SyntaxError):
                    continue
                if feats and (feats not in per[cur] or f1 > per[cur][feats]):
                    per[cur][feats] = f1
    fronts = {}
    for a, subs in per.items():
        by_k = {}
        for feats, f1 in subs.items():
            k = len(feats)
            if k not in by_k or f1 > by_k[k][0]:
                by_k[k] = (f1, set(feats))
        fronts[a] = by_k
    out = {}
    for lam in LAMBDAS:
        sup = set()
        for by_k in fronts.values():
            kstar = max(by_k, key=lambda k: by_k[k][0] - lam * k)
            sup |= by_k[kstar][1]
        combined = sorted(set(GLOBAL15) | sup)
        out[lam] = (sorted(sup), combined)
    return out


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
        if p and q:
            pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10,
                            verbose_eval=False))
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, features); del Xte_raw
    is_atk = (yte_m != nc); n_ben = int((~is_atk).sum()); n_atk = int(is_atk.sum())
    P = np.column_stack([b.predict(xgb.DMatrix(Xte)) for b in fl])
    votes = (P >= 0.5).astype(np.int16).sum(1)
    r = {}
    for k in (1, 2):
        pred = votes >= k
        tp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        fpr = 100 * fp / n_ben; rec = 100 * tp / n_atk
        prec = 100 * tp / max(tp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        r[k] = (f1, rec, prec, fpr, fp)
    return r


def main():
    sets = build_combined()
    print("[lc] conjuntos reconstruidos (superset per-ataque U global-15):", flush=True)
    for lam in LAMBDAS:
        sup, comb = sets[lam]
        print(f"  lambda={lam}: superset={len(sup)} combined={len(comb)}", flush=True)
    rows = []
    for lam in LAMBDAS:
        sup, comb = sets[lam]
        print(f"\n[lc] lambda={lam} -> {len(comb)} features, treinando+avaliando...", flush=True)
        r = train_eval(comb)
        rows.append((lam, len(sup), len(comb), r))
        f1a, _, _, fpra, fpa = r[1]; f1b, _, _, fprb, fpb = r[2]
        print(f"[lc] lambda={lam}: OR F1={f1a:.2f} FPR={fpra:.3f} #FP={fpa:,} | "
              f"k2 F1={f1b:.2f} #FP={fpb:,}", flush=True)
        with open("results/lambda_combined_eval.csv", "w") as fh:
            fh.write("lambda,n_superset,n_combined,or_f1,or_fpr,or_fp,k2_f1,k2_fpr,k2_fp\n")
            for (lm, ns, nc_, rr) in rows:
                a = rr[1]; b = rr[2]
                fh.write(f"{lm},{ns},{nc_},{a[0]:.4f},{a[3]:.4f},{a[4]},"
                         f"{b[0]:.4f},{b[3]:.4f},{b[4]}\n")

    print("\n===== LAMBDA x COMBINED (full test, seed 42) =====", flush=True)
    print(f"{'lambda':>7}{'#feat':>6}  |{'OR F1':>7}{'OR FPR':>8}{'OR #FP':>9}  |{'k2 F1':>7}{'k2 #FP':>8}")
    for (lm, ns, nc_, rr) in rows:
        a = rr[1]; b = rr[2]
        print(f"{lm:>7}{nc_:>6}  |{a[0]:>7.2f}{a[3]:>8.3f}{a[4]:>9,}  |{b[0]:>7.2f}{b[4]:>8,}", flush=True)
    print("\n[lc] CSV -> results/lambda_combined_eval.csv", flush=True)


if __name__ == "__main__":
    main()
