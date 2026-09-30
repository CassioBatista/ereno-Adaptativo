#!/usr/bin/env python3
"""Sweep de lambda para JUSTIFICAR a penalidade — selecao GLOBAL (multiclasse,
todos os ataques juntos), NAO por-ataque. Para cada lambda em LAMBDAS:
  (1) roda o GRASP global penalizado (F1_cv - lambda*k) em subamostra 50k;
  (2) treina os 10 especialistas (attack partition, seed 42) sobre as features
      selecionadas e avalia a fusao OR (k>=1) e k>=2 no teste cheio.
Saida: results/lambda_sweep_global.csv (parcial, reescrita a cada lambda) +
features/lambda_global_L{lam}.json (metadados GRASP com elapsed_s).

Rodar: ~/venv-ereno314/bin/python scripts/lambda_sweep_global.py
"""
import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

import python.config as config
import python.util as util
import main_grasp
from fd.dataset import load_and_partition

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
N, SEED = 10, 42
SAMPLE = 50000
LAMBDAS = [0.0, 0.05, 0.1, 0.2]
CSV = "results/lambda_sweep_global.csv"


def run_grasp(lam):
    out = f"features/lambda_global_L{lam}.json"
    config.FEATURE_PENALTY = lam
    print(f"\n========== GRASP GLOBAL lambda={lam} (sample {SAMPLE}) ==========",
          flush=True)
    t0 = time.time()
    main_grasp.main(["GR-G-VND", "6", DATASET, "--sample", str(SAMPLE),
                     "--no-improvement", "15", "--out", out])
    dt = time.time() - t0
    d = json.load(open(out))
    return d["features"], d.get("f1_grasp_cv"), d.get("elapsed_s", round(dt, 1))


def eval_fusion(features):
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
    out = {}
    for k in (1, 2):
        pred = votes >= k
        tp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        fpr = 100 * fp / n_ben; rec = 100 * tp / n_atk
        prec = 100 * tp / max(tp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[k] = (f1, rec, prec, fpr, fp)
    return out


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    hdr = ("lambda,n_features,f1_cv,grasp_elapsed_s,"
           "or_f1,or_fpr,or_fp,k2_f1,k2_fpr,k2_fp,features\n")
    for lam in LAMBDAS:
        feats, f1cv, elapsed = run_grasp(lam)
        print(f"[sweep] lambda={lam}: {len(feats)} features, f1_cv={f1cv}, "
              f"GRASP {elapsed}s -> evaluating fusion...", flush=True)
        r = eval_fusion(feats)
        f1a, _, _, fpra, fpa = r[1]; f1b, _, _, fprb, fpb = r[2]
        rows.append((lam, len(feats), f1cv, elapsed, f1a, fpra, fpa, f1b, fprb, fpb, feats))
        # grava parcial a cada lambda
        with open(CSV, "w") as fh:
            fh.write(hdr)
            for (lm, nf, fc, el, oa, ofpr, ofp, kb, kfpr, kfp, ft) in rows:
                fh.write(f'{lm},{nf},{fc},{el},{oa:.4f},{ofpr:.4f},{ofp},'
                         f'{kb:.4f},{kfpr:.4f},{kfp},"{ft}"\n')
        print(f"[sweep] lambda={lam}: OR F1={f1a:.2f} FPR={fpra:.3f} #FP={fpa:,} | "
              f"k2 F1={f1b:.2f} #FP={fpb:,}   (CSV atualizado)", flush=True)

    print("\n===== LAMBDA SWEEP (global selection, 50k, full test) =====", flush=True)
    print(f"{'lambda':>7}{'#feat':>6}{'f1_cv':>8}{'grasp_h':>8}  |{'OR F1':>7}"
          f"{'OR FPR':>8}{'OR #FP':>9}  |{'k2 F1':>7}{'k2 #FP':>8}")
    for (lm, nf, fc, el, oa, ofpr, ofp, kb, kfpr, kfp, ft) in rows:
        fcs = f"{fc:.2f}" if isinstance(fc, (int, float)) else str(fc)
        print(f"{lm:>7}{nf:>6}{fcs:>8}{el/3600:>8.2f}  |{oa:>7.2f}{ofpr:>8.3f}"
              f"{ofp:>9,}  |{kb:>7.2f}{kfp:>8,}", flush=True)
    print(f"\n[sweep] CSV -> {CSV}", flush=True)


if __name__ == "__main__":
    main()
