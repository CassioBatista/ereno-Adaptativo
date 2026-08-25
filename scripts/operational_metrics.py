#!/usr/bin/env python3
"""Metricas operacionais (seed 42, specialist, teste completo):
per-attack recall (dados por classe), matriz de confusao, MCC, macro-recall,
FP/milhao, PR-AUC (OR = max-prob), para OR(k=1) e k>=2. Threshold operacional tau=0.5.
Saida: results/operational_metrics.csv
Rodar: ~/venv-ereno314/bin/python scripts/operational_metrics.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, average_precision_score
import python.util as util
from fd.dataset import load_and_partition

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
N, SEED = 10, 42


def main():
    names = None
    with open(f"{DATASET}.csv", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if "@class@" in s and "{" in s:
                names = [x.strip() for x in s[s.index("{") + 1:s.index("}")].split(",")]
                break

    print("[op] particionando + treinando (seed 42, specialist)...")
    parts, _, _ = load_and_partition(f"{DATASET}.csv", FEATURES, N, seed=SEED,
                                     partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    fl = []
    for X_c, y_c in parts:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        yy = (yct != nc).astype(int)
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 2}
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10, verbose_eval=False))

    print("[op] teste completo...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_ben = int((~is_atk).sum()); n_atk = int(is_atk.sum())
    dte = xgb.DMatrix(Xte)
    P = np.column_stack([b.predict(dte) for b in fl])        # probabilidades por booster
    votes = (P >= 0.5).astype(np.int16).sum(1)
    maxprob = P.max(1)                                       # score OR (union margin)

    rows = []
    for k in (1, 2):
        pred = (votes >= k)
        tp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        fn = int((~pred & is_atk).sum()); tn = int((~pred & ~is_atk).sum())
        mcc = matthews_corrcoef(is_atk.astype(int), pred.astype(int))
        fpr = 100 * fp / n_ben; rec = 100 * tp / n_atk
        prec = 100 * tp / max(tp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        fpm = fp / n_ben * 1e6
        # per-attack recall (dados por classe)
        per = {}
        for c in sorted(int(v) for v in np.unique(yte_m) if v != nc):
            m = (yte_m == c)
            per[names[c] if names else str(c)] = 100 * int((pred & m).sum()) / int(m.sum())
        macro_rec = float(np.mean(list(per.values())))
        prauc = average_precision_score(is_atk.astype(int), maxprob) if k == 1 else float('nan')
        rows.append((k, f1, rec, prec, fpr, fp, mcc, macro_rec, fpm, prauc, tp, fn, tn, per))
        print(f"\n=== k>={k} (tau=0.5) ===")
        print(f"F1={f1:.2f} recall={rec:.2f} prec={prec:.2f} FPR={fpr:.3f}% "
              f"#FP={fp:,} MCC={mcc:.4f} macro-recall={macro_rec:.2f} FP/million={fpm:,.0f}"
              + (f" PR-AUC(OR)={prauc:.4f}" if k == 1 else ""))
        print(f"Confusion: TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")
        print("Per-attack recall:")
        for cn, r in per.items():
            print(f"  {cn:<24} {r:6.2f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/operational_metrics.csv", "w") as fh:
        fh.write("k,metric,value\n")
        for (k, f1, rec, prec, fpr, fp, mcc, macro, fpm, prauc, tp, fn, tn, per) in rows:
            for m, v in (("F1", f1), ("recall", rec), ("precision", prec), ("FPR", fpr),
                         ("FP", fp), ("MCC", mcc), ("macro_recall", macro), ("FP_per_million", fpm),
                         ("PR_AUC_OR", prauc), ("TP", tp), ("FN", fn), ("TN", tn)):
                fh.write(f"{k},{m},{v}\n")
            for cn, r in per.items():
                fh.write(f"{k},recall[{cn}],{r}\n")
    print("\n[op] CSV: results/operational_metrics.csv")


if __name__ == "__main__":
    main()
