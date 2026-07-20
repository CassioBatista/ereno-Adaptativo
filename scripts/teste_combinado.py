"""Compara global-15 × superset-10 × combinado-19 (união dos dois).

Responde: juntar as 9 features que só o global-15 tinha às 10 do superset
melhora o desempenho, ou as 9 (mais elétricas, pouco discriminativas por
ataque) só adicionam ruído?
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb

import python.util as util
from fd.dataset import load_and_partition

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
ALL58 = list(range(1, 59))
GLOBAL15   = [5, 7, 8, 11, 19, 31, 33, 40, 41, 42, 44, 45, 50, 57, 58]
SUPERSET10 = [8, 20, 40, 41, 42, 43, 46, 55, 57, 58]
COMBINADO19 = sorted(set(GLOBAL15) | set(SUPERSET10))
N, SEED = 10, 42
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}


def main():
    print(f"[comb] combinado-19 = {COMBINADO19}")
    partitions, _, _ = load_and_partition(
        f"{DATASET}.csv", ALL58, N, seed=SEED, partitioner="attack",
        benign_cap=500000)
    nc = util.normal_class
    Xte_all, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    is_ben = yte_m == nc; n_ben = int(is_ben.sum())
    is_atk = yte_m != nc; n_atk = int(is_atk.sum())

    def run(cols):
        c0 = [c - 1 for c in cols]
        preds = []
        for X_c, y_c in partitions:
            yb = (y_c != nc).astype(int)
            p, q = int(yb.sum()), int((yb == 0).sum())
            pr = dict(XGB_PARAMS)
            if p and q: pr["scale_pos_weight"] = q / p
            b = xgb.train(pr, xgb.DMatrix(X_c[:, c0], label=yb),
                          num_boost_round=10, verbose_eval=False)
            preds.append((b.predict(xgb.DMatrix(Xte_all[:, c0])) >= 0.5).astype(np.int8))
        P = np.column_stack(preds)
        pred = (P.sum(1) >= 1)
        vp = int((pred & is_atk).sum()); fp = int((pred & is_ben).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        m = [100 * int((P[:, c] & is_ben).sum()) / n_ben for c in (8, 9)]
        return f1, rec, prec, fpr, fp, m

    print(f"\n{'conjunto':<16}{'F1':>7}{'Recall':>8}{'Prec':>7}{'FPR':>7}{'#FP':>9}"
          f"{'masq8':>9}{'masq9':>9}")
    for nome, cols in [("global-15", GLOBAL15), ("superset-10", SUPERSET10),
                       ("combinado-19", COMBINADO19)]:
        f1, rec, prec, fpr, fp, m = run(cols)
        print(f"{nome:<16}{f1:>7.2f}{rec:>8.2f}{prec:>7.2f}{fpr:>7.3f}{fp:>9,}"
              f"{m[0]:>8.3f}%{m[1]:>8.3f}%")


if __name__ == "__main__":
    main()
