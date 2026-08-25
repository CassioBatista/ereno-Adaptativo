#!/usr/bin/env python3
"""RQ1 — equivalencia sob POOL IDENTICO (N=14, 2/ataque), teste completo.
FL e GL avaliam a MESMA uniao (14 boosters, seeded) para k=1,2,3.
Sob pool identico, FL=GL para TODO k (mesma uniao -> mesmos votos -> mesmo k-de-n).
Contrasta com a Tabela 7 (N=10), onde FL=N e GL=2N-1 (pools diferentes) e o k>=2 diverge.
Saida: results/identical_pool_n14.csv
Rodar: ~/venv-ereno314/bin/python scripts/identical_pool_n14.py
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
    print("[rq1] particionando N=14 (attack -> 2/ataque)...")
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
    print(f"[rq1] pool IDENTICO = {len(fl)} boosters (uniao seeded; FL e GL usam ESTE mesmo pool)")

    print("[rq1] teste COMPLETO...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    print(f"[rq1] teste: {len(yte_m):,} ({n_atk:,} ataque / {n_ben:,} benigno)")
    dte = xgb.DMatrix(Xte)
    votes = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in fl]).sum(1)  # pool IDENTICO

    def stats(pred):
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100*vp/n_atk; fpr = 100*fp/n_ben
        prec = 100*vp/max(vp+fp, 1); f1 = 2*prec*rec/max(prec+rec, 1e-9)
        return f1, rec, prec, fpr, fp

    rows = []
    print(f"\n{'rule':<9}{'arch':<5}{'F1':>8}{'Recall':>8}{'Prec':>8}{'FPR':>8}{'#FP':>9}")
    for k in (1, 2, 3):
        f1, rec, prec, fpr, fp = stats(votes >= k)                # FL=GL: pool identico
        rule = f"k>={k}" + (" (OR)" if k == 1 else "")
        for arch in ("FL", "GL"):
            print(f"{rule:<9}{arch:<5}{f1:>8.2f}{rec:>8.2f}{prec:>8.2f}{fpr:>8.3f}{fp:>9,}")
            rows.append((rule, arch, len(fl), f1, rec, prec, fpr, fp))

    os.makedirs("results", exist_ok=True)
    with open("results/identical_pool_n14.csv", "w") as fh:
        fh.write("rule,arch,pool,F1,Recall,Precision,FPR,FP\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print("\n[rq1] Sob POOL IDENTICO, FL == GL para k=1,2,3 (por construcao).")
    print("[rq1] CSV: results/identical_pool_n14.csv")


if __name__ == "__main__":
    main()
