#!/usr/bin/env python3
"""Check de estabilidade multi-seed dos headlines (FL), teste completo.
Carrega train e test UMA vez; para cada seed re-particiona (attack e iid) + treina 10
especialistas + avalia OR(k=1) e k>=2. Mede variacao entre seeds (mean, std, IC95).
Objetivo: mostrar que specialist vs iid e os headlines sao seed-estaveis (camada de
robustez; NAO substitui as tabelas, cuja referencia e o seed 42).
Saida: results/multiseed_headlines.csv
Rodar: ~/venv-ereno314/bin/python scripts/multiseed_headlines.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import python.util as util
from fd.dataset import _attack_per_client, _iid

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
SEEDS = [42, 7, 13, 21, 99]
N, CAP = 10, 500000


def main():
    print("[ms] carregando train (uma vez)...")
    X_all, y_all, _ = util.load_arff(f"{DATASET}.csv")
    nc = util.normal_class
    cols = [i - 1 for i in sorted(FEATURES)]
    Xf_full = y_full_ok = X_all[:, cols]
    y_full = y_all
    print(f"[ms] train: {len(y_full):,}  normal_class={nc}")

    def partition(seed, which):
        rng = np.random.default_rng(seed)
        normal_idx = np.where(y_full == nc)[0]
        drop = rng.permutation(normal_idx)[CAP:]
        keep = np.ones(len(y_full), dtype=bool); keep[drop] = False
        Xf, y = Xf_full[keep], y_full[keep]
        splits = _attack_per_client(Xf, y, N, rng) if which == "attack" else _iid(Xf, y, N, rng)
        return [(Xf[idx], y[idx]) for idx in splits]

    def train_pool(seed, which):
        pool = []
        for X_c, y_c in partition(seed, which):
            Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=seed)
            yy = (yct != nc).astype(int)
            p, q = int(yy.sum()), int((yy == 0).sum())
            pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": seed, "nthread": 2}
            if p and q: pr["scale_pos_weight"] = q / p
            pool.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10, verbose_eval=False))
        return pool

    print("[ms] carregando teste (uma vez)...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    dte = xgb.DMatrix(Xte)
    print(f"[ms] teste: {len(yte_m):,} ({n_atk:,} atk / {n_ben:,} ben)")

    def evaluate(pool, k):
        votes = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in pool]).sum(1)
        pred = votes >= k
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, fpr

    rows = []
    for seed in SEEDS:
        pa = train_pool(seed, "attack"); pi = train_pool(seed, "iid")
        for name, pool, k in (("attack_OR", pa, 1), ("attack_k2", pa, 2), ("iid_OR", pi, 1)):
            f1, rec, fpr = evaluate(pool, k)
            rows.append((seed, name, f1, rec, fpr))
            print(f"  seed {seed:>3} {name:<10} F1={f1:6.2f} rec={rec:6.2f} fpr={fpr:6.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/multiseed_headlines.csv", "w") as fh:
        fh.write("seed,config,F1,Recall,FPR\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    t95 = 2.776  # t(0.975, df=4)
    print(f"\n{'config':<10}{'F1 (mean+/-CI95)':>22}{'Recall':>18}{'FPR':>18}")
    for name in ("attack_OR", "attack_k2", "iid_OR"):
        d = {m: np.array([r[i] for r in rows if r[1] == name]) for m, i in (("F1", 2), ("Recall", 3), ("FPR", 4))}
        s = {}
        for m, arr in d.items():
            s[m] = (arr.mean(), t95 * arr.std(ddof=1) / math.sqrt(len(arr)))
        print(f"{name:<10}{s['F1'][0]:9.2f} +/-{s['F1'][1]:5.2f}     "
              f"{s['Recall'][0]:8.2f} +/-{s['Recall'][1]:4.2f}     "
              f"{s['FPR'][0]:7.3f} +/-{s['FPR'][1]:5.3f}")
    print("\n[ms] CSV: results/multiseed_headlines.csv")


if __name__ == "__main__":
    main()
