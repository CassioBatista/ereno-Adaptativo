#!/usr/bin/env python3
"""Fatorial particionador x k, 5 seeds, teste completo (FL, N=10).
Particionadores: specialist(attack), iid, dirichlet(0.5), shard(2).
Metricas: OR(k=1) e k>=2. Saida: matriz fatorial (mean+/-IC95) + 3 contrastes primarios
com t-test pareado (entre seeds) e correcao de Holm-Bonferroni.
Carrega train/test UMA vez; re-particiona por (seed, particionador).
Saida: results/factorial_holm.csv
Rodar: ~/venv-ereno314/bin/python scripts/factorial_holm.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy import stats
import python.util as util
from fd.dataset import _attack_per_client, _iid, _dirichlet, _shard

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
SEEDS = [42, 7, 13, 21, 99]
PARTS = ["specialist", "iid", "dirichlet", "shard"]
N, CAP = 10, 500000


def main():
    print("[fac] carregando train (uma vez)...")
    X_all, y_all, _ = util.load_arff(f"{DATASET}.csv")
    nc = util.normal_class
    cols = [i - 1 for i in sorted(FEATURES)]
    Xf_full, y_full = X_all[:, cols], y_all

    def partition(seed, which):
        rng = np.random.default_rng(seed)
        nidx = np.where(y_full == nc)[0]
        drop = rng.permutation(nidx)[CAP:]
        keep = np.ones(len(y_full), dtype=bool); keep[drop] = False
        Xf, y = Xf_full[keep], y_full[keep]
        if which == "specialist": sp = _attack_per_client(Xf, y, N, rng)
        elif which == "iid":       sp = _iid(Xf, y, N, rng)
        elif which == "dirichlet": sp = _dirichlet(Xf, y, N, 0.5, rng)
        else:                      sp = _shard(Xf, y, N, 2, rng)
        return [(Xf[i], y[i]) for i in sp]

    def train_pool(seed, which):
        pool = []
        for X_c, y_c in partition(seed, which):
            if len(np.unique(y_c)) < 2:   # cliente degenerado (shard/dirichlet)
                Xct, yct = X_c, y_c
            else:
                Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=seed)
            yy = (yct != nc).astype(int)
            p, q = int(yy.sum()), int((yy == 0).sum())
            pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": seed, "nthread": 2}
            if p and q: pr["scale_pos_weight"] = q / p
            if p == 0:   # sem ataques nessa fatia -> booster trivial ainda serve (nao dispara)
                yy = yy.copy(); yy[0] = 1
            pool.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10, verbose_eval=False))
        return pool

    print("[fac] carregando teste (uma vez)...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    dte = xgb.DMatrix(Xte)
    print(f"[fac] teste: {len(yte_m):,} ({n_atk:,} atk / {n_ben:,} ben)")

    def stats_at(pool, k):
        votes = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in pool]).sum(1)
        pred = votes >= k
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, fpr

    # D[(part,k,metric)] = list over seeds
    D = {}
    rows = []
    for seed in SEEDS:
        for part in PARTS:
            pool = train_pool(seed, part)
            for k in (1, 2):
                f1, rec, fpr = stats_at(pool, k)
                rows.append((seed, part, k, f1, rec, fpr))
                for m, v in (("F1", f1), ("Recall", rec), ("FPR", fpr)):
                    D.setdefault((part, k, m), []).append(v)
                print(f"  seed {seed:>3} {part:<11} k>={k}  F1={f1:6.2f} rec={rec:6.2f} fpr={fpr:6.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/factorial_holm.csv", "w") as fh:
        fh.write("seed,partitioner,k,F1,Recall,FPR\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    t95 = 2.776
    def mc(part, k, m):
        a = np.array(D[(part, k, m)]); mu = a.mean(); ci = t95 * a.std(ddof=1) / math.sqrt(len(a)); return mu, ci

    print("\n===== FACTORIAL MATRIX (mean +/- IC95, full test) =====")
    print(f"{'partitioner':<12}{'OR F1':>16}{'k>=2 F1':>16}{'OR FPR':>16}{'k>=2 FPR':>16}")
    for part in PARTS:
        f1o = mc(part, 1, "F1"); f1k = mc(part, 2, "F1"); fro = mc(part, 1, "FPR"); frk = mc(part, 2, "FPR")
        print(f"{part:<12}{f1o[0]:8.2f}+/-{f1o[1]:4.2f}{f1k[0]:9.2f}+/-{f1k[1]:4.2f}"
              f"{fro[0]:9.3f}+/-{fro[1]:5.3f}{frk[0]:9.3f}+/-{frk[1]:5.3f}")

    # ---- 3 contrastes primarios (t-test pareado entre seeds) ----
    S = len(SEEDS)
    def arr(part, k, m): return np.array(D[(part, k, m)])
    nonspec_or_f1 = np.mean([arr(p, 1, "F1") for p in ("iid", "dirichlet", "shard")], axis=0)
    nonspec_k2_f1 = np.mean([arr(p, 2, "F1") for p in ("iid", "dirichlet", "shard")], axis=0)

    c1 = arr("specialist", 1, "F1") - nonspec_or_f1          # C1: partition @ OR (F1)
    c2 = arr("specialist", 1, "FPR") - arr("specialist", 2, "FPR")  # C2: OR-FPR minus k2-FPR (reducao)
    c3 = arr("specialist", 2, "F1") - nonspec_k2_f1          # C3: partition @ k>=2 (F1)

    contrasts = {
        "C1 specialist>non-spec (OR, F1)": c1,
        "C2 k>=2 cuts FPR (specialist)":   c2,
        "C3 specialist>non-spec (k>=2, F1)": c3,
    }
    praw = {}
    print("\n===== 3 PRIMARY CONTRASTS (paired t-test across seeds) =====")
    for name, d in contrasts.items():
        t, p = stats.ttest_1samp(d, 0.0)
        praw[name] = p
        print(f"{name:<36} diff={d.mean():8.3f} (95%CI +/-{t95*d.std(ddof=1)/math.sqrt(S):.3f})  t={t:7.2f}  p={p:.2e}")

    # Holm-Bonferroni
    order = sorted(praw.items(), key=lambda kv: kv[1])
    m = len(order); running = 0.0; adj = {}
    for i, (name, p) in enumerate(order):
        a = min(1.0, p * (m - i)); running = max(running, a); adj[name] = running
    print("\n===== HOLM-BONFERRONI (family of 3) =====")
    for name, _ in order:
        sig = "***" if adj[name] < 0.001 else ("**" if adj[name] < 0.01 else ("*" if adj[name] < 0.05 else "ns"))
        print(f"{name:<36} p_raw={praw[name]:.2e}  p_holm={adj[name]:.2e}  {sig}")
    print("\n[fac] CSV: results/factorial_holm.csv")


if __name__ == "__main__":
    main()
