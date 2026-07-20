"""Superconjunto de features = união das features discriminativas por ataque.

Hipótese: cada ataque tem seu conjunto de features boas; se muitas forem
comuns, a UNIÃO (superconjunto) é compacta e cobre todos os ataques —
inclusive as pistas específicas do masquerade que a seleção GLOBAL
(otimizada para o problema todo, dominado pelos ataques fáceis) perde.

Etapas:
  1. para cada ataque: features com importância (gain) relevante num
     XGBoost binário (ataque vs normal) — proxy rápido de seleção wrapper;
  2. sobreposição entre os conjuntos (Jaccard) + superconjunto (união);
  3. testa o superconjunto: retreina os 10 especialistas e mede FPR
     individual + fusão OR, vs o baseline global-15.

Uso: python scripts/superset_features.py
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
XGB15 = [5, 7, 8, 11, 19, 31, 33, 40, 41, 42, 44, 45, 50, 57, 58]
N, SEED = 10, 42
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}
GAIN_SHARE = 0.03   # feature entra na seleção do ataque se gain >= 3% do total


def main():
    rng = np.random.default_rng(SEED)
    print("[super] carregando train (58 features)...")
    X, y_multi, classes = util.load_arff(f"{DATASET}.csv")
    X = X[:, [c - 1 for c in ALL58]]
    nc = util.normal_class
    atk_classes = sorted(int(c) for c in np.unique(y_multi) if int(c) != nc)

    # subamostra de normais p/ os treinos binários por ataque
    norm_idx = np.where(y_multi == nc)[0]
    norm_sub = rng.choice(norm_idx, size=min(60000, len(norm_idx)), replace=False)

    # ── 1. seleção por ataque (importância gain) ─────────────────────────────
    print("[super] seleção de features por ataque (gain >= 3%)...")
    per_attack = {}
    for a in atk_classes:
        a_idx = np.where(y_multi == a)[0]
        idx = np.concatenate([a_idx, norm_sub])
        Xa = X[idx]; ya = (y_multi[idx] == a).astype(int)
        p, q = int(ya.sum()), int((ya == 0).sum())
        pr = dict(XGB_PARAMS); pr["scale_pos_weight"] = q / max(p, 1)
        b = xgb.train(pr, xgb.DMatrix(Xa, label=ya,
                                      feature_names=[f"F{c}" for c in ALL58]),
                      num_boost_round=30, verbose_eval=False)
        sc = b.get_score(importance_type="gain")
        tot = sum(sc.values()) or 1.0
        sel = sorted(int(f[1:]) for f, g in sc.items() if g / tot >= GAIN_SHARE)
        per_attack[a] = set(sel)
        print(f"    {classes[a]:<22} ({len(sel)}): {sel}")

    # ── 2. sobreposição + superconjunto ──────────────────────────────────────
    superset = sorted(set().union(*per_attack.values()))
    print(f"\n[super] SUPERCONJUNTO (união, {len(superset)} features): {superset}")
    # features comuns a TODOS os ataques
    common = sorted(set.intersection(*per_attack.values())) if per_attack else []
    print(f"[super] comuns a TODOS os {len(atk_classes)} ataques ({len(common)}): {common}")
    # Jaccard médio entre pares
    keys = list(per_attack)
    j = [len(per_attack[keys[i]] & per_attack[keys[k]]) /
         max(len(per_attack[keys[i]] | per_attack[keys[k]]), 1)
         for i in range(len(keys)) for k in range(i + 1, len(keys))]
    print(f"[super] Jaccard médio entre ataques: {np.mean(j):.2f}")
    print(f"[super] vs global-15: {sorted(set(superset) & set(XGB15))} comuns; "
          f"só no superset: {sorted(set(superset) - set(XGB15))}; "
          f"só no global: {sorted(set(XGB15) - set(superset))}")

    # ── 3. teste: especialistas com superconjunto vs global-15 ───────────────
    print("\n[super] teste no split attack + teste global...")
    partitions, _, _ = load_and_partition(
        f"{DATASET}.csv", ALL58, N, seed=SEED, partitioner="attack",
        benign_cap=500000)
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
        pred_or = (P.sum(1) >= 1)
        vp = int((pred_or & is_atk).sum()); fp = int((pred_or & is_ben).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        # FPR dos 2 sensores de masquerade (clientes 8,9)
        m_fpr = [100 * int((P[:, c] & is_ben).sum()) / n_ben for c in (8, 9)]
        return f1, rec, prec, fpr, fp, m_fpr

    print(f"\n{'conjunto':<20}{'F1':>7}{'Recall':>8}{'Prec':>7}{'FPR':>7}{'#FP':>9}"
          f"{'masq8_FPR':>11}{'masq9_FPR':>11}")
    for nome, cols in [("global-15", XGB15), (f"superset-{len(superset)}", superset)]:
        f1, rec, prec, fpr, fp, m = run(cols)
        print(f"{nome:<20}{f1:>7.2f}{rec:>8.2f}{prec:>7.2f}{fpr:>7.3f}{fp:>9,}"
              f"{m[0]:>10.3f}%{m[1]:>10.3f}%")


if __name__ == "__main__":
    main()
