"""Divisão do especialista de masquerade + corroboração.

Divide masquerade_fake_fault (o sensor de maior FP) em 2 e 3 sub-especialistas
sobre metades/terços DISJUNTOS do ataque + fatias disjuntas de benigno, e mede:
  - recall (na classe masquerade_fake_fault) e FPR de cada sub;
  - SOBREPOSIÇÃO dos FP entre os subs (Jaccard) — a métrica decisiva:
      alto  = FP sistemático (masquerade parece normal p/ qualquer detector);
      baixo = FP decorrelacionado (idiossincrasia de treino → corroboração ajuda);
  - k≥2 / k≥3 (corroboração): recall e FPR resultantes.

Features: combinado-19 (adotado). Uso: python scripts/divisao_masquerade.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb

import python.util as util

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
COMBINED19 = [5, 7, 8, 11, 19, 20, 31, 33, 40, 41, 42, 43, 44, 45, 46, 50, 55, 57, 58]
SEED = 42
BENIGN_TOTAL = 500000
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}


def main():
    rng = np.random.default_rng(SEED)
    print("[div] carregando train...")
    X, y, classes = util.load_arff(f"{DATASET}.csv")
    X = X[:, [c - 1 for c in COMBINED19]]
    nc = util.normal_class
    TARGET = classes.index("masquerade_fake_fault")

    atk_idx = np.where(y == TARGET)[0]
    ben_idx = rng.permutation(np.where(y == nc)[0])[:BENIGN_TOTAL]
    print(f"[div] masquerade_fake_fault: {len(atk_idx)} amostras; benigno: {len(ben_idx)}")

    print("[div] carregando teste...")
    Xte, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = Xte[:, [c - 1 for c in COMBINED19]]
    is_ben = yte == nc; n_ben = int(is_ben.sum())
    is_tgt = yte == TARGET; n_tgt = int(is_tgt.sum())
    dte = xgb.DMatrix(Xte)
    ben_pos = np.where(is_ben)[0]   # posições dos benignos no teste

    def train_sub(a_sub, b_sub):
        Xtr = np.vstack([X[a_sub], X[b_sub]])
        ytr = np.concatenate([np.ones(len(a_sub)), np.zeros(len(b_sub))]).astype(int)
        pr = dict(XGB_PARAMS); pr["scale_pos_weight"] = len(b_sub) / max(len(a_sub), 1)
        b = xgb.train(pr, xgb.DMatrix(Xtr, label=ytr), num_boost_round=10,
                      verbose_eval=False)
        pred = (b.predict(dte) >= 0.5).astype(np.int8)
        rec = 100 * int(pred[is_tgt].sum()) / n_tgt
        fpr = 100 * int(pred[is_ben].sum()) / n_ben
        fp_set = set(ben_pos[pred[ben_pos] == 1])
        return pred, rec, fpr, fp_set

    for K in (1, 2, 3):
        a_parts = np.array_split(rng.permutation(atk_idx), K)
        b_parts = np.array_split(rng.permutation(ben_idx), K)
        subs = [train_sub(a_parts[i], b_parts[i]) for i in range(K)]
        print(f"\n{'='*60}\n  DIVISÃO em {K} sub-especialista(s)\n{'='*60}")
        for i, (_, rec, fpr, _) in enumerate(subs):
            print(f"  sub {i}: recall(masq)={rec:.2f}%  FPR={fpr:.3f}%")
        if K == 1:
            continue
        # sobreposição dos FP (Jaccard sobre pares)
        fp_sets = [s[3] for s in subs]
        pares = [(len(fp_sets[i] & fp_sets[j]) / max(len(fp_sets[i] | fp_sets[j]), 1))
                 for i in range(K) for j in range(i + 1, K)]
        print(f"  Jaccard médio dos FP entre subs: {np.mean(pares):.3f} "
              f"({'SISTEMÁTICO' if np.mean(pares) > 0.5 else 'DECORRELACIONADO'})")
        # corroboração k≥2 .. k≥K
        votes = np.sum([s[0] for s in subs], axis=0)
        for k in range(2, K + 1):
            pred = (votes >= k)
            rec = 100 * int(pred[is_tgt].sum()) / n_tgt
            fp = int(pred[is_ben].sum()); fpr = 100 * fp / n_ben
            fpr_ind = np.mean([s[2] for s in subs])
            print(f"  k≥{k}: recall(masq)={rec:.2f}%  FPR={fpr:.3f}%  "
                  f"(FPR ind. médio {fpr_ind:.3f}% → redução {100*(1-fpr/max(fpr_ind,1e-9)):.0f}%)")


if __name__ == "__main__":
    main()
