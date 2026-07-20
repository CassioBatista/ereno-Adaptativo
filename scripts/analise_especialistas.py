"""Diagnóstico dos especialistas ruidosos e teste de features por-especialista.

PARTE A — para cada especialista: recall NA PRÓPRIA classe de ataque + FPR.
          Distingue "problema de precisão" (detecta bem, erra em benigno)
          de "problema de cobertura" (detecta mal a própria classe).

PARTE B — hipótese do usuário: features por-especialista. Retreina os
          especialistas de masquerade (clientes 8,9) com 3 conjuntos de
          features e compara recall-da-classe + FPR:
            (i)  features XGB globais (15) — o baseline atual
            (ii) TODAS as 58 features
            (iii) só protocolo + derivadas de consistência (candidatas
                  discriminativas para masquerade)

Uso: python scripts/analise_especialistas.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb

import python.util as util
from fd.dataset import load_and_partition

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
ALL58     = list(range(1, 59))
XGB15     = [5, 7, 8, 11, 19, 31, 33, 40, 41, 42, 44, 45, 50, 57, 58]
# protocolo (SqNum,StNum,cbStatus,timeAllowed,gooseLen) + derivadas de
# consistência (stDiff,sqDiff,gooseLenDiff,cbStatusDiff,apduDiff,frameDiff,
# timestampDiff,tDiff,timeFromLastChange,delay) — candidatas p/ masquerade
PROTO_DIFF = [40, 41, 42, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
N, SEED = 10, 42
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}


def train_spec(X, y_multi, nc, cols_1based):
    cols = [c - 1 for c in cols_1based]
    Xc = X[:, cols]
    yb = (y_multi != nc).astype(int)
    p, q = int(yb.sum()), int((yb == 0).sum())
    pr = dict(XGB_PARAMS)
    if p and q: pr["scale_pos_weight"] = q / p
    return xgb.train(pr, xgb.DMatrix(Xc, label=yb), num_boost_round=10,
                     verbose_eval=False)


def main():
    print("[esp] particionando com TODAS as 58 features (attack, benign_cap)...")
    partitions, _, _ = load_and_partition(
        f"{DATASET}.csv", ALL58, N, seed=SEED,
        partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    _, _, classes = util.load_arff(f"{DATASET}.csv")
    spec = []
    for _, y_c in partitions:
        atk = sorted(int(c) for c in np.unique(y_c) if int(c) != nc)
        spec.append("+".join(classes[a] for a in atk))

    print("[esp] carregando teste (58 features)...")
    Xte_all, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    is_ben = yte_m == nc
    n_ben = int(is_ben.sum())
    # máscara por classe de ataque presente no teste
    atk_classes = sorted(int(c) for c in np.unique(yte_m) if int(c) != nc)

    def eval_spec(booster, cols_1based):
        cols = [c - 1 for c in cols_1based]
        pred = (booster.predict(xgb.DMatrix(Xte_all[:, cols])) >= 0.5).astype(np.int8)
        fpr = 100 * int(pred[is_ben].sum()) / n_ben
        rec_por_classe = {}
        for a in atk_classes:
            m = yte_m == a
            rec_por_classe[classes[a]] = 100 * int(pred[m].sum()) / int(m.sum())
        return pred, fpr, rec_por_classe

    # ── PARTE A: diagnóstico com as features XGB globais ─────────────────────
    print("\n" + "=" * 72)
    print("  PARTE A — cada especialista: recall NA PRÓPRIA classe + FPR")
    print("=" * 72)
    print(f"  {'cli':>3} {'especialista':<24} {'rec_própria':>11} {'FPR':>8}")
    for cid, (X_c, y_c) in enumerate(partitions):
        b = train_spec(X_c, y_c, nc, XGB15)
        _, fpr, recs = eval_spec(b, XGB15)
        own = spec[cid]
        # recall na própria classe (pode ser "a+b" p/ redundantes; usa a 1ª)
        own_cls = own.split("+")[0]
        print(f"  {cid:>3} {own:<24} {recs.get(own_cls, float('nan')):>10.2f}% {fpr:>7.3f}%")

    # ── PARTE B: features por-especialista nos sensores de masquerade ────────
    print("\n" + "=" * 72)
    print("  PARTE B — features por-especialista (clientes 8 e 9, masquerade)")
    print("=" * 72)
    for cid in (8, 9):
        X_c, y_c = partitions[cid]
        own_cls = spec[cid].split("+")[0]
        print(f"\n  cliente {cid} — {spec[cid]}:")
        print(f"    {'conjunto de features':<28}{'rec_própria':>11}{'FPR':>9}{'#feat':>7}")
        for nome, cols in [("XGB global (15)", XGB15),
                           ("todas (58)", ALL58),
                           ("protocolo+diffs (16)", PROTO_DIFF)]:
            b = train_spec(X_c, y_c, nc, cols)
            _, fpr, recs = eval_spec(b, cols)
            print(f"    {nome:<28}{recs.get(own_cls, float('nan')):>10.2f}%{fpr:>8.3f}%{len(cols):>7}")


if __name__ == "__main__":
    main()
