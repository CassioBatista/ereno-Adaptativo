"""Estrutura dos falsos positivos da fusão OR e regras de combinação.

Reproduz os 10 especialistas federados do ERENO (particionador attack,
benign_cap), coleta a predição de cada um no teste global e analisa:
  - quantas detecções são FP (fusão OR);
  - quantos especialistas disparam em cada benigno falso-alarmado;
  - recall / FP / FPR sob OR (k≥1), k≥2, k≥3, e excluindo os ruidosos.

Responde: "dá para zerar os FP combinando as detecções?"

Uso: python scripts/analise_fp.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb

import python.util as util
from fd.dataset import load_and_partition

DATASET   = "all_in_one_ereno_train"
TESTFILE  = "all_in_one_ereno_test"
FEATURES  = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]  # combinado-24
N_CLIENTS = 10
SEED      = 42
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}


def main():
    print("[fp] carregando e particionando (attack, benign_cap=500k)...")
    partitions, _, _ = load_and_partition(
        f"{DATASET}.csv", FEATURES, N_CLIENTS, seed=SEED,
        partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    _, _, classes = util.load_arff(f"{DATASET}.csv")

    # especialista de cada cliente (nome)
    spec = []
    for _, y_c in partitions:
        atk = sorted(int(c) for c in np.unique(y_c) if int(c) != nc)
        spec.append("+".join(classes[a] for a in atk))

    print("[fp] treinando 10 especialistas...")
    boosters = []
    for cid, (X_c, y_c) in enumerate(partitions):
        Xtr, ytr = X_c, (y_c != nc).astype(int)
        npos, nneg = int(ytr.sum()), int((ytr == 0).sum())
        p = dict(XGB_PARAMS)
        if npos and nneg:
            p["scale_pos_weight"] = nneg / npos
        boosters.append(xgb.train(p, xgb.DMatrix(Xtr, label=ytr),
                                  num_boost_round=10, verbose_eval=False))

    print("[fp] carregando teste...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES)
    del Xte_raw
    yte = (yte_m != nc).astype(int)
    dte = xgb.DMatrix(Xte)

    # matriz de predições [n_test x 10]
    P = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int8) for b in boosters])
    votes = P.sum(axis=1)                     # nº de especialistas que disparam
    is_atk = yte == 1
    is_ben = yte == 0
    n_atk, n_ben = int(is_atk.sum()), int(is_ben.sum())

    def stats(pred):
        vp = int((pred & is_atk).sum()); fp = int((pred & is_ben).sum())
        rec = 100 * vp / n_atk
        fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, prec, fpr, fp

    print(f"\n[fp] teste: {n_atk:,} ataques, {n_ben:,} normais")
    print(f"\n{'regra':<22}{'F1':>7}{'Recall':>8}{'Prec':>7}{'FPR':>7}{'#FP':>9}")
    for k in (1, 2, 3):
        f1, rec, prec, fpr, fp = stats(votes >= k)
        print(f"OR/k≥{k:<19}{f1:>7.2f}{rec:>8.2f}{prec:>7.2f}{fpr:>7.3f}{fp:>9,}")

    # FPR individual de cada especialista → identificar ruidosos
    print(f"\n[fp] FPR individual por especialista:")
    ind_fpr = []
    for cid in range(N_CLIENTS):
        fpr = 100 * int((P[:, cid] & is_ben).sum()) / n_ben
        ind_fpr.append(fpr)
        print(f"    cli {cid} {spec[cid]:<24} FPR={fpr:.3f}%")

    # OR só com os "limpos" (FPR individual abaixo do limiar)
    for lim in (0.05, 0.01):
        keep = [c for c in range(N_CLIENTS) if ind_fpr[c] < lim]
        pred = P[:, keep].sum(axis=1) >= 1
        f1, rec, prec, fpr, fp = stats(pred)
        print(f"\nOR só limpos (FPR<{lim}%, {len(keep)} especialistas {keep}):")
        print(f"    F1={f1:.2f} Recall={rec:.2f} Prec={prec:.2f} FPR={fpr:.3f}% #FP={fp:,}")

    # histograma: quantos especialistas disparam nos benignos falso-alarmados
    fp_votes = votes[is_ben & (votes >= 1)]
    print(f"\n[fp] dos {len(fp_votes):,} benignos falso-alarmados (fusão OR),")
    print(f"     disparados por 1 especialista: {int((fp_votes==1).sum()):,} "
          f"({100*(fp_votes==1).mean():.1f}%)")
    print(f"     por ≥2 especialistas: {int((fp_votes>=2).sum()):,} "
          f"({100*(fp_votes>=2).mean():.1f}%)")


if __name__ == "__main__":
    main()
