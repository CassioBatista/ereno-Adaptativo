"""Baseline CENTRALIZADO (monolitico) — XGBoost puro, SEM Flower.

Um unico modelo XGBoost treinado em TODO o train (sem particao, sem clientes,
sem rounds), avaliado no teste real. E o baseline nao-distribuido da tabela
comparativa; roda independente do pipeline Flower.

Detalhes (iguais ao monolitico embutido em main_dist.py, para reproduzir o
mesmo numero):
  - features: combinado_def-24 (GRASP penalizado L0.05);
  - train: mesmo conjunto da federacao (benign_cap=500k, ataques intactos);
  - rotulo binario: normal=0, ataque=1;
  - 100 arvores (orcamento fixo = total de arvores da federacao, p/ comparacao
    justa; o modelo NAO tem clientes — 100 e so o numero de arvores);
  - scale_pos_weight = n_neg/n_pos; teste do autor (2,9M).

Uso: python scripts/teste_central_xgb.py
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
COMBINED24 = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
N_TREES = 100      # orcamento fixo (= total de arvores da federacao); sem clientes
SEED = 42


def main():
    print(f"[central] features combinado-24 ({len(COMBINED24)}); {N_TREES} arvores (monolitico, sem clientes)")
    # mesmo train da federacao (benign_cap 500k); concatena as particoes = train inteiro
    partitions, _, _ = load_and_partition(
        f"{DATASET}.csv", ALL58, 10, seed=SEED, partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    c0 = [c - 1 for c in COMBINED24]
    X_tr = np.vstack([Xc[:, c0] for Xc, _ in partitions])
    y_tr = np.concatenate([(yc != nc).astype(int) for _, yc in partitions])

    Xte_all, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = Xte_all[:, c0]
    is_ben = yte == nc; n_ben = int(is_ben.sum())
    is_atk = yte != nc; n_atk = int(is_atk.sum())

    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    print(f"[central] train: {len(y_tr):,} ({n_pos:,} ataque / {n_neg:,} normal); teste: {len(yte):,}")
    booster = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "logloss",
         "max_depth": 4, "eta": 0.1, "seed": SEED, "nthread": 2,
         "scale_pos_weight": n_neg / max(n_pos, 1)},
        xgb.DMatrix(X_tr, label=y_tr), num_boost_round=N_TREES, verbose_eval=False)
    pred = (booster.predict(xgb.DMatrix(Xte)) >= 0.5).astype(int)

    vp = int((pred & is_atk).sum()); fp = int((pred & is_ben).sum())
    rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
    prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n{'conjunto':<16}{'n':>4}{'F1':>8}{'Recall':>8}{'Prec':>7}{'FPR':>8}{'#FP':>9}")
    print(f"{'CENTRALIZADO':<16}{len(COMBINED24):>4}{f1:>8.2f}{rec:>8.2f}{prec:>7.2f}{fpr:>8.3f}{fp:>9,}")


if __name__ == "__main__":
    main()
