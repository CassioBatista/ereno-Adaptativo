"""Contraste GL != FL com o operador de MERGE (nao-idempotente) do XGBoost.

Complementa o resultado do OR (union, idempotente -> GL=FL, 100% concordancia).
Aqui usa aggregation='xgb' (merge: concatena arvores, soma margens):
  - FL-merge: funde os 10 especialistas UMA vez (one-shot).
  - GL-merge: difusao gossip com merge -> um especialista chega por varios
    caminhos e e fundido varias vezes -> arvores duplicadas -> booster diferente.
Mede: nº de arvores de cada, F1/recall, e concordancia por-amostra FL vs GL.
Esperado: GL != FL (concordancia < 100%), ao contrario do OR.

Rodar: ~/venv-ereno314/bin/python scripts/merge_gl_vs_fl.py
"""
import os
import sys
from functools import reduce
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from flwr.common import (parameters_to_ndarrays, ndarrays_to_parameters,
                         FitRes, Status, Code)

import python.util as util
from fd.dataset import load_and_partition
from fd.client_xgb import XgbClient
from fd.strategy.glow_strategy import GlowStrategy, _merge_boosters
from fd.topology import build_topology

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
N, SEED = 10, 42
GL_ROUNDS = 14
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}


def _boosters(params):
    out = []
    for a in parameters_to_ndarrays(params):
        if a.size:
            b = xgb.Booster(); b.load_model(bytearray(a.tobytes())); out.append(b)
    return out


def _ntrees(b):
    try:
        return b.num_boosted_rounds()
    except Exception:
        return len(b.get_dump())


def _gl_diffuse(aggregation, client_tr):
    glow = GlowStrategy(build_topology("ring", N), aggregation=aggregation)
    clients = {str(i): XgbClient(i, Xct, y, Xct[:2], y[:2], XGB_PARAMS)
               for i, (Xct, y) in enumerate(client_tr)}

    class P:
        def __init__(s, c): s.cid = str(c)
    class M:
        def __init__(s, ps): s._p = {p.cid: p for p in ps}
        def all(s): return dict(s._p)
    mgr = M([P(i) for i in range(N)])
    params = ndarrays_to_parameters([np.array([], dtype=np.uint8)])
    for rnd in range(1, GL_ROUNDS + 1):
        instr = glow.configure_fit(rnd, params, mgr)
        res = []
        for proxy, ins in instr:
            nds = parameters_to_ndarrays(ins.parameters)
            out, n, m = clients[proxy.cid].fit(nds, dict(ins.config))
            res.append((proxy, FitRes(Status(Code.OK, "ok"),
                                      ndarrays_to_parameters(out), n, {**m, "cid": int(proxy.cid)})))
        agg, _ = glow.aggregate_fit(rnd, res, [])
        if agg is not None: params = agg
    return glow


def main():
    print("[merge] particionando...")
    partitions, X_f, y_all = load_and_partition(
        f"{DATASET}.csv", FEATURES, N, seed=SEED, partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    client_tr = []
    for X_c, y_c in partitions:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        client_tr.append((Xct, (yct != nc).astype(int)))

    print("[merge] treinando 10 especialistas...")
    fl = []
    for Xc, yy in client_tr:
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = dict(XGB_PARAMS)
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xc, label=yy), num_boost_round=10, verbose_eval=False))

    print("[merge] FL-merge (one-shot dos 10)...")
    fl_merged = reduce(_merge_boosters, fl)

    print(f"[merge] GL-merge (difusao {GL_ROUNDS} rounds, aggregation=xgb)...")
    glow = _gl_diffuse("xgb", client_tr)
    gl_list = _boosters(glow.get_node_model(0))
    gl_merged = gl_list[0] if len(gl_list) == 1 else reduce(_merge_boosters, gl_list)

    print("[merge] carregando teste...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    dte = xgb.DMatrix(Xte)

    def stats(booster):
        pred = (booster.predict(dte) >= 0.5).astype(np.int8)
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return pred, f1, rec, prec, fpr

    pf, f1f, rf, pcf, fprf = stats(fl_merged)
    pg, f1g, rg, pcg, fprg = stats(gl_merged)
    ag = 100 * (pf == pg).mean()
    ndiff = int((pf != pg).sum())

    print(f"\n{'modelo (MERGE)':<16}{'árvores':>9}{'F1':>8}{'Recall':>8}{'Prec':>7}{'FPR':>8}")
    print(f"{'FL-merge':<16}{_ntrees(fl_merged):>9}{f1f:>8.2f}{rf:>8.2f}{pcf:>7.2f}{fprf:>8.3f}")
    print(f"{'GL-merge':<16}{_ntrees(gl_merged):>9}{f1g:>8.2f}{rg:>8.2f}{pcg:>7.2f}{fprg:>8.3f}")
    print(f"\n[merge] concordância FL-merge vs GL-merge: {ag:.4f}%  ({ndiff:,} amostras divergem de {len(pf):,})")
    print(f"[merge] (comparar: com OR/union deu 100.00%, 0 divergências)")


if __name__ == "__main__":
    main()
