"""Votação ENTRE as arquiteturas: Centralizado × FL × GL.

Reproduz os três modelos finais (ERENO, features XGB, 10 clientes attack,
benign_cap) e vota a detecção por sample no teste global:
  k=1: alarme se QUALQUER arquitetura detecta (união)
  k=2: alarme se ≥2 das 3 concordam
  k=3: alarme se as 3 concordam (interseção)

Objetivo: verificar se combinar arquiteturas reduz os falsos positivos —
o centralizado é um modelo independente do ensemble distribuído (FL/GL).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from flwr.common import (ndarrays_to_parameters, parameters_to_ndarrays,
                         FitRes, Status, Code)

import python.util as util
from fd.dataset import load_and_partition
from fd.client_xgb import XgbClient
from fd.strategy.glow_strategy import GlowStrategy
from fd.topology import build_topology

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [5, 7, 8, 11, 19, 31, 33, 40, 41, 42, 44, 45, 50, 57, 58]
N, SEED = 10, 42
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}
GL_ROUNDS = 14   # suficiente para difundir num anel de 10


def _boosters(params):
    out = []
    for a in parameters_to_ndarrays(params):
        if a.size:
            b = xgb.Booster(); b.load_model(bytearray(a.tobytes())); out.append(b)
    return out


def _or_predict(boosters, dte):
    return (np.max([b.predict(dte) for b in boosters], axis=0) >= 0.5).astype(np.int8)


def main():
    print("[voto] particionando (attack, benign_cap=500k)...")
    partitions, X_f, y_all = load_and_partition(
        f"{DATASET}.csv", FEATURES, N, seed=SEED,
        partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    y_all_b = (y_all != nc).astype(int)

    # splits locais por cliente (80/20, como no main_dist)
    client_tr = []
    for X_c, y_c in partitions:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        client_tr.append((Xct, (yct != nc).astype(int)))

    # ── 1. CENTRALIZADO (monolítico) ─────────────────────────────────────────
    print("[voto] treinando centralizado...")
    npos, nneg = int(y_all_b.sum()), int((y_all_b == 0).sum())
    cen = xgb.train({**XGB_PARAMS, "eval_metric": "logloss",
                     "scale_pos_weight": nneg / max(npos, 1)},
                    xgb.DMatrix(X_f, label=y_all_b),
                    num_boost_round=N * 10, verbose_eval=False)

    # ── 2. FL (10 especialistas do zero, união OR) ───────────────────────────
    print("[voto] treinando FL (10 especialistas)...")
    fl = []
    for Xct, yb in client_tr:
        p, q = int(yb.sum()), int((yb == 0).sum())
        pr = dict(XGB_PARAMS)
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xct, label=yb),
                            num_boost_round=10, verbose_eval=False))

    # ── 3. GL (difusão gossip em processo, pool evoluído) ────────────────────
    print(f"[voto] simulando GL ({GL_ROUNDS} rounds de gossip)...")
    glow = GlowStrategy(build_topology("ring", N), aggregation="xgb_union")
    clients = {str(i): XgbClient(i, Xct, yb, Xct[:2], yb[:2], XGB_PARAMS)
               for i, (Xct, yb) in enumerate(client_tr)}

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
                                      ndarrays_to_parameters(out), n,
                                      {**m, "cid": int(proxy.cid)})))
        agg, _ = glow.aggregate_fit(rnd, res, [])
        if agg is not None: params = agg
    gl = _boosters(glow.get_node_model(0))   # nó 0 (todos idênticos após difusão)

    # ── teste ────────────────────────────────────────────────────────────────
    print("[voto] carregando teste...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    yte = (yte_m != nc).astype(int)
    dte = xgb.DMatrix(Xte)

    pred_cen = (cen.predict(dte) >= 0.5).astype(np.int8)
    pred_fl  = _or_predict(fl, dte)
    pred_gl  = _or_predict(gl, dte)

    is_atk, is_ben = yte == 1, yte == 0
    n_atk, n_ben = int(is_atk.sum()), int(is_ben.sum())

    def stats(pred):
        vp = int((pred & is_atk).sum()); fp = int((pred & is_ben).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, prec, fpr, fp

    print(f"\n[voto] teste: {n_atk:,} ataques, {n_ben:,} normais")
    print(f"\n{'arquitetura/regra':<24}{'F1':>7}{'Recall':>8}{'Prec':>7}{'FPR':>7}{'#FP':>9}")
    for nome, pred in [("Centralizado", pred_cen), ("FL", pred_fl), ("GL", pred_gl)]:
        f1, rec, prec, fpr, fp = stats(pred)
        print(f"{nome:<24}{f1:>7.2f}{rec:>8.2f}{prec:>7.2f}{fpr:>7.3f}{fp:>9,}")

    votes = pred_cen.astype(int) + pred_fl + pred_gl
    print()
    for k, desc in [(1, "≥1 (união)"), (2, "≥2 concordam"), (3, "as 3 concordam")]:
        f1, rec, prec, fpr, fp = stats((votes >= k).astype(np.int8))
        print(f"voto k={k} ({desc:<15}){f1:>7.2f}{rec:>8.2f}{prec:>7.2f}{fpr:>7.3f}{fp:>9,}")

    # concordância FL vs GL (esperado: quase idênticos)
    ag = 100 * (pred_fl == pred_gl).mean()
    print(f"\n[voto] concordância FL≡GL por sample: {ag:.2f}%")
    print(f"[voto] Centralizado vs FL: {100*(pred_cen==pred_fl).mean():.2f}%  "
          f"Centralizado vs GL: {100*(pred_cen==pred_gl).mean():.2f}%")


if __name__ == "__main__":
    main()
