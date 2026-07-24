"""k-de-n: GL vs FL no simulador ereno-adaptativo (GlowStrategy), combinado-24.

Roda a difusao gossip REAL (GlowStrategy, 20 rounds) -> pool de cada no; e a
uniao federada (10 especialistas). Aplica k>=1, k>=2, k>=3 (contagem de votos)
a CADA um e compara GL vs FL amostra a amostra. Testa se k-de-n (nao-idempotente)
quebra a equivalencia GL=FL (que so vale para OR/k=1, idempotente).

Rodar: ~/venv-ereno314/bin/python scripts/kden_gl_vs_fl.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from flwr.common import (parameters_to_ndarrays, ndarrays_to_parameters,
                         FitRes, Status, Code)

import python.util as util
from fd.dataset import load_and_partition
from fd.client_xgb import XgbClient
from fd.strategy.glow_strategy import GlowStrategy
from fd.topology import build_topology

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
N, SEED, GL_ROUNDS = 10, 42, 20
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}


def _boosters(params):
    out = []
    for a in parameters_to_ndarrays(params):
        if a.size:
            b = xgb.Booster(); b.load_model(bytearray(a.tobytes())); out.append(b)
    return out


def main():
    print("[kden] particionando...")
    partitions, X_f, y_all = load_and_partition(
        f"{DATASET}.csv", FEATURES, N, seed=SEED, partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    client_tr = []
    for X_c, y_c in partitions:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        client_tr.append((Xct, (yct != nc).astype(int)))

    print("[kden] treinando 10 especialistas (FL)...")
    fl = []
    for Xc, yy in client_tr:
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = dict(XGB_PARAMS)
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xc, label=yy), num_boost_round=10, verbose_eval=False))

    print(f"[kden] difusao gossip real (GlowStrategy, {GL_ROUNDS} rounds)...")
    glow = GlowStrategy(build_topology("ring", N), aggregation="xgb_union")
    clients = {str(i): XgbClient(i, Xc, yy, Xc[:2], yy[:2], XGB_PARAMS)
               for i, (Xc, yy) in enumerate(client_tr)}

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
    gl = _boosters(glow.get_node_model(0))

    print("[kden] carregando teste...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    dte = xgb.DMatrix(Xte)

    # matriz de votos (boosters x amostras) para FL e GL
    V_fl = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in fl])
    V_gl = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in gl])
    votes_fl = V_fl.sum(1); votes_gl = V_gl.sum(1)
    print(f"[kden] boosters: FL={len(fl)}  GL(pool)={len(gl)}")

    def stats(pred):
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, prec, fpr, fp

    print(f"\n{'regra':<8}{'arq':<4}{'F1':>8}{'Recall':>8}{'Prec':>7}{'FPR':>8}{'#FP':>9}   concord.FL≡GL")
    for k in (1, 2, 3):
        pf = votes_fl >= k; pg = votes_gl >= k
        f1f, rf, pcf, fprf, fpf = stats(pf)
        f1g, rg, pcg, fprg, fpg = stats(pg)
        ag = 100 * (pf == pg).mean(); nd = int((pf != pg).sum())
        print(f"k≥{k:<6}{'FL':<4}{f1f:>8.2f}{rf:>8.2f}{pcf:>7.2f}{fprf:>8.3f}{fpf:>9,}")
        print(f"{'':<8}{'GL':<4}{f1g:>8.2f}{rg:>8.2f}{pcg:>7.2f}{fprg:>8.3f}{fpg:>9,}   {ag:.4f}% ({nd:,} div.)")


if __name__ == "__main__":
    main()
