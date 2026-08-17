#!/usr/bin/env python3
"""Reducao de nos DINAMICA 10->3, k>=2 (e k>=1).
GL: gossip cold-difunde em 10 nos (satura ~2N-1=19), depois DERRUBA nos 10->3
(topology.set_status down) e RE-GOSSIPA entre os sobreviventes a cada passo; mede
o pool REAL do no 0 (sobrevivente) e k-de-n. FL: re-agrega os especialistas
PRESENTES (pool = N ativos).
Saida: results/reducao_nos_dyn.csv
Rodar: ~/venv-ereno314/bin/python scripts/reducao_nos_dyn.py
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
N, SEED = 10, 42
SAT_ROUNDS, PHASE_ROUNDS = 20, 5          # satura em 10 nos; re-gossipa a cada queda
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}


def _boosters(params):
    out = []
    for a in parameters_to_ndarrays(params):
        if a.size:
            b = xgb.Booster(); b.load_model(bytearray(a.tobytes())); out.append(b)
    return out


def main():
    print("[dyn] particionando N=10 (attack)...")
    partitions, X_f, y_all = load_and_partition(
        f"{DATASET}.csv", FEATURES, N, seed=SEED, partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    client_tr = []
    for X_c, y_c in partitions:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        client_tr.append((Xct, (yct != nc).astype(int)))

    print("[dyn] treinando 10 especialistas (FL)...")
    fl = []
    for Xc, yy in client_tr:
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = dict(XGB_PARAMS)
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xc, label=yy), num_boost_round=10, verbose_eval=False))

    topo = build_topology("ring", N)
    glow = GlowStrategy(topo, aggregation="xgb_union")
    clients = {str(i): XgbClient(i, Xc, yy, Xc[:2], yy[:2], XGB_PARAMS)
               for i, (Xc, yy) in enumerate(client_tr)}

    class P:
        def __init__(s, c): s.cid = str(c)
    class M:
        def __init__(s, ps): s._p = {p.cid: p for p in ps}
        def all(s): return dict(s._p)
    mgr = M([P(i) for i in range(N)])
    params = ndarrays_to_parameters([np.array([], dtype=np.uint8)])
    gclock = [0]

    def gossip_round():
        gclock[0] += 1
        instr = glow.configure_fit(gclock[0], params[0] if isinstance(params, list) else params, mgr)
        res = []
        for proxy, ins in instr:
            nds = parameters_to_ndarrays(ins.parameters)
            out, n, m = clients[proxy.cid].fit(nds, dict(ins.config))
            res.append((proxy, FitRes(Status(Code.OK, "ok"),
                                      ndarrays_to_parameters(out), n, {**m, "cid": int(proxy.cid)})))
        agg, _ = glow.aggregate_fit(gclock[0], res, [])
        return agg

    print(f"[dyn] cold-difusao em 10 nos ({SAT_ROUNDS} rounds)...")
    for _ in range(SAT_ROUNDS):
        a = gossip_round()
        if a is not None: params = a

    # ---- carrega teste uma vez ----
    print("[dyn] carregando teste...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    dte = xgb.DMatrix(Xte)
    V_fl = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in fl])

    def stats(pred):
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, prec, fpr

    rows = []
    for n_active in range(10, 2, -1):                       # 10 -> 3
        for node in range(n_active, N):                    # derruba nos de indice alto
            topo.set_status(node, False)
        for _ in range(PHASE_ROUNDS):                      # RE-GOSSIPA entre sobreviventes
            a = gossip_round()
            if a is not None: params = a
        gl = _boosters(glow.get_node_model(0))             # pool REAL do no 0 (sobrevivente)
        votes_gl = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in gl]).sum(1)
        S = list(range(n_active))
        votes_fl = V_fl[:, S].sum(1)
        for k in (1, 2):
            f1f, rf, pcf, fprf = stats(votes_fl >= k)
            f1g, rg, pcg, fprg = stats(votes_gl >= k)
            rows.append((n_active, k, n_active, len(gl),
                         f1f, rf, pcf, fprf, f1g, rg, pcg, fprg))
        print(f"[dyn] N={n_active:2d}  GL pool(real)={len(gl):2d}  "
              f"k2: FL={stats(votes_fl>=2)[0]:.2f}  GL={stats(votes_gl>=2)[0]:.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/reducao_nos_dyn.csv", "w") as fh:
        fh.write("N,k,FL_boosters,GL_boosters,FL_F1,FL_Recall,FL_Prec,FL_FPR,"
                 "GL_F1,GL_Recall,GL_Prec,GL_FPR\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    for k in (1, 2):
        print(f"\n=== k>={k} (DINAMICO) ===")
        print(f"{'N':>2} | {'FLb':>3} {'FL_F1':>7} {'FL_rec':>7} | {'GLb':>3} {'GL_F1':>7} {'GL_rec':>7}")
        for r in rows:
            if r[1] == k:
                print(f"{r[0]:>2} | {r[2]:>3} {r[4]:>7.2f} {r[5]:>7.2f} | "
                      f"{r[3]:>3} {r[8]:>7.2f} {r[9]:>7.2f}")
    print("\n[dyn] CSV: results/reducao_nos_dyn.csv")


if __name__ == "__main__":
    main()
