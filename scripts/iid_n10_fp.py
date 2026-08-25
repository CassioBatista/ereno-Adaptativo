#!/usr/bin/env python3
"""RAW IID (N=10) com #FP exato: FL(10) e GL(pool cold 2N-1=19), k in {1,2}.
Para as linhas IID da tabela de particionamento (attack vs iid).
Saida: results/iid_n10_fp.csv com config_id + #FP.
Rodar: ~/venv-ereno314/bin/python scripts/iid_n10_fp.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes, Status, Code
import python.util as util
from fd.dataset import load_and_partition
from fd.client_xgb import XgbClient
from fd.strategy.glow_strategy import GlowStrategy
from fd.topology import build_topology

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
N, SEED, GL_ROUNDS = 10, 42, 20
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 2}


def _boosters(params):
    out = []
    for a in parameters_to_ndarrays(params):
        if a.size:
            b = xgb.Booster(); b.load_model(bytearray(a.tobytes())); out.append(b)
    return out


def main():
    print("[iid] particionando N=10 (IID)...")
    partitions, _, _ = load_and_partition(f"{DATASET}.csv", FEATURES, N, seed=SEED,
                                          partitioner="iid", benign_cap=500000)
    nc = util.normal_class
    client_tr = []
    for X_c, y_c in partitions:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        client_tr.append((Xct, (yct != nc).astype(int)))

    print("[iid] treinando 10 generalistas (FL)...")
    fl = []
    for Xc, yy in client_tr:
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = dict(XGB_PARAMS)
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xc, label=yy), num_boost_round=10, verbose_eval=False))

    print(f"[iid] difusao gossip fria ({GL_ROUNDS} rounds)...")
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
    print(f"[iid] boosters: FL=10  GL(pool)={len(gl)}")

    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    dte = xgb.DMatrix(Xte)
    V_fl = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in fl]).sum(1)
    V_gl = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in gl]).sum(1)

    def stats(pred):
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, prec, fpr, fp, vp

    rows = []
    for k in (1, 2):
        for arch, votes, pool in (("FL", V_fl, 10), ("GL", V_gl, len(gl))):
            f1, rec, prec, fpr, fp, vp = stats(votes >= k)
            cid = f"ereno-{arch.lower()}-iid-k{k}-n10"
            rows.append((cid, k, arch, pool, f1, rec, prec, fpr, fp, vp, n_atk - vp))

    os.makedirs("results", exist_ok=True)
    with open("results/iid_n10_fp.csv", "w") as fh:
        fh.write("config_id,k,arch,pool,F1,Recall,Prec,FPR,FP,TP,FN\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    print(f"\n{'config_id':<24}{'k':>3}{'arch':>5}{'F1':>8}{'Rec':>8}{'FPR':>8}{'#FP':>9}")
    for r in rows:
        print(f"{r[0]:<24}{r[1]:>3}{r[2]:>5}{r[4]:>8.2f}{r[5]:>8.2f}{r[7]:>8.3f}{r[8]:>9,}")
    print("\n[iid] CSV: results/iid_n10_fp.csv")


if __name__ == "__main__":
    main()
