"""Convergencia POR NO/CLIENTE — GL (difusao, sobe) vs FL (especialista, plano).

Estende a difusao gossip em processo (como voto_arquiteturas) para avaliar o
pool de CADA no no teste global A CADA round -> 10 curvas GL que sobem do recall
solo ate o recall da rede inteira. Para o FL, avalia os 10 especialistas fixos
(constantes) -> 10 linhas planas parciais. Salva results/conv_por_cliente.csv.

Mensagem: o gossip da a CADA no o que o FL so da ao servidor.
Rodar com venv que tem flwr: ~/venv-ereno314/bin/python scripts/conv_por_cliente.py
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
GL_ROUNDS = 14
TEST_SUB = 400000   # subamostra do teste p/ avaliar por-no/round (recall e' o alvo)
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}
OUT = "results/conv_por_cliente.csv"


def _boosters(params):
    out = []
    for a in parameters_to_ndarrays(params):
        if a.size:
            b = xgb.Booster(); b.load_model(bytearray(a.tobytes())); out.append(b)
    return out


def main():
    rng = np.random.default_rng(SEED)
    print("[conv-cli] particionando...")
    partitions, X_f, y_all = load_and_partition(
        f"{DATASET}.csv", FEATURES, N, seed=SEED, partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    class_of = []
    client_tr = []
    for X_c, y_c in partitions:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        client_tr.append((Xct, (yct != nc).astype(int)))
        atk = [int(v) for v in np.unique(y_c) if int(v) != nc]
        class_of.append(atk[0] if atk else -1)

    print("[conv-cli] carregando teste + subamostra...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    yb = (yte_m != nc).astype(int)
    idx = rng.permutation(len(yb))[:TEST_SUB]
    Xte = util.filter_features(Xte_raw[idx], FEATURES); del Xte_raw
    is_atk = yb[idx] == 1; n_atk = int(is_atk.sum())
    dte = xgb.DMatrix(Xte)

    def recall(boosters):
        if not boosters:
            return 0.0
        pred = (np.max([b.predict(dte) for b in boosters], axis=0) >= 0.5)
        return 100 * int((pred & is_atk).sum()) / n_atk

    # ── FL: 10 especialistas fixos (recall parcial, constante) ───────────────
    print("[conv-cli] treinando FL (10 especialistas)...")
    fl = []
    for Xct, y in client_tr:
        p, q = int(y.sum()), int((y == 0).sum())
        pr = dict(XGB_PARAMS)
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xct, label=y), num_boost_round=10, verbose_eval=False))
    fl_each = [recall([b]) for b in fl]
    fl_agg = recall(fl)
    print(f"[conv-cli] FL por-cliente recall: {[round(r,1) for r in fl_each]}  agg={fl_agg:.2f}")

    # ── GL: difusao round a round, recall de cada no por round ───────────────
    print(f"[conv-cli] simulando GL ({GL_ROUNDS} rounds), avaliando cada no/round...")
    glow = GlowStrategy(build_topology("ring", N), aggregation="xgb_union")
    clients = {str(i): XgbClient(i, Xct, y, Xct[:2], y[:2], XGB_PARAMS)
               for i, (Xct, y) in enumerate(client_tr)}

    class P:
        def __init__(s, c): s.cid = str(c)
    class M:
        def __init__(s, ps): s._p = {p.cid: p for p in ps}
        def all(s): return dict(s._p)
    mgr = M([P(i) for i in range(N)])
    params = ndarrays_to_parameters([np.array([], dtype=np.uint8)])
    gl_node = {c: [] for c in range(N)}
    gl_agg = []
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
        for c in range(N):
            gl_node[c].append(recall(_boosters(glow.get_node_model(c))))
        gl_agg.append(np.mean([gl_node[c][-1] for c in range(N)]))
        print(f"  round {rnd}: GL media dos nos = {gl_agg[-1]:.2f}")

    # ── salva CSV (long format) ──────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("serie,tipo,round,recall\n")
        for rnd in range(1, GL_ROUNDS + 1):
            fh.write(f"FL_agg,fl_agg,{rnd},{fl_agg:.4f}\n")
            fh.write(f"GL_agg,gl_agg,{rnd},{gl_agg[rnd-1]:.4f}\n")
            for c in range(N):
                fh.write(f"FL_cli{c},fl_node,{rnd},{fl_each[c]:.4f}\n")
                fh.write(f"GL_no{c},gl_node,{rnd},{gl_node[c][rnd-1]:.4f}\n")
    print(f"[conv-cli] salvo {OUT}")


if __name__ == "__main__":
    main()
