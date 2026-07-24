"""k-de-n GL vs FL variando o numero de clientes (10..3), combinado-24.

Carrega train+test UMA vez; para cada N replica benign_cap + _attack_per_client
(mesma sequencia RNG do load_and_partition), treina FL, difunde GL (GlowStrategy
anel de N nos), e mede OR/k>=2/k>=3 para FL e GL + concordancia GL=FL.
Mostra como o numero de clientes muda: tamanho do pool, o sweet-spot GL-k>=2, e a
divergencia GL!=FL do k-de-n.

Rodar: ~/venv-ereno314/bin/python scripts/kden_vary_clients.py
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
from fd.dataset import _attack_per_client
from fd.client_xgb import XgbClient
from fd.strategy.glow_strategy import GlowStrategy
from fd.topology import build_topology

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
SEED, GL_ROUNDS, BENIGN_CAP = 42, 20, 500000
CLIENT_COUNTS = [10, 9, 8, 7, 6, 5, 4, 3]
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}


def _boosters(params):
    out = []
    for a in parameters_to_ndarrays(params):
        if a.size:
            b = xgb.Booster(); b.load_model(bytearray(a.tobytes())); out.append(b)
    return out


def main():
    print("[vary] carregando train + test (uma vez)...")
    X_all, y_all_raw, _ = util.load_arff(f"{DATASET}.csv")
    cols = [c - 1 for c in FEATURES]
    Xf_all = X_all[:, cols]; del X_all
    nc = util.normal_class

    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = Xte_raw[:, cols]; del Xte_raw
    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    dte = xgb.DMatrix(Xte)
    print(f"[vary] teste: {n_atk:,} ataques, {n_ben:,} normais\n")

    def stats(pred):
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, prec, fpr, fp

    rows = []
    for N in CLIENT_COUNTS:
        # replica load_and_partition: rng -> benign_cap -> _attack_per_client
        rng = np.random.default_rng(SEED)
        normal_idx = np.where(y_all_raw == nc)[0]
        drop = rng.permutation(normal_idx)[BENIGN_CAP:]
        keep = np.ones(len(y_all_raw), dtype=bool); keep[drop] = False
        Xf, y = Xf_all[keep], y_all_raw[keep]
        splits = _attack_per_client(Xf, y, N, rng)
        client_tr = []
        for idx in splits:
            Xc, yc = Xf[idx], y[idx]
            Xct, _, yct, _ = train_test_split(Xc, yc, test_size=0.2, random_state=SEED)
            client_tr.append((Xct, (yct != nc).astype(int)))

        fl = []
        for Xc, yy in client_tr:
            p, q = int(yy.sum()), int((yy == 0).sum())
            pr = dict(XGB_PARAMS)
            if p and q: pr["scale_pos_weight"] = q / p
            fl.append(xgb.train(pr, xgb.DMatrix(Xc, label=yy), num_boost_round=10, verbose_eval=False))

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

        vfl = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in fl]).sum(1)
        vgl = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in gl]).sum(1)

        r = {"N": N, "flb": len(fl), "glb": len(gl)}
        for k in (1, 2, 3):
            pf, pg = vfl >= k, vgl >= k
            r[f"fl{k}"] = stats(pf); r[f"gl{k}"] = stats(pg)
            r[f"ag{k}"] = 100 * (pf == pg).mean()
        rows.append(r)
        f1g2, rg2, _, fprg2, _ = r["gl2"]
        print(f"[vary] N={N:>2}  FL={r['flb']:>2} GL={r['glb']:>2} boosters | "
              f"OR F1={r['gl1'][0]:.2f} FPR={r['gl1'][3]:.3f} | "
              f"GL-k2 F1={f1g2:.2f} rec={rg2:.2f} FPR={fprg2:.3f} | "
              f"agree k2={r['ag2']:.2f}% k3={r['ag3']:.2f}%")

    # ── tres tabelas: k>=1, k>=2, k>=3 ───────────────────────────────────────
    for k in (1, 2, 3):
        print(f"\n{'='*74}\n  k >= {k}\n{'='*74}")
        print(f"{'N':>3}{'FL_b':>5}{'GL_b':>5} | {'F1_FL':>7}{'F1_GL':>7} | "
              f"{'Rec_FL':>7}{'Rec_GL':>7} | {'Prc_FL':>7}{'Prc_GL':>7} | "
              f"{'FPR_FL':>7}{'FPR_GL':>7} | {'GL=FL':>8}")
        for r in rows:
            f1f, rf, pcf, fprf, _ = r[f"fl{k}"]
            f1g, rg, pcg, fprg, _ = r[f"gl{k}"]
            print(f"{r['N']:>3}{r['flb']:>5}{r['glb']:>5} | {f1f:>7.2f}{f1g:>7.2f} | "
                  f"{rf:>7.2f}{rg:>7.2f} | {pcf:>7.2f}{pcg:>7.2f} | "
                  f"{fprf:>7.3f}{fprg:>7.3f} | {r[f'ag{k}']:>7.2f}%")

    # ── CSV (long) ───────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/kden_vary_clients.csv", "w") as fh:
        fh.write("k,N,FL_b,GL_b,arq,F1,Recall,Prec,FPR,nFP,agree\n")
        for r in rows:
            for k in (1, 2, 3):
                for arq in ("fl", "gl"):
                    f1, rc, pc, fpr, fp = r[f"{arq}{k}"]
                    fh.write(f"{k},{r['N']},{r['flb']},{r['glb']},{arq.upper()},"
                             f"{f1:.4f},{rc:.4f},{pc:.4f},{fpr:.4f},{fp},{r[f'ag{k}']:.4f}\n")
    print("\n[vary] CSV: results/kden_vary_clients.csv")


if __name__ == "__main__":
    main()
