"""Escalabilidade FL vs GL: custo (pool, bytes, voto, difusao) e acuracia
variando N (10..50), combinado-24. Responde ao "high communication cost /
hard to scale" do P2P FL (Yahata): mede como o custo do GL cresce com N e se
a acuracia (OR e k>=2) se mantem. Orcamento fixo de GL_ROUNDS rounds.

FL bagging e invariante a rounds (pool = N sempre); GL difunde em GL_ROUNDS.
Rodar: ~/venv-ereno314/bin/python scripts/escalabilidade.py

Provenance (v1.6): this is the generator of results/escalabilidade.csv (Section 4.5 of the
Paper 1 revision, N = 10..100 as in CLIENT_COUNTS), recovered unchanged from the untracked
files of the original run; it had never been committed. It loads the full ARFF files
through python.util.load_arff, which peaks near 9 GB of RAM (see REPRODUCE.md and the lean
reader of scripts/k_fusion_audit.py). Plotted by scripts/plot_scaling_pair.py.
"""
import os, sys, time, gc
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
SEED, BENIGN_CAP = 42, 500000
TEST_SUB = 200000   # subamostra do teste (proporção natural) p/ caber na RAM + voto limpo
CAP_FACTOR = 4      # teto de rounds GL = CAP_FACTOR*N (difusão a frio, com early-stop)
CLIENT_COUNTS = [10, 14, 20, 30, 50, 70, 100]
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 2}


def _boosters(params):
    out = []
    for a in parameters_to_ndarrays(params):
        if a.size:
            b = xgb.Booster(); b.load_model(bytearray(a.tobytes())); out.append(b)
    return out


def main():
    cols = [c - 1 for c in FEATURES]

    # 1) TEST primeiro: carrega, subamostra p/ TEST_SUB (mantém proporção natural),
    #    libera o bruto. Fica só um Xte pequeno (float32) na RAM.
    print("[esc] carregando TEST (subamostra p/ caber na RAM)...", flush=True)
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    if len(yte_m) > TEST_SUB:
        sel = np.random.default_rng(SEED).choice(len(yte_m), size=TEST_SUB, replace=False)
        Xte = Xte_raw[sel][:, cols].astype(np.float32); yte_m = yte_m[sel].copy()
    else:
        Xte = Xte_raw[:, cols].astype(np.float32)
    del Xte_raw

    # 2) TRAIN sozinho (o bruto do test já foi liberado): pico = só o train.
    print("[esc] carregando TRAIN...", flush=True)
    X_all, y_all_raw, _ = util.load_arff(f"{DATASET}.csv")
    Xf_all = X_all[:, cols].astype(np.float32); del X_all
    nc = util.normal_class

    is_atk = (yte_m != nc); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    dte = xgb.DMatrix(Xte); del Xte
    print(f"[esc] teste (subamostra {len(yte_m):,}): {n_atk:,} ataques, {n_ben:,} normais\n", flush=True)

    def stats(pred):
        vp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        rec = 100 * vp / n_atk; fpr = 100 * fp / n_ben
        prec = 100 * vp / max(vp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return f1, rec, prec, fpr

    def vote_timed(boosters):
        t0 = time.perf_counter()
        v = np.column_stack([(b.predict(dte) >= 0.5).astype(np.int16) for b in boosters]).sum(1)
        return v, (time.perf_counter() - t0) * 1000.0

    rows = []
    for N in CLIENT_COUNTS:
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

        # FL: bagging union (pool = N)
        fl = []
        for Xc, yy in client_tr:
            p, q = int(yy.sum()), int((yy == 0).sum())
            pr = dict(XGB_PARAMS)
            if p and q:
                pr["scale_pos_weight"] = q / p
            fl.append(xgb.train(pr, xgb.DMatrix(Xc, label=yy), num_boost_round=10, verbose_eval=False))
        fl_bytes = int(sum(len(b.save_raw("json")) for b in fl))

        # GL: difusao gossip com medicao de comunicacao
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
        # difusao a frio COM early-stop: para quando o pool do no 0 nao
        # cresce por um ciclo completo de heads (N+1 rounds) -> registra
        # rounds-para-saturar (custo que o ERENO evita ao semear GL do FL).
        gl_comm = 0; gl_comm_bytes = 0
        node0_size = 0; no_growth = 0; rnd = 0; cap = CAP_FACTOR * N
        while rnd < cap:
            rnd += 1
            instr = glow.configure_fit(rnd, params, mgr)
            res = []
            for proxy, ins in instr:
                nds = parameters_to_ndarrays(ins.parameters)
                out, n, m = clients[proxy.cid].fit(nds, dict(ins.config))
                res.append((proxy, FitRes(Status(Code.OK, "ok"),
                            ndarrays_to_parameters(out), n, {**m, "cid": int(proxy.cid)})))
            agg, _ = glow.aggregate_fit(rnd, res, [])
            if agg is not None:
                params = agg
            head = glow._head_for_round(rnd)
            parts = [head] + glow.topology.up_neighbors(head)
            # comunicacao: o head puxa os pools dos vizinhos (contagem + bytes reais)
            for i in parts:
                p = glow.pool_parameters.get(i)
                if p is not None:
                    gl_comm += len(p.tensors)
                    gl_comm_bytes += sum(len(t) for t in p.tensors)
            p0 = glow.get_node_model(0)
            cur = len(p0.tensors) if p0 is not None else 0
            if cur > node0_size:
                node0_size = cur; no_growth = 0
            else:
                no_growth += 1
            if no_growth >= N + 1:
                break
        gl_rounds = rnd
        gl = _boosters(glow.get_node_model(0))
        gl_bytes = int(sum(len(b.save_raw("json")) for b in gl))

        vfl, fl_ms = vote_timed(fl)
        vgl, gl_ms = vote_timed(gl)

        # booster medio (KB), footprint por no (MB), comunicacoes (MB)
        avg_booster_kb = (gl_bytes / max(len(gl), 1)) / 1e3
        fl_comm_bytes = fl_bytes * (N + 1)   # FL 1-shot: N uploads + broadcast da uniao
        r = {"N": N, "flb": len(fl), "glb": len(gl), "gl_rounds": gl_rounds,
             "fl_bytes": fl_bytes, "gl_bytes": gl_bytes, "fl_ms": fl_ms,
             "gl_ms": gl_ms, "gl_comm": gl_comm, "gl_comm_bytes": gl_comm_bytes,
             "fl_comm_bytes": fl_comm_bytes, "avg_kb": avg_booster_kb}
        for k in (1, 2):
            r[f"fl{k}"] = stats(vfl >= k)
            r[f"gl{k}"] = stats(vgl >= k)
            r[f"ag{k}"] = 100 * ((vfl >= k) == (vgl >= k)).mean()
        rows.append(r)
        print(f"[esc] N={N:>3} | GLb={r['glb']:>3}/2N-1={2*N-1} em {gl_rounds}r | "
              f"booster~{avg_booster_kb:.1f}KB foot={gl_bytes/1e6:.2f}MB "
              f"coldComm={gl_comm_bytes/1e6:.1f}MB | "
              f"OR_rec={r['gl1'][1]:.2f} F1={r['gl1'][0]:.2f} k2_F1={r['gl2'][0]:.2f} "
              f"(GL=FL)", flush=True)
        del fl, gl, glow, clients, client_tr, Xf, y, splits, vfl, vgl
        gc.collect()

    print("\n" + "=" * 110)
    print("  Detector em regime (FL-uniao = GL semeado do FL) + difusao a frio do GL")
    print("=" * 110)
    hdr = (f"{'N':>4} | {'GLb':>4}{'rnds':>6} | {'bstKB':>6}{'footMB':>8}{'coldMB':>8}{'flMB':>8} | "
           f"{'GLvote':>8} | {'OR_rec':>7}{'OR_F1':>7}{'k2_F1':>7}  (GL=FL)")
    print(hdr)
    for r in rows:
        print(f"{r['N']:>4} | {r['glb']:>4}{r['gl_rounds']:>6} | "
              f"{r['avg_kb']:>6.1f}{r['gl_bytes']/1e6:>8.2f}{r['gl_comm_bytes']/1e6:>8.1f}"
              f"{r['fl_comm_bytes']/1e6:>8.1f} | {r['gl_ms']:>7.0f}m | "
              f"{r['gl1'][1]:>7.2f}{r['gl1'][0]:>7.2f}{r['gl2'][0]:>7.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/escalabilidade.csv", "w") as fh:
        fh.write("N,FL_b,GL_b,sat_2Nm1,GL_rounds,avg_booster_KB,FL_bytes,GL_bytes,"
                 "GL_comm_bytes,FL_comm_bytes,FL_vote_ms,GL_vote_ms,GL_comm,"
                 "OR_F1_FL,OR_rec_FL,OR_F1_GL,OR_rec_GL,k2_F1_FL,k2_rec_FL,k2_F1_GL,k2_rec_GL,"
                 "OR_FPR_FL,OR_FPR_GL,k2_FPR_GL,agree_OR,agree_k2\n")
        for r in rows:
            fh.write(f"{r['N']},{r['flb']},{r['glb']},{2*r['N']-1},{r['gl_rounds']},"
                     f"{r['avg_kb']:.3f},{r['fl_bytes']},{r['gl_bytes']},"
                     f"{r['gl_comm_bytes']},{r['fl_comm_bytes']},"
                     f"{r['fl_ms']:.1f},{r['gl_ms']:.1f},{r['gl_comm']},"
                     f"{r['fl1'][0]:.4f},{r['fl1'][1]:.4f},{r['gl1'][0]:.4f},{r['gl1'][1]:.4f},"
                     f"{r['fl2'][0]:.4f},{r['fl2'][1]:.4f},{r['gl2'][0]:.4f},{r['gl2'][1]:.4f},"
                     f"{r['fl1'][3]:.4f},{r['gl1'][3]:.4f},{r['gl2'][3]:.4f},{r['ag1']:.4f},{r['ag2']:.4f}\n")
    print("\n[esc] CSV: results/escalabilidade.csv")


if __name__ == "__main__":
    main()
