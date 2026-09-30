#!/usr/bin/env python3
"""Convergence-time of a NOVEL attack: FL vs GL, k>=1 vs k>=2 (reviewer question).

Beyond communication, this measures the SECOND axis that justifies FL as the
normal operating mode: TIME TO CONVERGENCE when new knowledge enters the system.

Setup (decided): 12 specialist nodes = 2 per attack over 6 KNOWN attacks (clean
k>=2 redundancy); one attack is HELD OUT as the novel one. Rounds 1-10 run with
the 6 known attacks. At round 11 the novel attack is INSERTED at ONE node (node 0
retrains as a specialist for its original attack ∪ the novel attack). We then track
per-round recall on the NOVEL attack (and overall) for FL-k1, FL-k2, GL-k1, GL-k2.

Live round-by-round run: specialists are REAL XGBoost boosters trained on ERENO,
evaluated on the REAL full ERENO test set. GL diffusion is the GLow head-election
neighbourhood-union spread of the new booster from node 0 over the ring (GLow is a
simulation stand-in in this project — Flower cannot do real P2P — so the GL side is
modelled by its diffusion, exactly as elsewhere in v2).

Expected (single novel source): FL-k1 steps up at r11 (~1 round); GL-k1 ramps over
~diffusion rounds; k>=2 stays ~0 for the novel attack (needs >=2 sources).
Out: results/new_attack_convergence.{csv,png,pdf}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
import python.util as util
from fd.topology import build_topology

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
SEED = 42
N_KNOWN_ATTACKS = 6
NODES = 12                 # 2 specialists per known attack
INJECT_ROUND = 11
ROUNDS = 30
INJECT_NODE = 0            # the node that starts observing the novel attack at r11
BENIGN_CAP = 500_000
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 4}
NBR = 10


def train_specialist(X, y_bin):
    p, q = int(y_bin.sum()), int((y_bin == 0).sum())
    pr = dict(XGB_PARAMS)
    if p and q:
        pr["scale_pos_weight"] = q / p
    return xgb.train(pr, xgb.DMatrix(X, label=y_bin), num_boost_round=NBR, verbose_eval=False)


def main():
    rng = np.random.default_rng(SEED)
    nc = util.normal_class

    # ---- load train, feature-select, benign_cap ----
    print("[conv] carregando treino...")
    Xtr_raw, ytr, _ = util.load_arff(f"{DATASET}.csv")
    Xtr = util.filter_features(Xtr_raw, FEATURES); del Xtr_raw
    normal_idx = np.where(ytr == nc)[0]
    if len(normal_idx) > BENIGN_CAP:
        drop = rng.permutation(normal_idx)[BENIGN_CAP:]
        keep = np.ones(len(ytr), dtype=bool); keep[drop] = False
        Xtr, ytr = Xtr[keep], ytr[keep]

    attacks = sorted((int(c) for c in np.unique(ytr) if c != nc),
                     key=lambda c: -int((ytr == c).sum()))
    print(f"[conv] {len(attacks)} ataques (por tamanho desc): {attacks}")
    # hold out the SMALLEST attack as the novel one (a fresh, rare signal), keep 6
    known = attacks[:N_KNOWN_ATTACKS]
    novel = attacks[N_KNOWN_ATTACKS] if len(attacks) > N_KNOWN_ATTACKS else attacks[-1]
    print(f"[conv] KNOWN (6): {known}  |  NOVEL (held out): {novel}")

    # ---- assign 2 nodes per known attack; node i -> one known attack ----
    node_attack = []
    for a in known:
        node_attack += [a, a]                     # 2 specialists per attack
    node_attack = node_attack[:NODES]
    print(f"[conv] node->attack: {node_attack}")

    benign_all = np.where(ytr == nc)[0]
    benign_slices = np.array_split(rng.permutation(benign_all), NODES)

    # ---- train the 12 BASE specialists (each: its known attack vs benign) ----
    print("[conv] treinando 12 especialistas base...")
    base_boost = []
    for cid in range(NODES):
        a = node_attack[cid]
        a_idx = np.where(ytr == a)[0]
        # split this attack's samples across the 2 nodes that share it
        peers = [c for c in range(NODES) if node_attack[c] == a]
        share = peers.index(cid)
        a_slice = np.array_split(rng.permutation(a_idx), len(peers))[share]
        idx = np.concatenate([a_slice, benign_slices[cid]])
        yb = (ytr[idx] != nc).astype(int)
        base_boost.append(train_specialist(Xtr[idx], yb))

    # ---- train node 0's AUGMENTED specialist (its attack ∪ NOVEL, vs benign) ----
    print(f"[conv] treinando especialista aumentado do nó {INJECT_NODE} (ataque {node_attack[INJECT_NODE]} ∪ novel {novel})...")
    a0 = node_attack[INJECT_NODE]
    a0_idx = np.where(ytr == a0)[0]
    peers0 = [c for c in range(NODES) if node_attack[c] == a0]
    a0_slice = np.array_split(rng.permutation(a0_idx), len(peers0))[peers0.index(INJECT_NODE)]
    nov_idx = np.where(ytr == novel)[0]
    idx_aug = np.concatenate([a0_slice, nov_idx, benign_slices[INJECT_NODE]])
    yb_aug = (ytr[idx_aug] != nc).astype(int)
    aug_boost = train_specialist(Xtr[idx_aug], yb_aug)
    del Xtr, ytr

    # ---- test set (full, includes the novel attack) ----
    print("[conv] carregando teste completo...")
    Xte_raw, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    dte = xgb.DMatrix(Xte); del Xte
    is_novel = (yte == novel)
    is_atk = (yte != nc)
    n_novel = int(is_novel.sum()); n_atk = int(is_atk.sum()); n_ben = int((~is_atk).sum())
    print(f"[conv] teste: {len(yte):,} amostras | novel({novel})={n_novel:,} | ataque={n_atk:,} | benigno={n_ben:,}")

    # per-booster binary predictions on the test set (columns = nodes)
    base_pred = [(b.predict(dte) >= 0.5).astype(np.int16) for b in base_boost]
    aug_pred = (aug_boost.predict(dte) >= 0.5).astype(np.int16)

    def recall_novel(pred_mask):
        return 100.0 * int((pred_mask & is_novel).sum()) / max(n_novel, 1)

    def f1_overall(pred_mask):
        vp = int((pred_mask & is_atk).sum()); fp = int((pred_mask & ~is_atk).sum())
        rec = 100 * vp / max(n_atk, 1); prec = 100 * vp / max(vp + fp, 1)
        return 2 * prec * rec / max(prec + rec, 1e-9)

    def votes(cols):
        return np.sum(cols, axis=0)

    # GL diffusion of node 0's NEW booster over the ring (head-election union)
    topo = build_topology("ring", NODES)
    reached = {INJECT_NODE}
    def gl_step(r):
        active = list(range(NODES))
        head = active[(r - 1) % NODES]
        if any(nb in reached for nb in topo.neighbors(head)):
            reached.add(head)

    rows = []
    for r in range(1, ROUNDS + 1):
        injected = r >= INJECT_ROUND
        # node 0's column is the augmented booster once injected, else its base
        cols = list(base_pred)
        if injected:
            cols[INJECT_NODE] = aug_pred
            gl_step(r)                              # diffuse the new booster one round

        v = votes(cols)                             # FL: central sees all present nodes
        fl_k1 = v >= 1; fl_k2 = v >= 2

        # GL: each node predicts with ITS pool. Only node 0's booster detects the
        # novel attack; a node contributes that detection iff it has RECEIVED it.
        # Novel-attack recall (network mean) scales with the reached fraction;
        # k>=2 needs two DISTINCT novel-capable boosters -> impossible (1 source).
        frac = (len(reached) / NODES) if injected else 0.0
        gl_k1_novel = recall_novel(aug_pred) * frac
        gl_k2_novel = 0.0                           # single source can never reach k>=2

        rows.append({
            "round": r,
            "fl_k1_novel": recall_novel(fl_k1) if injected else 0.0,
            "fl_k2_novel": recall_novel(fl_k2) if injected else 0.0,   # ~0: single source
            "gl_k1_novel": gl_k1_novel,
            "gl_k2_novel": gl_k2_novel,
            "gl_reached": len(reached) if injected else 0,
            "fl_k1_f1": f1_overall(fl_k1), "fl_k2_f1": f1_overall(fl_k2),
        })
        print(f"[conv] r={r:2d} inj={int(injected)} reached={rows[-1]['gl_reached']:2d}  "
              f"FLk1_nov={rows[-1]['fl_k1_novel']:6.2f} FLk2_nov={rows[-1]['fl_k2_novel']:6.2f}  "
              f"GLk1_nov={rows[-1]['gl_k1_novel']:6.2f} GLk2_nov={rows[-1]['gl_k2_novel']:6.2f}")

    os.makedirs("results", exist_ok=True)
    hdr = list(rows[0].keys())
    with open("results/new_attack_convergence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader()
        for r in rows: w.writerow(r)

    # ---- plot: novel-attack recall convergence ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [r["round"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    ax.axvline(INJECT_ROUND, color="0.4", ls=":", lw=1)
    ax.annotate(f"novel attack inserted\n@round {INJECT_ROUND} (1 node)",
                (INJECT_ROUND + 0.2, 50), fontsize=8.5, color="0.35")
    ax.plot(R, [r["fl_k1_novel"] for r in rows], "-o", color="#b2182b", lw=2.3, ms=5,
            label="FL, k$\\geq$1 (central: ~1-round convergence)")
    ax.plot(R, [r["gl_k1_novel"] for r in rows], "-o", color="#1b7837", lw=2.3, ms=5,
            label="GL, k$\\geq$1 (diffusion ramp)")
    ax.plot(R, [r["fl_k2_novel"] for r in rows], "--s", color="#d6604d", lw=1.8, ms=4,
            label="FL, k$\\geq$2 (~0: single source)")
    ax.plot(R, [r["gl_k2_novel"] for r in rows], "--^", color="#74c476", lw=1.8, ms=4,
            label="GL, k$\\geq$2 (~0: single source)")
    ax.set_xlabel("round"); ax.set_ylabel("recall on the NOVEL attack (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("Convergence to a NOVEL attack inserted at 1 node ($N{=}12$, 2/attack, ring):\n"
                 "FL converges in ~1 round; GL ramps over diffusion; k$\\geq$2 needs a 2nd source",
                 fontsize=11.5)
    ax.grid(True, alpha=0.3); ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/new_attack_convergence.png", dpi=175)
    fig.savefig("results/new_attack_convergence.pdf")
    # convergence latencies
    fl_conv = next((r["round"] for r in rows if r["fl_k1_novel"] > 0.9 * recall_novel(aug_pred)), None)
    gl_conv = next((r["round"] for r in rows if r["gl_k1_novel"] > 0.9 * recall_novel(aug_pred)), None)
    print(f"\n[conv] booster novel-recall ceiling = {recall_novel(aug_pred):.2f}%")
    print(f"[conv] FL-k1 converged @round {fl_conv} (latency {None if fl_conv is None else fl_conv-INJECT_ROUND})")
    print(f"[conv] GL-k1 converged @round {gl_conv} (latency {None if gl_conv is None else gl_conv-INJECT_ROUND})")
    print("[conv] ok -> results/new_attack_convergence.{csv,png,pdf}")


if __name__ == "__main__":
    main()
