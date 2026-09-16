#!/usr/bin/env python3
"""Convergence-time of a NOVEL attack: FL vs GL, k>=1 vs k>=2 (Paper 1 reviewer).

Beyond communication, this is the SECOND axis that justifies FL as the normal
operating mode: TIME TO CONVERGENCE when new knowledge enters the system.

Setup: 12 specialist nodes = 2 per attack over 6 KNOWN attacks (clean k>=2
redundancy); one attack is HELD OUT as the novel one. A DEDICATED novel specialist
is created at node 0 at round 11 (the node observed enough labelled samples of the
novel attack to train an expert). We track per-round recall on the NOVEL attack for
FL-k1, FL-k2, GL-k1, GL-k2.

Three honesty features:
  (1) ZERO-DAY baseline: before any expert exists (r<11) detection is NOT zero —
      it is the incidental cross-firing of the 6 known specialists (c*_known). The
      local LEARNING latency (arrival -> expert ready) is a mode-INDEPENDENT offset
      (FL and GL both wait for the expert), so fixing r11 = "expert ready" isolates
      the PROPAGATION term, which is what differs between FL and GL.
  (2) SYMMETRIC cross-firing: GL-k1/GL-k2 include the same known-specialist cross-
      firing as FL (the knowns are fully diffused, present at every node); a node
      additionally gains the novel expert once it has DIFFUSED to it.
  (3) SECOND SOURCE (companion): with a single source, k>=2 on the novel attack
      cannot be met (RQ4: corroboration needs >=2 experts). A 2nd node learns the
      novel attack at round 16 -> k>=2 activates. This makes RQ4 a TEMPORAL result.

Live: specialists are REAL XGBoost boosters trained on ERENO, evaluated on the REAL
full ERENO test set. GL propagation is the GLow head-election neighbourhood-union
diffusion (GLow is a simulation stand-in in this project; the GL side is modelled by
its diffusion, as elsewhere). Run on the v1.1 lineage (branch paper1-revision).
Out: results/new_attack_convergence_{1src,2src}.{csv,png,pdf}
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
NODES = 12
INJECT_ROUND = 11          # node 0 has the novel expert ready
SECOND_ROUND = 16          # node 6 learns the novel attack (companion)
SECOND_NODE = 6            # far side of the ring -> clear diffusion contrast
FIRST_NODE = 0
ROUNDS = 34
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


def diffuse_step(topo, reached, r, n):
    """One GLow head-election round: the elected head joins `reached` if any
    neighbour already has the item."""
    head = list(range(n))[(r - 1) % n]
    if head not in reached and any(nb in reached for nb in topo.neighbors(head)):
        reached.add(head)


def main():
    rng = np.random.default_rng(SEED)
    nc = util.normal_class

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
    known = attacks[:N_KNOWN_ATTACKS]
    novel = attacks[N_KNOWN_ATTACKS] if len(attacks) > N_KNOWN_ATTACKS else attacks[-1]
    print(f"[conv] KNOWN (6): {known}  |  NOVEL (held out): {novel}")

    node_attack = []
    for a in known:
        node_attack += [a, a]
    node_attack = node_attack[:NODES]

    benign_all = np.where(ytr == nc)[0]
    benign_slices = np.array_split(rng.permutation(benign_all), NODES)

    print("[conv] treinando 12 especialistas base (ataques conhecidos)...")
    base_boost = []
    for cid in range(NODES):
        a = node_attack[cid]
        a_idx = np.where(ytr == a)[0]
        peers = [c for c in range(NODES) if node_attack[c] == a]
        a_slice = np.array_split(rng.permutation(a_idx), len(peers))[peers.index(cid)]
        idx = np.concatenate([a_slice, benign_slices[cid]])
        base_boost.append(train_specialist(Xtr[idx], (ytr[idx] != nc).astype(int)))

    # two DEDICATED novel specialists on DISJOINT novel shards (independent experts)
    print(f"[conv] treinando 2 especialistas dedicados do ataque novo ({novel}) em shards disjuntos...")
    nov_idx = np.where(ytr == novel)[0]
    shardA, shardB = np.array_split(rng.permutation(nov_idx), 2)
    idxA = np.concatenate([shardA, benign_slices[FIRST_NODE]])
    yA = np.r_[np.ones(len(shardA)), np.zeros(len(benign_slices[FIRST_NODE]))].astype(int)
    idxB = np.concatenate([shardB, benign_slices[SECOND_NODE]])
    yB = np.r_[np.ones(len(shardB)), np.zeros(len(benign_slices[SECOND_NODE]))].astype(int)
    nov0 = train_specialist(Xtr[idxA], yA)
    nov6 = train_specialist(Xtr[idxB], yB)
    del Xtr, ytr

    print("[conv] carregando teste completo...")
    Xte_raw, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    dte = xgb.DMatrix(Xte); del Xte
    is_novel = (yte == novel)
    n_novel = int(is_novel.sum())
    print(f"[conv] teste: {len(yte):,} | novel({novel})={n_novel:,}")

    base_pred = [(b.predict(dte) >= 0.5).astype(np.int16) for b in base_boost]
    votes_known = np.sum(base_pred, axis=0)
    nov0_pred = (nov0.predict(dte) >= 0.5).astype(np.int16)
    nov6_pred = (nov6.predict(dte) >= 0.5).astype(np.int16)

    def combo(has0, has6):
        total = votes_known + (nov0_pred if has0 else 0) + (nov6_pred if has6 else 0)
        k1 = 100.0 * int(((total >= 1) & is_novel).sum()) / max(n_novel, 1)
        k2 = 100.0 * int(((total >= 2) & is_novel).sum()) / max(n_novel, 1)
        return k1, k2
    C = {(0, 0): combo(0, 0), (1, 0): combo(1, 0), (0, 1): combo(0, 1), (1, 1): combo(1, 1)}
    print(f"[conv] baselines novel-recall  none(zero-day)={C[(0,0)]}  nov0={C[(1,0)]}  "
          f"nov6={C[(0,1)]}  both={C[(1,1)]}")

    topo = build_topology("ring", NODES)

    def run_scenario(two_source):
        reached0, reached6 = set(), set()
        rows = []
        for r in range(1, ROUNDS + 1):
            if r >= INJECT_ROUND:
                if not reached0:
                    reached0.add(FIRST_NODE)
                else:
                    diffuse_step(topo, reached0, r, NODES)
            if two_source and r >= SECOND_ROUND:
                if not reached6:
                    reached6.add(SECOND_NODE)
                else:
                    diffuse_step(topo, reached6, r, NODES)

            # FL: central pool has every expert that EXISTS this round
            fl = C[(1 if r >= INJECT_ROUND else 0,
                    1 if (two_source and r >= SECOND_ROUND) else 0)]
            # GL: per-node pool; network mean over the 4 combos by node counts
            both = len(reached0 & reached6)
            only0 = len(reached0 - reached6)
            only6 = len(reached6 - reached0)
            none = NODES - both - only0 - only6
            gl_k1 = (none * C[(0, 0)][0] + only0 * C[(1, 0)][0] +
                     only6 * C[(0, 1)][0] + both * C[(1, 1)][0]) / NODES
            gl_k2 = (none * C[(0, 0)][1] + only0 * C[(1, 0)][1] +
                     only6 * C[(0, 1)][1] + both * C[(1, 1)][1]) / NODES
            rows.append({"round": r, "fl_k1": fl[0], "fl_k2": fl[1],
                         "gl_k1": gl_k1, "gl_k2": gl_k2,
                         "reached0": len(reached0), "reached6": len(reached6)})
        return rows

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot(rows, path, title, two_source):
        R = [r["round"] for r in rows]
        fig, ax = plt.subplots(figsize=(9.6, 5.3))
        ax.axvspan(1, INJECT_ROUND, color="0.85", alpha=0.5)
        ax.annotate(f"zero-day baseline\n(no expert:\ncross-firing ~{C[(0,0)][0]:.1f}%)",
                    (1.4, 55), fontsize=8, color="0.4", va="center")
        ax.axvline(INJECT_ROUND, color="0.4", ls=":", lw=1)
        ax.annotate(f"novel expert ready\n@node 0, r{INJECT_ROUND}", (INJECT_ROUND + 0.2, 30),
                    fontsize=8.3, color="0.35")
        if two_source:
            ax.axvline(SECOND_ROUND, color="#6a3d9a", ls=":", lw=1)
            ax.annotate(f"2nd source\n@node 6, r{SECOND_ROUND}", (SECOND_ROUND + 0.2, 12),
                        fontsize=8.3, color="#6a3d9a")
        ax.plot(R, [r["fl_k1"] for r in rows], "-o", color="#b2182b", lw=2.3, ms=5,
                label="FL, k$\\geq$1")
        ax.plot(R, [r["gl_k1"] for r in rows], "-o", color="#1b7837", lw=2.3, ms=5,
                label="GL, k$\\geq$1")
        ax.plot(R, [r["fl_k2"] for r in rows], "--s", color="#d6604d", lw=1.9, ms=4,
                label="FL, k$\\geq$2")
        ax.plot(R, [r["gl_k2"] for r in rows], "--^", color="#74c476", lw=1.9, ms=4,
                label="GL, k$\\geq$2")
        ax.set_xlabel("round"); ax.set_ylabel("recall on the NOVEL attack (%)")
        ax.set_ylim(-3, 103)
        ax.set_title(title, fontsize=11.5)
        ax.grid(True, alpha=0.3); ax.legend(loc="center right", fontsize=9)
        fig.tight_layout()
        fig.savefig(path + ".png", dpi=175); fig.savefig(path + ".pdf")
        hdr = list(rows[0].keys())
        with open(path + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader()
            for r in rows: w.writerow(r)

    os.makedirs("results", exist_ok=True)
    r1 = run_scenario(two_source=False)
    plot(r1, "results/new_attack_convergence_1src",
         "Convergence to a NOVEL attack, SINGLE source ($N{=}12$, 2/attack, ring):\n"
         "FL k$\\geq$1 ~1 round; GL k$\\geq$1 diffuses; k$\\geq$2 stays low (RQ4: needs a 2nd expert)",
         two_source=False)
    r2 = run_scenario(two_source=True)
    plot(r2, "results/new_attack_convergence_2src",
         "Same, with a SECOND source at r16: k$\\geq$2 ACTIVATES (RQ4 in time)\n"
         "FL k$\\geq$2 steps at r16; GL k$\\geq$2 ramps as both experts diffuse",
         two_source=True)

    def conv(rows, key, ceil):
        return next((x["round"] for x in rows if x[key] >= 0.9 * ceil), None)
    ceil1 = C[(1, 0)][0]
    print(f"\n[1src] zero-day k1 baseline={C[(0,0)][0]:.2f}  ceiling k1={ceil1:.2f}")
    print(f"[1src] FL-k1 conv @r{conv(r1,'fl_k1',ceil1)}  GL-k1 conv @r{conv(r1,'gl_k1',ceil1)}")
    print(f"[1src] FL-k2 max={max(x['fl_k2'] for x in r1):.2f}  GL-k2 max={max(x['gl_k2'] for x in r1):.2f} (single source)")
    ceil2 = C[(1, 1)][1]
    print(f"[2src] k2 ceiling(both)={ceil2:.2f}  FL-k2 conv @r{conv(r2,'fl_k2',ceil2)}  GL-k2 conv @r{conv(r2,'gl_k2',ceil2)}")
    print("[conv] ok -> results/new_attack_convergence_{1src,2src}.{csv,png,pdf}")


if __name__ == "__main__":
    main()
