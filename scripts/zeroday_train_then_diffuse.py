#!/usr/bin/env python3
"""Convergence of a NEW attack = LEARNING (vs data, multi-seed) THEN DISSIPATION.

The convergence of a novel attack starts at the TRAINING of its specialist. In the
graduation scenario a node BUFFERS labelled novel samples over time and retrains, so
the natural learning axis is the NUMBER OF NOVEL SAMPLES accumulated.

  Panel 1 — LEARNING (local, mode-independent): specialist recall vs number of novel
            samples, AVERAGED over N_SEEDS random draws per size (mean +/- std band).
            The averaging removes the single-seed noise of tiny samples so the true
            learning curve (rise then saturation) is visible from n=1.

  Panel 2 — DISSIPATION: once trained (ceiling), the specialist spreads. Network recall
            vs round: FL redistributes at the server (depth-1); GL diffuses (~Theta(N)).

Two NEW attacks, both DETECTED by Tier-2 (coherent autonomous discovery):
  injection      — Tier-2 100%; known ensemble alone ~23% -> needs a specialist.
  random_replay  — Tier-2 ~98%; known ensemble alone ~90% -> specialist tops it up.
Recall is measured only on the novel-class test rows (fast). Real ERENO; GL = GLow
head-election neighbourhood-union diffusion on a ring.
Out: results/zeroday_train_then_diffuse.{png,pdf,csv}
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
NODES = 12
NOVELS = [5, 1]           # injection, random_replay
SIZES = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 250, 400, 650, 1000]
N_SEEDS = 10              # random draws per sample-size (learning curve averaging)
NBR = 10
DISS_ROUNDS = 26
BENIGN_CAP = 500_000
BENIGN_NEG_LEARN = 8_000  # benign negatives for the learning-curve specialists (recall-insensitive)
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "nthread": 4}
FIRST_NODE = 0
CLASS_NAME = {1: "random_replay", 2: "inverse_replay", 3: "masquerade_fake_fault",
              4: "masquerade_fake_normal", 5: "injection", 6: "high_StNum",
              7: "poisoned_high_rate"}
COLOR = {5: "#b2182b", 1: "#2166ac"}


def train(X, y, seed=SEED, nbr=NBR):
    p, q = int(y.sum()), int((y == 0).sum())
    pr = dict(XGB_PARAMS, seed=seed)
    if p and q:
        pr["scale_pos_weight"] = q / p
    return xgb.train(pr, xgb.DMatrix(X, label=y), num_boost_round=nbr, verbose_eval=False)


def diffuse_step(topo, reached, r, n):
    head = (r - 1) % n
    if head not in reached and any(nb in reached for nb in topo.neighbors(head)):
        reached.add(head)


def main():
    rng = np.random.default_rng(SEED)
    nc = util.normal_class

    print("[ttd] carregando treino...")
    Xtr_raw, ytr, _ = util.load_arff(f"{DATASET}.csv")
    Xtr = util.filter_features(Xtr_raw, FEATURES); del Xtr_raw
    nidx = np.where(ytr == nc)[0]
    if len(nidx) > BENIGN_CAP:
        drop = rng.permutation(nidx)[BENIGN_CAP:]
        keep = np.ones(len(ytr), bool); keep[drop] = False
        Xtr, ytr = Xtr[keep], ytr[keep]
    attacks = sorted(int(c) for c in np.unique(ytr) if c != nc)
    benign_idx = np.where(ytr == nc)[0]
    benign_slices = np.array_split(rng.permutation(benign_idx), NODES)

    print("[ttd] carregando teste completo...")
    Xte_raw, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw

    topo = build_topology("ring", NODES)
    learn = {}     # novel -> (sizes, mean, std)
    diss = {}

    for novel in NOVELS:
        name = CLASS_NAME[novel]
        is_novel = (yte == novel)
        dnov = xgb.DMatrix(Xte[is_novel])          # predict only on novel-class test rows (fast)
        n_test = int(is_novel.sum())
        nov_all = np.where(ytr == novel)[0]
        avail = len(nov_all)

        # ---- LEARNING (multi-seed): recall vs #samples, mean +/- std ----
        print(f"[ttd] {name}: curva de aprendizado multi-semente ({N_SEEDS} seeds/tamanho)...")
        means, stds = [], []
        for nsz in SIZES:
            k = min(nsz, avail)
            recs = []
            for s in range(N_SEEDS):
                rs = np.random.default_rng(9973 * novel + 131 * s + nsz)
                pos = rs.permutation(nov_all)[:k]
                neg = rs.permutation(benign_idx)[:BENIGN_NEG_LEARN]
                idx = np.concatenate([pos, neg])
                y = np.r_[np.ones(k), np.zeros(len(neg))].astype(int)
                m = train(Xtr[idx], y, seed=1000 + s)
                pred = (m.predict(dnov) >= 0.5)
                recs.append(100.0 * float(pred.mean()))
            means.append(float(np.mean(recs))); stds.append(float(np.std(recs)))
        learn[novel] = (list(SIZES), means, stds)
        print(f"[ttd]   {name} mean: " + ", ".join(f"{n}:{m:.1f}±{s:.1f}"
              for n, m, s in zip(SIZES, means, stds)))

        # ---- DISSIPATION (deterministic): 6 known + full novel expert ----
        known = [a for a in attacks if a != novel][:6]
        node_attack = []
        for a in known:
            node_attack += [a, a]
        node_attack = node_attack[:NODES]
        votes = np.zeros(n_test, dtype=np.int16)
        for cid in range(NODES):
            a = node_attack[cid]
            a_idx = np.where(ytr == a)[0]
            peers = [c for c in range(NODES) if node_attack[c] == a]
            a_slice = np.array_split(rng.permutation(a_idx), len(peers))[peers.index(cid)]
            idx = np.concatenate([a_slice, benign_slices[cid]])
            b = train(Xtr[idx], (ytr[idx] != nc).astype(int))
            votes += (b.predict(dnov) >= 0.5).astype(np.int16)
        idxN = np.concatenate([nov_all, benign_slices[FIRST_NODE]])
        yN = np.r_[np.ones(avail), np.zeros(len(benign_slices[FIRST_NODE]))].astype(int)
        nov = train(Xtr[idxN], yN)
        nov_pred = (nov.predict(dnov) >= 0.5).astype(np.int16)
        cross = 100.0 * float((votes >= 1).mean())
        ceiling = 100.0 * float(((votes + nov_pred) >= 1).mean())
        rows, reached = [], set()
        for r in range(1, DISS_ROUNDS + 1):
            if not reached:
                reached.add(FIRST_NODE)
            else:
                diffuse_step(topo, reached, r, NODES)
            mrs = len(reached)
            rows.append({"round": r, "fl": ceiling,
                         "gl": (mrs * ceiling + (NODES - mrs) * cross) / NODES})
        gl_full = next((d["round"] for d in rows if d["gl"] >= ceiling - 1e-9), None)
        diss[novel] = dict(rows=rows, cross=cross, ceiling=ceiling, gl_full=gl_full)
        print(f"[ttd]   {name} dissipation: cross={cross:.1f}%  ceiling={ceiling:.1f}%  GL full@r{gl_full}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.5))

    # Panel 1 — learning (mean +/- std band)
    for novel in NOVELS:
        xs, mean, std = learn[novel]
        mean = np.array(mean); std = np.array(std)
        lo = np.clip(mean - std, 0, 100); hi = np.clip(mean + std, 0, 100)
        a1.fill_between(xs, lo, hi, color=COLOR[novel], alpha=0.16)
        a1.plot(xs, mean, "-o", color=COLOR[novel], lw=2.4, ms=5, label=CLASS_NAME[novel])
    a1.set_xscale("log")
    a1.set_xlabel("novel samples accumulated (buffer size) — training progress")
    a1.set_ylabel("recall on the NEW attack (%) — specialist alone")
    a1.set_title(f"1 · LEARNING — training vs accumulated data ({N_SEEDS} seeds, mean ± std)\n"
                 "(local, mode-independent: same for FL and GL)", fontsize=10.2)
    a1.set_ylim(-3, 103); a1.set_xlim(1, 1000); a1.grid(True, which="both", alpha=0.3)
    a1.legend(loc="lower right", fontsize=9.5, title="new attack")

    # Panel 2 — dissipation
    for novel in NOVELS:
        d = diss[novel]; R = [x["round"] for x in d["rows"]]
        a2.axhline(d["cross"], color=COLOR[novel], ls=":", lw=1.0, alpha=0.6)
        a2.plot(R, [x["fl"] for x in d["rows"]], "-", color=COLOR[novel], lw=2.6,
                label=f"{CLASS_NAME[novel]} — FL (depth-1)")
        a2.plot(R, [x["gl"] for x in d["rows"]], "--o", color=COLOR[novel], lw=2.0, ms=4,
                label=f"{CLASS_NAME[novel]} — GL (full @r{d['gl_full']})")
    a2.set_xlabel("round since the specialist is trained")
    a2.set_ylabel("network recall on the NEW attack (%)")
    a2.set_title("2 · DISSIPATION — spreading the trained specialist\n"
                 f"FL ~1 round; GL diffuses over ~Θ(N) ($N{{=}}{NODES}$, ring)", fontsize=10.2)
    a2.set_ylim(-3, 103); a2.set_xlim(1, DISS_ROUNDS); a2.grid(True, alpha=0.3)
    a2.legend(loc="center right", fontsize=8.2)

    fig.suptitle("Convergence of a NEW attack: learning (vs data) THEN dissipation — "
                 "injection vs random_replay", fontsize=11.5)
    fig.text(0.5, 0.015,
             "Both attacks are Tier-2-detectable (coherent autonomous discovery). Learning "
             "latency = collection rounds to a good specialist; dissipation = FL depth-1 vs GL ~Θ(N).",
             ha="center", va="bottom", fontsize=8.2, color="0.4", style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/zeroday_train_then_diffuse.png", dpi=175)
    fig.savefig("results/zeroday_train_then_diffuse.pdf")
    with open("results/zeroday_train_then_diffuse.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attack", "phase", "x", "mean_or_fl", "std_or_gl"])
        for novel in NOVELS:
            xs, mean, std = learn[novel]
            for n, m, s in zip(xs, mean, std):
                w.writerow([CLASS_NAME[novel], "learning_samples", n, f"{m:.3f}", f"{s:.3f}"])
            for d in diss[novel]["rows"]:
                w.writerow([CLASS_NAME[novel], "dissipation_round", d["round"],
                            f"{d['fl']:.3f}", f"{d['gl']:.3f}"])
    print("[ttd] ok -> results/zeroday_train_then_diffuse.{png,pdf,csv}")


if __name__ == "__main__":
    main()
