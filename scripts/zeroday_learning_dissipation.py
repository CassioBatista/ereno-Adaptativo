#!/usr/bin/env python3
"""Zero-day LEARNING + DISSIPATION curve for FL vs GL (Paper 1, two-tier).

Two panels, one per zero-day TYPE, each with two phases:

  LEARNING  (arrival -> a specialist is graduated): a LOCAL latency, the SAME for
            FL and GL (both wait for the expert to be trained). During this gap
            Tier-1 sees only incidental cross-firing of the known specialists; the
            two-tier system is held up (or not) by Tier-2, the benign protocol
            spec, which is immediate and mode-independent.

  DISSIPATION (graduated specialist propagates): where FL and GL DIFFER. FL unites
            at the server -> every node has the expert in ~1 round (depth-1). GL
            diffuses epidemically over the overlay -> recall ramps over ~Theta(N).

Key honest contrast (the point of the two panels):
  * protocol-violating zero-day (e.g. high_StNum): Tier-2 already covers the gap
    (~100%), so recall is high from t=0; FL-vs-GL dissipation adds precise attack
    typing but recall was never exposed.
  * stealthy masquerade zero-day: Tier-2 CANNOT cover it (~2-3%, irreducible), so
    the network is EXPOSED throughout learning -> the dissipation SPEED is what
    matters, and FL (depth-1) closes the exposure far faster than GL (~Theta(N)).

Everything measured: base + novel specialists are REAL XGBoost boosters trained on
ERENO and scored on the REAL full ERENO test set; Tier-2 is the real SpecDetector
fit on benign; GL propagation is the GLow head-election neighbourhood-union diffusion.
Out: results/zeroday_learning_dissipation.{png,pdf} + _<name>.csv per case
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
import python.util as util
from fd.topology import build_topology
from fd.spec_detector import SpecDetector, fields_from_feature_list

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
SEED = 42
NODES = 12
GRADUATE_ROUND = 8
ROUNDS = 28
BENIGN_CAP = 500_000
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 4}
NBR = 10
FIRST_NODE = 0
# ERENO class encoding (util.load_arff, by @attribute order)
CLASS_NAME = {1: "random_replay", 2: "inverse_replay", 3: "masquerade_fake_fault",
              4: "masquerade_fake_normal", 5: "injection", 6: "high_StNum",
              7: "poisoned_high_rate"}
# the two zero-day cases: (class, short label, subtitle, autonomous?)
# injection: cross-firing leaves 23% but Tier-2 catches 100% -> Tier-2 genuinely fills
# the gap AND triggers autonomous graduation. masquerade_fake_fault: cross 3.8% / Tier-2
# 2.6% -> nobody covers it, so there is no autonomous discovery; only the operator path.
CASES = [(5, "injection",
          "protocol-violating → Tier-2 fills the gap; graduation is AUTONOMOUS (Path B)", True),
         (3, "masquerade_fake_fault",
          "stealthy → Tier-2 blind; convergence only via the OPERATOR (Path A)", False)]


def train_specialist(X, y_bin):
    p, q = int(y_bin.sum()), int((y_bin == 0).sum())
    pr = dict(XGB_PARAMS)
    if p and q:
        pr["scale_pos_weight"] = q / p
    return xgb.train(pr, xgb.DMatrix(X, label=y_bin), num_boost_round=NBR, verbose_eval=False)


def diffuse_step(topo, reached, r, n):
    head = list(range(n))[(r - 1) % n]
    if head not in reached and any(nb in reached for nb in topo.neighbors(head)):
        reached.add(head)


def main():
    rng = np.random.default_rng(SEED)
    nc = util.normal_class

    print("[ld] carregando treino...")
    Xtr_raw, ytr, _ = util.load_arff(f"{DATASET}.csv")
    Xtr = util.filter_features(Xtr_raw, FEATURES); del Xtr_raw
    normal_idx = np.where(ytr == nc)[0]
    if len(normal_idx) > BENIGN_CAP:
        drop = rng.permutation(normal_idx)[BENIGN_CAP:]
        keep = np.ones(len(ytr), dtype=bool); keep[drop] = False
        Xtr, ytr = Xtr[keep], ytr[keep]
    all_attacks = sorted(int(c) for c in np.unique(ytr) if c != nc)
    print(f"[ld] attacks presentes: {all_attacks}")

    print("[ld] ajustando Tier-2 (SpecDetector) no benigno...")
    fields = fields_from_feature_list(FEATURES)
    det = SpecDetector(fields).fit(Xtr[ytr == nc])
    print(f"[ld]   fields (colunas no vetor filtrado): {fields}")

    benign_all = np.where(ytr == nc)[0]
    benign_slices = np.array_split(rng.permutation(benign_all), NODES)

    print("[ld] carregando teste completo...")
    Xte_raw, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    is_ben = (yte == nc)
    t2_fpr = 100.0 * float(det.flag(Xte[is_ben]).mean())
    # Tier-2 recall per (zero-day) class, real
    t2_recall = {}
    for cls, _, _ in CASES:
        m = (yte == cls)
        t2_recall[cls] = 100.0 * float(det.flag(Xte[m]).mean()) if m.any() else float("nan")
    dte = xgb.DMatrix(Xte); del Xte
    print(f"[ld] Tier-2 benign FPR={t2_fpr:.3f}%  recall: " +
          ", ".join(f"{CLASS_NAME[c]}={t2_recall[c]:.1f}%" for c, _, _ in CASES))

    topo = build_topology("ring", NODES)

    def run_case(novel):
        known = [a for a in all_attacks if a != novel][:6]
        node_attack = []
        for a in known:
            node_attack += [a, a]
        node_attack = node_attack[:NODES]
        base_boost = []
        for cid in range(NODES):
            a = node_attack[cid]
            a_idx = np.where(ytr == a)[0]
            peers = [c for c in range(NODES) if node_attack[c] == a]
            a_slice = np.array_split(rng.permutation(a_idx), len(peers))[peers.index(cid)]
            idx = np.concatenate([a_slice, benign_slices[cid]])
            base_boost.append(train_specialist(Xtr[idx], (ytr[idx] != nc).astype(int)))
        nov_idx = np.where(ytr == novel)[0]
        idxN = np.concatenate([nov_idx, benign_slices[FIRST_NODE]])
        yN = np.r_[np.ones(len(nov_idx)), np.zeros(len(benign_slices[FIRST_NODE]))].astype(int)
        nov0 = train_specialist(Xtr[idxN], yN)

        is_novel = (yte == novel); n_novel = int(is_novel.sum())
        votes_known = np.sum([(b.predict(dte) >= 0.5).astype(np.int16) for b in base_boost], axis=0)
        nov_pred = (nov0.predict(dte) >= 0.5).astype(np.int16)
        base_cross = 100.0 * int(((votes_known >= 1) & is_novel).sum()) / max(n_novel, 1)
        with_expert = 100.0 * int((((votes_known + nov_pred) >= 1) & is_novel).sum()) / max(n_novel, 1)
        t2 = t2_recall[novel]

        rows, reached = [], set()
        for r in range(1, ROUNDS + 1):
            if r >= GRADUATE_ROUND:
                if not reached:
                    reached.add(FIRST_NODE)
                else:
                    diffuse_step(topo, reached, r, NODES)
            nreach = len(reached)
            t1_fl = with_expert if r >= GRADUATE_ROUND else base_cross
            t1_gl = (nreach * with_expert + (NODES - nreach) * base_cross) / NODES
            rows.append({"round": r, "reached": nreach, "tier2": t2,
                         "t1_fl": t1_fl, "t1_gl": t1_gl,
                         "fl": max(t2, t1_fl), "gl": max(t2, t1_gl)})
        gl_conv = next((x["round"] for x in rows if x["gl"] >= 0.9 * with_expert), None)
        print(f"[ld] {CLASS_NAME[novel]:>22}: T2={t2:.1f}%  cross={base_cross:.1f}%  "
              f"ceiling={with_expert:.1f}%  FL conv r{GRADUATE_ROUND}  GL conv r{gl_conv}")
        return rows, dict(base_cross=base_cross, with_expert=with_expert, t2=t2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.9), sharey=True)

    for ax, (novel, name, subtitle, autonomous) in zip(axes, CASES):
        rows, st = run_case(novel)
        R = [x["round"] for x in rows]
        # learning / exposure band (before the specialist exists)
        band_col = "#f2e8c9" if autonomous else "#f6d5cf"
        ax.axvspan(1, GRADUATE_ROUND, color=band_col, alpha=0.6)
        if autonomous:
            ax.text((1 + GRADUATE_ROUND) / 2, 55, "LEARNING\n(local, autonomous;\nTier-2 covers)",
                    ha="center", va="center", fontsize=7.8, color="#8a6d1a")
            trig = f"graduated — AUTONOMOUS\n(Tier-2 → Path B), r{GRADUATE_ROUND}"
        else:
            ax.text((1 + GRADUATE_ROUND) / 2, 55, "EXPOSED\n(no autonomous\ndiscovery)",
                    ha="center", va="center", fontsize=7.8, color="#b2182b")
            trig = f"labelled by OPERATOR\n(external, Path A) — r{GRADUATE_ROUND}\ntiming not autonomous"
        ax.axvline(GRADUATE_ROUND, color="0.4", ls=":", lw=1.2)
        ax.annotate(trig, (GRADUATE_ROUND + 0.3, 26), fontsize=7.6, color="0.25")
        ax.text((GRADUATE_ROUND + ROUNDS) / 2 + 1, 62, "DISSIPATION\nFL depth-1 · GL ~Θ(N)",
                ha="center", va="center", fontsize=8.0, color="#333")
        # Tier-2 spec floor (immediate, mode-independent)
        ax.axhline(st["t2"], color="#6a3d9a", ls=(0, (6, 3)), lw=1.8)
        ax.text(ROUNDS - 0.3, st["t2"] + (2 if st["t2"] < 90 else -7),
                f"Tier-2 (spec): {st['t2']:.0f}%", ha="right", va="bottom",
                fontsize=8.4, color="#6a3d9a")
        # Tier-1-only cross-firing floor
        ax.axhline(st["base_cross"], color="0.6", ls=":", lw=1.1)
        ax.text(1.3, st["base_cross"] + 2, f"Tier-1 cross-firing: {st['base_cross']:.0f}%",
                fontsize=7.6, color="0.5")
        # Tier-1-only dissipation (the specialist spreading), thin dashed
        ax.plot(R, [x["t1_fl"] for x in rows], "--", color="#b2182b", lw=1.3, alpha=0.7,
                label="Tier-1 only, FL")
        ax.plot(R, [x["t1_gl"] for x in rows], "--", color="#1b7837", lw=1.3, alpha=0.7,
                label="Tier-1 only, GL")
        # two-tier total (what the deployed system detects), thick solid
        ax.plot(R, [x["fl"] for x in rows], "-o", color="#b2182b", lw=2.6, ms=4.5,
                label="two-tier, FL")
        ax.plot(R, [x["gl"] for x in rows], "-o", color="#1b7837", lw=2.6, ms=4.5,
                label="two-tier, GL")
        ax.set_xlabel("round"); ax.set_xlim(1, ROUNDS); ax.set_ylim(-3, 103)
        ax.set_title(f"{name}\n{subtitle}", fontsize=10.0)
        ax.grid(True, alpha=0.3); ax.legend(loc="center right", fontsize=8.2)
        with open(f"results/zeroday_learning_dissipation_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
            for x in rows: w.writerow(x)

    axes[0].set_ylabel("zero-day detection recall — network (%)")
    fig.suptitle("Zero-day: learning + information dissipation, FL vs GL "
                 f"($N{{=}}{NODES}$, ring) — Tier-2 self-heals protocol violations; "
                 "stealthy masquerade needs the operator, and only then does FL dissipate faster",
                 fontsize=10.8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/zeroday_learning_dissipation.png", dpi=170)
    fig.savefig("results/zeroday_learning_dissipation.pdf")
    print("[ld] ok -> results/zeroday_learning_dissipation.{png,pdf} + per-case csv")


if __name__ == "__main__":
    main()
