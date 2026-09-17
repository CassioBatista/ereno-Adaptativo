#!/usr/bin/env python3
"""Zero-day detection via an unsupervised anomaly detector (Isolation Forest).

STANDALONE experiment (Paper 1 revision, branch paper1-revision) — independent of
v1.1/v2 and of the XGBoost specialists. It measures whether a Tier-2 anomaly detector
trained ONLY on benign ERENO traffic can detect a held-out attack that HAS NO
specialist (a zero-day), where the supervised ensemble floors at ~3.8% incidental
cross-firing (from scripts/new_attack_convergence.py, at the specialists' ~0.6% FPR).

The Isolation Forest never sees any attack; it flags "not benign". We report the
zero-day (held-out class) recall at several benign-FPR operating points, and compare
to the 3.8% cross-firing floor. Uses the same core (util + combined-24 features);
touches nothing in the pipeline.
Out: results/zeroday_anomaly.{csv,png,pdf}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util
from sklearn.ensemble import IsolationForest

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
SEED = 42
NOVEL = 3                     # held-out "zero-day" class (same as new_attack_convergence)
N_FIT = 200_000              # benign samples to fit the IF (subsample for speed)
N_BEN_EVAL = 300_000        # benign test samples for FPR calibration
XFIRE_FLOOR = 3.81          # cross-firing floor at ~0.6% FPR (measured, N=12 run)
FPR_TARGETS = [0.005, 0.006, 0.01, 0.02, 0.05]


def main():
    rng = np.random.default_rng(SEED)
    nc = util.normal_class

    print("[zd] carregando treino (benigno p/ fit)...")
    Xtr_raw, ytr, _ = util.load_arff(f"{DATASET}.csv")
    Xtr = util.filter_features(Xtr_raw, FEATURES); del Xtr_raw
    ben = np.where(ytr == nc)[0]
    fit_idx = rng.permutation(ben)[:N_FIT]
    Xfit = Xtr[fit_idx]; del Xtr, ytr
    print(f"[zd] fit Isolation Forest em {len(Xfit):,} amostras BENIGNAS (sem ataques)...")
    iso = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=SEED, n_jobs=-1).fit(Xfit)
    del Xfit

    print("[zd] carregando teste completo...")
    Xte_raw, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw

    is_ben = (yte == nc)
    is_novel = (yte == NOVEL)
    is_atk = (yte != nc)
    n_novel = int(is_novel.sum())
    print(f"[zd] teste: {len(yte):,} | benigno={int(is_ben.sum()):,} | "
          f"novel({NOVEL})={n_novel:,} | ataque(total)={int(is_atk.sum()):,}")

    # anomaly score: higher = more anomalous
    def score(mask, cap=None):
        idx = np.where(mask)[0]
        if cap and len(idx) > cap:
            idx = rng.permutation(idx)[:cap]
        return -iso.score_samples(Xte[idx])

    print("[zd] pontuando benigno / novel / todos-ataques...")
    s_ben = score(is_ben, N_BEN_EVAL)
    s_novel = score(is_novel)                      # all novel (zero-day) samples
    s_atk = score(is_atk, N_BEN_EVAL)              # all attacks (reference)

    rows = []
    for f in FPR_TARGETS:
        tau = np.quantile(s_ben, 1.0 - f)
        rec_novel = 100.0 * float((s_novel >= tau).mean())
        rec_atk = 100.0 * float((s_atk >= tau).mean())
        fpr_real = 100.0 * float((s_ben >= tau).mean())
        rows.append({"fpr_target_pct": 100 * f, "tau": float(tau),
                     "fpr_real_pct": fpr_real, "zeroday_recall_pct": rec_novel,
                     "all_attacks_recall_pct": rec_atk})
        print(f"[zd] FPR~{100*f:>4.1f}%  τ={tau:+.4f}  zero-day recall={rec_novel:6.2f}%  "
              f"(all-attacks {rec_atk:5.2f}%)   vs cross-firing floor {XFIRE_FLOOR}%")

    os.makedirs("results", exist_ok=True)
    hdr = ["fpr_target_pct", "tau", "fpr_real_pct", "zeroday_recall_pct", "all_attacks_recall_pct"]
    with open("results/zeroday_anomaly.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader()
        for r in rows: w.writerow(r)

    # figure: zero-day recall vs FPR, with the cross-firing floor
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fprs = [r["fpr_real_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(fprs, [r["zeroday_recall_pct"] for r in rows], "-o", color="#1b7837",
            lw=2.4, ms=8, label="Isolation Forest — zero-day recall (class %d)" % NOVEL)
    ax.plot(fprs, [r["all_attacks_recall_pct"] for r in rows], "--s", color="#2166ac",
            lw=1.8, ms=6, label="Isolation Forest — all-attacks recall (ref.)")
    ax.axhline(XFIRE_FLOOR, color="#b2182b", ls=":", lw=2,
               label="supervised cross-firing floor (%.1f%%, no expert)" % XFIRE_FLOOR)
    ax.set_xlabel("benign false-positive rate (%)")
    ax.set_ylabel("recall on the zero-day attack (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("Off-the-shelf anomaly detection (Isolation Forest) on the GRASP features\n"
                 "FAILS to detect this zero-day: recall stays at/below the cross-firing floor",
                 fontsize=11.5)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/zeroday_anomaly.png", dpi=175)
    fig.savefig("results/zeroday_anomaly.pdf")
    print("[zd] ok -> results/zeroday_anomaly.{csv,png,pdf}")


if __name__ == "__main__":
    main()
