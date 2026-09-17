#!/usr/bin/env python3
"""Zero-day detection via a SPECIFICATION / protocol-invariant detector (per-attack sweep).

TRUE zero-day (zero attack examples): the detector is built ONLY from benign IEC-61850
behaviour on the protocol-invariant fields and flags VIOLATIONS. Novel attacks that
break the protocol (replay reuses StNum; injection breaks SqNum; high_StNum; rate
attacks) should be caught with zero examples; masquerade (mimics legitimate traffic)
is expected to remain hard for everyone.

Two zero-example detectors on the protocol fields (F40 SqNum, F41 StNum, F42 cbStatus,
F44 gooseTimeAllowedtoLive, F57 timeFromLastChange):
  (1) robust-envelope (learned specification): per-field robust z = |x-median|/(1.4826*MAD)
      on benign; score = max over fields; threshold at a benign-FPR quantile.
  (2) Isolation Forest restricted to the protocol fields (vs. the earlier IF on all 24
      GRASP features, which failed).
Prints per-class inspection stats, then a PER-ATTACK recall table at benign FPR ~1%.
Compares to the supervised cross-firing floor (~3.8%). Standalone; touches nothing.
Out: results/zeroday_spec.{csv,png,pdf}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util
from sklearn.ensemble import IsolationForest

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
PROTO_1BASED = [40, 41, 42, 44, 57]          # SqNum, StNum, cbStatus, TTL, timeFromLastChange
PROTO = [c - 1 for c in PROTO_1BASED]        # 0-based columns in the FULL feature matrix
PROTO_NAME = {39: "F40 SqNum", 40: "F41 StNum", 41: "F42 cbStatus",
              43: "F44 TTL", 56: "F57 timeFromLastChange"}
SEED = 42
N_FIT = 200_000
N_BEN_EVAL = 300_000
XFIRE_FLOOR = 3.81
FPR_TARGET = 0.01


def main():
    rng = np.random.default_rng(SEED)

    print("[spec] carregando treino (todas as 58 features)...")
    Xtr, ytr, cvals = util.load_arff(f"{DATASET}.csv")     # X = ALL features (unfiltered)
    nc = util.normal_class
    print(f"[spec] classes (idx→nome): {list(enumerate(cvals))}")
    print(f"[spec] normal_class idx = {nc} ({cvals[nc]})")

    # ---- inspection: per-class stats on the protocol fields ----
    print("\n[spec] === inspeção: mediana [IQR] por classe nos campos de protocolo ===")
    for col in PROTO:
        line = f"  {PROTO_NAME[col]:>24}: "
        for c in range(len(cvals)):
            v = Xtr[ytr == c, col]
            if len(v) == 0:
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            line += f"{cvals[c][:10]}={med:.3g}[{q1:.3g},{q3:.3g}]  "
        print(line)

    ben_tr = np.where(ytr == nc)[0]
    fit_idx = rng.permutation(ben_tr)[:N_FIT]
    Xb = Xtr[fit_idx][:, PROTO]

    # detector 1: robust envelope on benign
    med = np.median(Xb, axis=0)
    mad = np.median(np.abs(Xb - med), axis=0) * 1.4826 + 1e-9
    def env_score(M):
        return np.max(np.abs(M[:, PROTO] - med) / mad, axis=1)

    # detector 2: Isolation Forest on protocol fields only
    print("\n[spec] fit Isolation Forest (protocol fields) em benigno...")
    iso = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=SEED, n_jobs=-1).fit(Xb)
    def iso_score(M):
        return -iso.score_samples(M[:, PROTO])
    del Xtr, ytr

    print("[spec] carregando teste...")
    Xte, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc

    is_ben = (yte == nc)
    def cap(mask, n):
        idx = np.where(mask)[0]
        return rng.permutation(idx)[:n] if len(idx) > n else idx
    ben_eval = cap(is_ben, N_BEN_EVAL)

    for name, scorer in [("robust-envelope (spec)", env_score),
                         ("isolation-forest (proto)", iso_score)]:
        s_ben = scorer(Xte[ben_eval])
        tau = np.quantile(s_ben, 1.0 - FPR_TARGET)
        fpr = 100.0 * float((s_ben >= tau).mean())
        print(f"\n[spec] --- {name} --- (FPR~{fpr:.2f}% @ τ={tau:.4g}) ---")
        rows = []
        for c in range(len(cvals)):
            if c == nc:
                continue
            idx = np.where(yte == c)[0]
            rec = 100.0 * float((scorer(Xte[idx]) >= tau).mean()) if len(idx) else float("nan")
            rows.append((cvals[c], rec, len(idx)))
            tag = "  <-- masquerade (hard)" if "masquerade" in cvals[c] else ""
            print(f"[spec]   {cvals[c]:>22}: recall={rec:6.2f}%  (n={len(idx):>6}){tag}")
        # keep for CSV/plot (last scorer wins the file naming; store both)
        globals()[f"_rows_{name.split()[0]}"] = (rows, fpr)

    # ---- outputs (both detectors) ----
    env_rows, env_fpr = globals()["_rows_robust-envelope"]
    iso_rows, iso_fpr = globals()["_rows_isolation-forest"]
    os.makedirs("results", exist_ok=True)
    with open("results/zeroday_spec.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attack", "n", "envelope_recall_pct", "iso_proto_recall_pct",
                    "env_fpr_pct", "iso_fpr_pct", "xfire_floor_pct"])
        for (a, er, n), (_, ir, _n) in zip(env_rows, iso_rows):
            w.writerow([a, n, f"{er:.2f}", f"{ir:.2f}", f"{env_fpr:.2f}", f"{iso_fpr:.2f}", XFIRE_FLOOR])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    attacks = [a for a, _, _ in env_rows]
    x = np.arange(len(attacks))
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.bar(x - 0.2, [r for _, r, _ in env_rows], 0.38, color="#1b7837",
           label="robust-envelope (learned spec)")
    ax.bar(x + 0.2, [r for _, r, _ in iso_rows], 0.38, color="#2166ac",
           label="Isolation Forest (protocol fields)")
    ax.axhline(XFIRE_FLOOR, color="#b2182b", ls=":", lw=2,
               label="supervised cross-firing floor (%.1f%%)" % XFIRE_FLOOR)
    ax.set_xticks(x); ax.set_xticklabels(attacks, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("zero-day recall (%) @ benign FPR ~1%")
    ax.set_ylim(0, 105)
    ax.set_title("Zero-day detection by protocol specification (zero attack examples):\n"
                 "per-attack held-out recall — invariant-violating attacks caught; "
                 "masquerade stays hard", fontsize=11.5)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/zeroday_spec.png", dpi=170); fig.savefig("results/zeroday_spec.pdf")
    print("\n[spec] ok -> results/zeroday_spec.{csv,png,pdf}")


if __name__ == "__main__":
    main()
