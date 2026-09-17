#!/usr/bin/env python3
"""Extend the explicit benign-spec to ALL features to find the rate/sequence signal.

The per-field VALUE rules on the 5 protocol fields miss poisoned_high_rate (a RATE
attack) and inverse_replay (an ORDER attack). This script learns one benign rule per
feature over ALL 58 features (allowed-set for near-constant, robust range otherwise),
flags a packet on ANY violation, and — for the two missed attacks — prints WHICH
features fire (with their benign FP). If ERENO already encodes rate/sequence deltas in
some per-sample feature, a benign-range rule on it IS the rate/sequence rule; if not,
true stateful reconstruction over an ordered per-source stream would be needed.
Standalone (paper1-revision); zero attack examples (benign-only rules).
Out: results/zeroday_rules_full.{csv,png,pdf}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
SET_MAX_UNIQUE = 8
Q_LO, Q_HI = 0.0005, 0.9995
XFIRE_FLOOR = 3.81
# known protocol-field names (1-based -> name); others printed as F{n}
KNOWN = {40: "SqNum", 41: "StNum", 42: "cbStatus", 44: "TTL", 57: "timeFromLastChange"}


def fname(col0):
    n = col0 + 1
    return f"F{n}" + (f"({KNOWN[n]})" if n in KNOWN else "")


def main():
    print("[full] carregando treino (benigno p/ regras em TODAS as features)...")
    Xtr, ytr, cvals = util.load_arff(f"{DATASET}.csv")
    nc = util.normal_class
    Xb = Xtr[ytr == nc]
    nfeat = Xb.shape[1]

    rules = []
    for col in range(nfeat):
        vals = Xb[:, col]
        uniq = np.unique(vals)
        if len(uniq) <= SET_MAX_UNIQUE:
            rules.append(("set", set(uniq.tolist())))
        else:
            lo, hi = np.quantile(vals, [Q_LO, Q_HI])
            rules.append(("range", (float(lo), float(hi))))
    del Xtr, ytr

    def viol(M):
        out = np.zeros((M.shape[0], nfeat), dtype=bool)
        for col in range(nfeat):
            kind, spec = rules[col]
            x = M[:, col]
            if kind == "set":
                out[:, col] = ~np.isin(x, np.array(sorted(spec)))
            else:
                lo, hi = spec
                out[:, col] = (x < lo) | (x > hi)
        return out

    print("[full] carregando teste...")
    Xte, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    V = viol(Xte)
    flagged = V.any(axis=1)
    is_ben = (yte == nc)
    ben_fp_per_feat = V[is_ben].mean(axis=0)
    fpr = 100.0 * float(flagged[is_ben].mean())
    print(f"\n[full] benign FPR (union over ALL {nfeat} features) = {fpr:.3f}%")

    print("[full] --- per-attack recall (union over all features) ---")
    rows = []
    for c in range(len(cvals)):
        if c == nc:
            continue
        m = (yte == c)
        rec = 100.0 * float(flagged[m].mean()) if m.any() else float("nan")
        rows.append({"attack": cvals[c], "recall_pct": rec, "n": int(m.sum())})
        print(f"[full]   {cvals[c]:>22}: recall={rec:6.2f}%  (n={int(m.sum())})")

    # which features catch the previously-missed attacks (clean = low benign FP)
    for target in ("poisoned_high_rate", "inverse_replay"):
        c = cvals.index(target)
        m = (yte == c)
        av = V[m].mean(axis=0)                       # attack violation rate per feature
        # rank by (attack violation - benign FP): clean discriminators
        margin = av - ben_fp_per_feat
        order = np.argsort(-margin)[:8]
        print(f"\n[full] === features that catch {target} (attack% / benignFP%) ===")
        for col in order:
            if av[col] < 0.02:
                continue
            print(f"[full]   {fname(col):>22}: attack={100*av[col]:6.2f}%  benignFP={100*ben_fp_per_feat[col]:5.2f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/zeroday_rules_full.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["attack", "n", "recall_pct", "fpr_pct"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "fpr_pct": f"{fpr:.3f}"})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    att = [r["attack"] for r in rows]; rec = [r["recall_pct"] for r in rows]
    x = np.arange(len(att)); cols = ["#b2182b" if "masquerade" in a else "#1b7837" for a in att]
    fig, ax = plt.subplots(figsize=(11, 5.7))
    ax.bar(x, rec, 0.6, color=cols)
    for xi, ri in zip(x, rec):
        ax.text(xi, ri + 1.5, f"{ri:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(XFIRE_FLOOR, color="#666", ls=":", lw=2, label=f"cross-firing floor ({XFIRE_FLOOR}%)")
    ax.set_xticks(x); ax.set_xticklabels(att, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"zero-day recall (%)  [benign FPR = {fpr:.2f}%]"); ax.set_ylim(0, 112)
    ax.set_title("Benign-spec rules over ALL features (zero attack examples):\n"
                 "does ERENO encode the rate/sequence signal per-sample?", fontsize=11.3)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/zeroday_rules_full.png", dpi=170); fig.savefig("results/zeroday_rules_full.pdf")
    print("\n[full] ok -> results/zeroday_rules_full.{csv,png,pdf}")


if __name__ == "__main__":
    main()
