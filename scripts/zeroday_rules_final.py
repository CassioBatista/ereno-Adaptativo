#!/usr/bin/env python3
"""Final protocol specification for zero-day: VALUE + RATE + SEQUENCE rule families.

Built on the all-feature discovery (scripts/zeroday_rules_full.py): the rate signal is
carried by F55/F56 (benign-invariant, violated by the high-rate attack); the order
signal is only PARTIALLY encoded per-sample (F49/F50/F52). Each rule is learned from
BENIGN only (zero attack examples): allowed-set for near-constant fields, robust range
otherwise. A packet is flagged if it violates any rule in any family.

  VALUE     F40 SqNum, F41 StNum, F42 cbStatus, F44 TTL, F57 timeFromLastChange
  RATE      F55, F56   (inter-packet / timing; catch poisoned_high_rate)
  SEQUENCE  F49, F50, F52   (partial order signal; help inverse_replay)

Honest limit: full order detection (inverse_replay) needs STATEFUL StNum/SqNum
monotonicity over an ordered per-source stream, which the aggregated dataset does not
support; masquerade_fake_normal is irreducible. Feature roles are inferred from
behaviour + the ERENO consensus cores; exact semantics of F49/F50/F52/F55/F56 should be
confirmed against the ERENO feature dictionary.
Out: results/zeroday_rules_final.{csv,png,pdf}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FAMILIES = {                                   # 1-based feature indices per family
    "VALUE":    [40, 41, 42, 44, 57],
    "RATE":     [55, 56],
    "SEQUENCE": [49, 50, 52],
}
SET_MAX_UNIQUE = 8
Q_LO, Q_HI = 0.0005, 0.9995
XFIRE_FLOOR = 3.81
ALL_COLS = sorted({c - 1 for cols in FAMILIES.values() for c in cols})
FAM_OF = {c - 1: fam for fam, cols in FAMILIES.items() for c in cols}


def main():
    print("[final] carregando treino (benigno)...")
    Xtr, ytr, cvals = util.load_arff(f"{DATASET}.csv")
    nc = util.normal_class
    Xb = Xtr[ytr == nc]

    rules = {}
    print("[final] === especificação final (VALUE + RATE + SEQUENCE), aprendida do benigno ===")
    for fam, cols1 in FAMILIES.items():
        for c1 in cols1:
            col = c1 - 1
            vals = Xb[:, col]
            uniq = np.unique(vals)
            if len(uniq) <= SET_MAX_UNIQUE:
                rules[col] = ("set", set(uniq.tolist()))
                spec = "{" + ", ".join(f"{u:g}" for u in uniq) + "}"
            else:
                lo, hi = np.quantile(vals, [Q_LO, Q_HI])
                rules[col] = ("range", (float(lo), float(hi)))
                spec = f"[{lo:.4g}, {hi:.4g}]"
            print(f"  [{fam:>8}] F{c1}: {spec}")
    del Xtr, ytr

    def viol(M):
        out = np.zeros((M.shape[0], len(ALL_COLS)), dtype=bool)
        for k, col in enumerate(ALL_COLS):
            kind, spec = rules[col]
            x = M[:, col]
            out[:, k] = (~np.isin(x, np.array(sorted(spec)))) if kind == "set" \
                else (x < spec[0]) | (x > spec[1])
        return out

    print("\n[final] carregando teste...")
    Xte, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    V = viol(Xte)
    flagged = V.any(axis=1)
    is_ben = (yte == nc)
    fpr = 100.0 * float(flagged[is_ben].mean())
    # per-family flag (union within family)
    fam_idx = {fam: [k for k, col in enumerate(ALL_COLS) if FAM_OF[col] == fam] for fam in FAMILIES}
    print(f"\n[final] benign FPR = {fpr:.3f}%  (VALUE+RATE+SEQUENCE)")
    print("[final] --- per-attack recall + dominant family ---")
    rows = []
    for c in range(len(cvals)):
        if c == nc:
            continue
        m = (yte == c)
        rec = 100.0 * float(flagged[m].mean()) if m.any() else float("nan")
        fam_rec = {fam: 100.0 * float(V[np.ix_(m, ks)].any(axis=1).mean())
                   for fam, ks in fam_idx.items()}
        dom = max(fam_rec, key=fam_rec.get)
        rows.append({"attack": cvals[c], "n": int(m.sum()), "recall_pct": rec,
                     "dominant_family": dom, **{f"{f}_pct": v for f, v in fam_rec.items()}})
        tag = "  <-- masquerade (hard)" if "masquerade" in cvals[c] else ""
        print(f"[final]   {cvals[c]:>22}: recall={rec:6.2f}%  "
              f"[VALUE={fam_rec['VALUE']:.0f} RATE={fam_rec['RATE']:.0f} SEQ={fam_rec['SEQUENCE']:.0f}]{tag}")

    os.makedirs("results", exist_ok=True)
    hdr = ["attack", "n", "recall_pct", "dominant_family", "VALUE_pct", "RATE_pct",
           "SEQUENCE_pct", "fpr_pct"]
    with open("results/zeroday_rules_final.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader()
        for r in rows:
            w.writerow({**{k: r.get(k) for k in hdr if k != "fpr_pct"}, "fpr_pct": f"{fpr:.3f}"})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FCOL = {"VALUE": "#1b7837", "RATE": "#e08214", "SEQUENCE": "#6a3d9a"}
    att = [r["attack"] for r in rows]; rec = [r["recall_pct"] for r in rows]
    cols = ["#b2182b" if "masquerade" in a else FCOL[r["dominant_family"]] for a, r in zip(att, rows)]
    x = np.arange(len(att))
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.bar(x, rec, 0.62, color=cols)
    for xi, ri in zip(x, rec):
        ax.text(xi, ri + 1.6, f"{ri:.0f}%", ha="center", fontsize=9.5, fontweight="bold")
    ax.annotate("partial — full order needs\nstateful StNum/SqNum monotonicity",
                (1, 62), fontsize=7.6, ha="center", color="#6a3d9a", style="italic")
    ax.axhline(XFIRE_FLOOR, color="#666", ls=":", lw=2, label=f"cross-firing floor ({XFIRE_FLOOR}%)")
    from matplotlib.patches import Patch
    handles = [Patch(fc=FCOL[f], label=f"{f} rules") for f in FAMILIES] + \
              [Patch(fc="#b2182b", label="masquerade (stealth)")]
    ax.set_xticks(x); ax.set_xticklabels(att, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"zero-day recall (%)   [benign FPR = {fpr:.2f}%]"); ax.set_ylim(0, 112)
    ax.set_title("Zero-day by protocol spec: VALUE + RATE + SEQUENCE rules (benign-only, zero examples):\n"
                 "rate solved (poisoned 100%); order partial (inverse_replay); masquerade irreducible",
                 fontsize=10.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=handles + [plt.Line2D([0], [0], ls=":", color="#666", lw=2, label=f"floor {XFIRE_FLOOR}%")],
              loc="center right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/zeroday_rules_final.png", dpi=170); fig.savefig("results/zeroday_rules_final.pdf")
    print("\n[final] ok -> results/zeroday_rules_final.{csv,png,pdf}")


if __name__ == "__main__":
    main()
