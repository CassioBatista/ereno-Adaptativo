#!/usr/bin/env python3
"""Zero-day detection via EXPLICIT protocol-specification rules (per-attack sweep).

Deterministic, auditable, ZERO attack examples: each rule is LEARNED FROM BENIGN ONLY.
For every IEC-61850 protocol field we derive one rule:
  * allowed-set   — if benign takes few distinct values (e.g., TTL = {11000}): value
                    must be in the benign set;
  * range [lo,hi] — otherwise: value must lie in a robust benign interval
                    [q_LO, q_HI].
A packet is flagged (candidate zero-day) if it VIOLATES any rule. We report the learned
rules, the per-attack held-out recall, the benign FPR, and which field catches which
attack. Compares to the learned IF-on-protocol-fields and the 3.8% cross-firing floor.
Standalone (paper1-revision); touches nothing.
Out: results/zeroday_rules.{csv,png,pdf}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
PROTO_1BASED = [40, 41, 42, 44, 57]
PROTO = [c - 1 for c in PROTO_1BASED]
NAME = {39: "F40 SqNum", 40: "F41 StNum", 41: "F42 cbStatus",
        43: "F44 TTL", 56: "F57 timeFromLastChange"}
SET_MAX_UNIQUE = 8          # <= this many benign distinct values -> allowed-set rule
Q_LO, Q_HI = 0.0005, 0.9995 # robust benign range for continuous fields
XFIRE_FLOOR = 3.81


def main():
    print("[rules] carregando treino (benigno p/ aprender as regras)...")
    Xtr, ytr, cvals = util.load_arff(f"{DATASET}.csv")
    nc = util.normal_class
    Xb = Xtr[ytr == nc][:, PROTO]

    # ---- learn one explicit rule per protocol field, from benign only ----
    rules = {}
    print("[rules] === especificação aprendida do BENIGNO (zero exemplo de ataque) ===")
    for j, col in enumerate(PROTO):
        vals = Xb[:, j]
        uniq = np.unique(vals)
        if len(uniq) <= SET_MAX_UNIQUE:
            rules[col] = ("set", set(uniq.tolist()))
            print(f"  {NAME[col]:>24}: value ∈ {{{', '.join(f'{u:g}' for u in uniq)}}}  (allowed-set)")
        else:
            lo, hi = np.quantile(vals, [Q_LO, Q_HI])
            rules[col] = ("range", (float(lo), float(hi)))
            print(f"  {NAME[col]:>24}: value ∈ [{lo:.4g}, {hi:.4g}]  (robust range)")
    del Xtr, ytr

    def violations(M):
        """Boolean matrix (n_samples x n_fields): True where the rule is violated."""
        out = np.zeros((M.shape[0], len(PROTO)), dtype=bool)
        for j, col in enumerate(PROTO):
            kind, spec = rules[col]
            x = M[:, col]
            if kind == "set":
                allowed = np.array(sorted(spec))
                out[:, j] = ~np.isin(x, allowed)
            else:
                lo, hi = spec
                out[:, j] = (x < lo) | (x > hi)
        return out

    print("\n[rules] carregando teste...")
    Xte, yte, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc

    V = violations(Xte)                       # per-field violation
    flagged = V.any(axis=1)                   # union rule = candidate zero-day

    is_ben = (yte == nc)
    fpr = 100.0 * float(flagged[is_ben].mean())
    print(f"\n[rules] benign FPR = {fpr:.3f}%  (union of all rules)")
    print(f"[rules] --- per-attack held-out recall (zero examples) ---")
    rows = []
    for c in range(len(cvals)):
        if c == nc:
            continue
        m = (yte == c)
        rec = 100.0 * float(flagged[m].mean()) if m.any() else float("nan")
        # which fields catch this attack
        per_field = {NAME[PROTO[j]]: 100.0 * float(V[m, j].mean()) for j in range(len(PROTO))}
        top = sorted(per_field.items(), key=lambda kv: -kv[1])[:2]
        why = ", ".join(f"{k.split()[0]}={v:.0f}%" for k, v in top if v > 1)
        rows.append({"attack": cvals[c], "n": int(m.sum()), "recall_pct": rec, "why": why})
        tag = "  <-- masquerade (hard)" if "masquerade" in cvals[c] else ""
        print(f"[rules]   {cvals[c]:>22}: recall={rec:6.2f}%  (n={int(m.sum()):>6})  by[{why}]{tag}")

    os.makedirs("results", exist_ok=True)
    with open("results/zeroday_rules.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["attack", "n", "recall_pct", "why", "fpr_pct"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "fpr_pct": f"{fpr:.3f}"})

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    att = [r["attack"] for r in rows]
    rec = [r["recall_pct"] for r in rows]
    x = np.arange(len(att))
    cols = ["#b2182b" if "masquerade" in a else "#1b7837" for a in att]
    fig, ax = plt.subplots(figsize=(11, 5.7))
    ax.bar(x, rec, 0.6, color=cols)
    for xi, ri in zip(x, rec):
        ax.text(xi, ri + 1.5, f"{ri:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(XFIRE_FLOOR, color="#666", ls=":", lw=2,
               label=f"supervised cross-firing floor ({XFIRE_FLOOR}%)")
    ax.set_xticks(x); ax.set_xticklabels(att, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"zero-day recall (%)   [benign FPR = {fpr:.2f}%]")
    ax.set_ylim(0, 108)
    ax.set_title("Zero-day by EXPLICIT protocol-spec rules (learned from benign, zero attack examples):\n"
                 "protocol-violating attacks caught near-perfectly at ~0% FP; masquerade stays hard",
                 fontsize=11.3)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/zeroday_rules.png", dpi=170); fig.savefig("results/zeroday_rules.pdf")
    print("\n[rules] ok -> results/zeroday_rules.{csv,png,pdf}")


if __name__ == "__main__":
    main()
