#!/usr/bin/env python3
"""Three primary contrasts vs IID (paired across seeds) + Holm-Bonferroni.

Produces Table 13 (specialist vs IID reference). Reads the per-seed factorial
results (results/factorial_holm.csv) and computes the paired t-tests against the
IID baseline (NOT the non-specialist average), which is what the paper reports:
  C1: specialist > IID under OR (k=1), F1
  C2: k>=2 cuts the specialist false-positive rate (OR FPR - k>=2 FPR)
  C3: specialist > IID under corroboration (k>=2), F1
No retraining: consumes the saved per-seed data. Saida: results/contrasts_vs_iid.csv
Rodar: ~/venv-ereno314/bin/python scripts/contrasts_vs_iid.py
"""
import collections
import csv
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "results", "factorial_holm.csv")
OUT = os.path.join(HERE, "results", "contrasts_vs_iid.csv")


def main():
    d = collections.defaultdict(dict)
    with open(SRC) as fh:
        for r in csv.DictReader(fh):
            d[int(r["seed"])][(r["partitioner"], int(r["k"]))] = {
                "F1": float(r["F1"]), "FPR": float(r["FPR"])}
    seeds = sorted(d)

    def col(part, k, metric):
        return np.array([d[s][(part, k)][metric] for s in seeds])

    contrasts = {
        "C1: specialist>IID (OR, F1)":  col("specialist", 1, "F1") - col("iid", 1, "F1"),
        "C2: k>=2 cuts FPR (specialist)": col("specialist", 1, "FPR") - col("specialist", 2, "FPR"),
        "C3: specialist>IID (k>=2, F1)": col("specialist", 2, "F1") - col("iid", 2, "F1"),
    }
    raw = {name: (float(arr.mean()), float(stats.ttest_1samp(arr, 0.0).pvalue))
           for name, arr in contrasts.items()}

    # Holm-Bonferroni (step-down) over the family of 3
    order = sorted(raw, key=lambda n: raw[n][1])
    m = len(order); run = 0.0; holm = {}
    for i, name in enumerate(order):
        run = max(run, min(1.0, raw[name][1] * (m - i)))
        holm[name] = run

    print(f"[contrasts] {len(seeds)} seeds: {seeds}")
    print(f"{'contrast':<32}{'effect':>12}{'p_raw':>12}{'p_Holm':>12}  sig")
    rows = []
    for name in ("C1: specialist>IID (OR, F1)", "C2: k>=2 cuts FPR (specialist)",
                 "C3: specialist>IID (k>=2, F1)"):
        diff, p = raw[name]; ph = holm[name]
        sig = "***" if ph < 0.001 else "**" if ph < 0.01 else "*" if ph < 0.05 else "ns"
        print(f"{name:<32}{diff:>+12.3f}{p:>12.2e}{ph:>12.2e}  {sig}")
        rows.append((name, diff, p, ph, sig))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("contrast,effect,p_raw,p_holm,significance\n")
        for name, diff, p, ph, sig in rows:
            fh.write(f'"{name}",{diff:.4f},{p:.6e},{ph:.6e},{sig}\n')
    print(f"\n[contrasts] CSV -> {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
