#!/usr/bin/env python3
"""Does the flagged-fraction rule alone do what volume + fraction do (1-s windows)?"""
import csv

R = list(csv.DictReader(open("results/window_prevalence_sweep_ereno_windows.csv")))
for r in R:
    for k in ("n_samples", "n_flags", "attack", "k_votes"):
        r[k] = int(r[k])
    r["frac"] = r["n_flags"] / r["n_samples"]
A = [r for r in R if r["attack"]]
B = [r for r in R if not r["attack"]]
for f in (0.0364, 0.0433, 0.05):
    tp = sum(r["frac"] > f for r in A); fp = sum(r["frac"] > f for r in B)
    tpv = sum(r["frac"] > f or r["n_flags"] >= 324 for r in A)
    print(f"fraction > {f:.4f}: alone TP {tp}/{len(A)} FP {fp}/{len(B)} | "
          f"with volume>=324: TP {tpv}/{len(A)}")
only_vol = [r for r in A if r["n_flags"] >= 324 and r["frac"] <= 0.0433]
print("attack windows caught by volume but NOT by fraction>0.0433:", len(only_vol))
for r in only_vol:
    print(f"  {r['class']:22s} samples {r['n_samples']} n_flags {r['n_flags']} frac {r['frac']:.4f}")
