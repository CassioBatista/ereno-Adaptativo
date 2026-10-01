#!/usr/bin/env python3
"""The attack windows that neither absolute volume (n_flags >= 275) nor flagged fraction
(n_flags / window_samples >= 0.05) catches: what are they, and does k_votes help?"""
import csv

R = list(csv.DictReader(open("results/window_prevalence_sweep_ereno_windows.csv")))
for r in R:
    for k in ("n_samples", "n_flags", "attack", "attack_samples", "k_votes"):
        r[k] = int(r[k])
B = [r for r in R if not r["attack"]]
A = [r for r in R if r["attack"]]
kb = max(r["k_votes"] for r in B if r["n_flags"] > 0)
print(f"benign alarming windows: k_votes max = {kb}")
rest = [r for r in A if r["n_flags"] < 275 and r["n_flags"] / r["n_samples"] < 0.05]
print(f"residual attack windows: {len(rest)}")
for r in rest:
    print(f"  {r['class']:22s} samples {r['n_samples']:5d}  attack_samples "
          f"{r['attack_samples']:4d}  n_flags {r['n_flags']:4d}  k_votes {r['k_votes']}"
          f"  {'<- k_votes above every benign window' if r['k_votes'] > kb else ''}")
