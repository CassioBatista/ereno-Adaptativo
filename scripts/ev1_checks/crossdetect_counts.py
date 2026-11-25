"""Counts for the 5.2 text from results/ev1_crossdetect_<dom>.csv (cover 50 %, precise FPR <= 1 %):
types with no precise cover; FPR range of the imprecise specialists covering them; specialists
covering all of the largest group; rows share."""
import os
import numpy as np
import pandas as pd
os.chdir(os.path.expanduser("~/ereno-Adaptativo/results"))
for d in ["ereno", "cicids", "morris4", "ciciot"]:
    df = pd.read_csv(f"ev1_crossdetect_{d}.csv")
    fpr = df[df.type == "benign_fpr"].iloc[0, 3:].astype(float).values
    t = df[df.type != "benign_fpr"]
    M = t.iloc[:, 3:].astype(float).values
    cols = list(t.columns[3:])
    prec = fpr <= 0.01
    nopc = ~((M >= 0.5) & prec).any(1)
    imp = sorted({fpr[j] for i in np.where(nopc)[0] for j in np.where(M[i] >= 0.5)[0]})
    g1 = t.group.values == 1
    allc = [cols[j] for j in range(len(cols)) if prec[j] and (M[g1, j] >= 0.5).all()]
    print(f"{d}: types {len(t)}; no precise cover {nopc.sum()} ({100 * t.rows[nopc].sum() / t.rows.sum():.2f} % rows): "
          f"{list(t.type[nopc])}; FPR of their covers {100 * min(imp) if imp else 0:.2f}-{100 * max(imp) if imp else 0:.2f} %")
    print(f"   group 1: {g1.sum()} types, {100 * t.rows[g1].sum() / t.rows.sum():.1f} % rows; precise specialists covering all: {allc}")
