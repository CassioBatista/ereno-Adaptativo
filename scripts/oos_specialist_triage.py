#!/usr/bin/env python3
"""Per-specialist window triage on ERENO, out of sample (same protocol as oos_thresholds.py).

On the corrected CICIDS2017 the ERENO volume/fraction rule misses slow attacks; counting
the flagged samples per SPECIALIST category (a slow attack accumulates in its own
specialist, benign false alarms spread over specialists) recovers them. This script checks
that adding that rule does not degrade ERENO, where the reported operating point is
blocked-5, margin x1.25, R2 = volume OR fraction: 276/282 attack windows, 1/724 benign.

Per window and attack category c: S_c = fused-flagged samples (k >= 2) on which a
specialist of c fired. Calibration (benign windows of the four calibration blocks):
    T_c = max(ceil(1.25 * max benign S_c) + 1, floor)
Rules: R2 (reference), SPEC (any S_c >= T_c), R2 OR SPEC; floors 1, 2, 3.

  python scripts/oos_specialist_triage.py
Out: results/oos_specialist_triage_ereno.txt
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from scored_cache import get_scored

K_FUSION, MARGIN, FLOORS = 2, 0.25, (1, 2, 3)
OUT = "results/oos_specialist_triage_ereno.txt"


def windows(ts, fired, y, nc, cv, spec_attack):
    votes = fired.sum(axis=0)
    flag = votes >= K_FUSION
    cats = sorted(set(spec_attack))
    rows = {c: [i for i, a in enumerate(spec_attack) if a == c] for c in cats}
    per = np.stack([fired[rows[c]].any(axis=0) & flag for c in cats])     # C x n
    win = np.floor(ts - float(ts[0])).astype(np.int64)
    bnd = np.searchsorted(win, np.arange(int(win[-1]) + 2))
    cs = np.r_[0, np.cumsum(flag)]
    cp = np.concatenate([np.zeros((len(cats), 1), np.int64), np.cumsum(per, axis=1)], axis=1)
    W = []
    for w in range(int(win[-1]) + 1):
        lo, hi = int(bnd[w]), int(bnd[w + 1])
        if hi == lo:
            continue
        nf = int(cs[hi] - cs[lo])
        yy = y[lo:hi]
        atk = bool((yy != nc).any())
        W.append({"n": hi - lo, "nf": nf, "frac": nf / (hi - lo), "S": cp[:, hi] - cp[:, lo],
                  "atk": atk, "cls": cv[int(np.bincount(yy[yy != nc]).argmax())] if atk else "benign"})
    return W, cats


def main():
    ts, fired, y, nc, cv, spec_attack, _ = get_scored(benign_cap=10 ** 9)
    W, cats = windows(ts, fired, y, nc, cv, spec_attack)
    nA = sum(r["atk"] for r in W); nB = len(W) - nA
    lines = [f"ERENO per-specialist triage, blocked-5, margin x1.25, k>=2: {nA} attack / {nB} benign windows",
             f"specialist categories: {cats}"]
    folds = np.array_split(np.arange(len(W)), 5)
    for floor in FLOORS:
        tot = {k: [0, 0] for k in ("R2", "SPEC", "R2 OR SPEC")}
        lines.append(f"\n--- floor {floor} ---")
        for fi, held in enumerate(folds):
            hs = set(held.tolist())
            cal = [W[i] for i in range(len(W)) if i not in hs and not W[i]["atk"]]
            T = int(math.ceil(max(r["nf"] for r in cal) * (1 + MARGIN))) + 1
            f = max(r["frac"] for r in cal) * (1 + MARGIN)
            Sm = np.max([r["S"] for r in cal], axis=0)
            Tc = np.maximum(np.ceil(Sm * (1 + MARGIN)) + 1, floor)
            lines.append(f"  fold {fi}: T={T} f={f:.4f} T_c=" +
                         ", ".join(f"{c}={int(t)}" for c, t in zip(cats, Tc)))
            for i in held:
                r = W[i]
                r2 = r["nf"] >= T or r["frac"] > f
                sp = bool((r["S"] >= Tc).any())
                for k, hit in (("R2", r2), ("SPEC", sp), ("R2 OR SPEC", r2 or sp)):
                    tot[k][0 if r["atk"] else 1] += hit
        for k, (tp, fp) in tot.items():
            lines.append(f"  {k:<11} attack windows {tp}/{nA} ({100 * tp / nA:.1f}%)   "
                         f"benign false alarms {fp}/{nB} ({100 * fp / nB:.2f}%)")
    print("\n".join(lines))
    open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    main()
