#!/usr/bin/env python3
"""Per-pair FP Jaccard for the same-attack specialist pairs, per benign cap. The mean
over pairs hides which pair drives it: pairs with a handful of FPs can drag it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from scored_cache import get_scored

for name, cap in [("500k", 500_000), ("1M", 1_000_000), ("all", 10 ** 9)]:
    ts, fired, y, nc, cv, spec_attack, _ = get_scored(benign_cap=cap)
    Fb = fired[:, y == nc]
    print(f"--- cap {name} ---")
    for at in sorted(set(spec_attack)):
        i, j = [k for k, a in enumerate(spec_attack) if a == at]
        a, b = Fb[i], Fb[j]
        inter = int((a & b).sum()); uni = int((a | b).sum())
        print(f"  {cv[at]:24s} FP A {int(a.sum()):6d}  B {int(b.sum()):6d}  "
              f"both {inter:6d}  Jaccard {inter / uni if uni else float('nan'):.3f}")
    k1 = Fb.any(axis=0)
    k2 = Fb.sum(axis=0) >= 2
    masq = [k for k, a in enumerate(spec_attack) if "masquerade" in cv[a]]
    share = (Fb[masq].any(axis=0) & k1).sum() / max(k1.sum(), 1)
    print(f"  OR false positives: {int(k1.sum())}, of which raised by a masquerade "
          f"specialist: {100 * share:.1f}%;  survive k>=2: {int(k2.sum())}")
