#!/usr/bin/env python3
"""Regret of each policy over every situation of the trust-level replay (paper 2, 5.3).

Cells: domain (ERENO, CICIDS2017) x trace (T1, T2) x arrival offset (0 = registered, +5, +20 min;
the offsets are exploratory) x block (1, 2). In each cell, the exposure of ReSIDS, fixed GL and
fixed FL (results/ev1_tl_offset_summary.csv, all node-rounds after the arrival); regret = the
policy's exposure minus the best of the three. Cells where the three are within 1 point carry no
information and are counted apart.

  python scripts/ev1_tl_regret.py
Out: results/ev1_tl_regret.txt
"""
import csv
import os

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
rows = list(csv.DictReader(open(os.path.join(ROOT, "results/ev1_tl_offset_summary.csv"))))
POLS = ["ReSIDS", "Fixed GL", "Fixed FL"]
cells = {}
for r in rows:
    for b in (1, 2):
        key = (r["domain"], r["trace"], int(r["offset_min"]), b)
        cells.setdefault(key, {})[r["config"]] = 100 * float(r[f"b{b}_exposure"])
lines = ["regret = exposure (% of node-rounds after arrival) minus the best of the three policies", ""]
inform = {k: v for k, v in cells.items() if max(v.values()) - min(v.values()) > 1.0}
lines.append(f"{len(cells)} cells, {len(inform)} with a spread above 1 point")
lines.append(f"{'cell':<34}" + "".join(f"{p:>10}" for p in POLS) + "   regret R/GL/FL")
for k in sorted(inform):
    v = inform[k]
    best = min(v.values())
    lines.append(f"{k[0]:<11}{k[1]} +{k[2]:>2} min block {k[3]}      " + "".join(f"{v[p]:>10.1f}" for p in POLS)
                 + "   " + "/".join(f"{v[p] - best:.1f}" for p in POLS))
lines.append("")
for p in POLS:
    reg = [v[p] - min(v.values()) for v in inform.values()]
    worst = sum(1 for v in inform.values() if v[p] == max(v.values()) and v[p] - min(v.values()) > 1.0)
    best = sum(1 for v in inform.values() if v[p] - min(v.values()) <= 1.0)
    lines.append(f"{p:<9} regret mean {np.mean(reg):5.1f}, max {np.max(reg):5.1f} points; best (within 1 point) "
                 f"in {best}/{len(inform)}; worst in {worst}/{len(inform)}")
reg0 = {p: [v[p] - min(v.values()) for k, v in inform.items() if k[2] == 0] for p in POLS}
lines.append("registered arrivals only (offset 0): " + "; ".join(
    f"{p} mean {np.mean(r):.1f}, max {np.max(r):.1f}" for p, r in reg0.items()))
regx = {p: [v[p] - min(v.values()) for k, v in inform.items() if k[2] != 0] for p in POLS}
lines.append("shifted arrivals only (+5, +20): " + "; ".join(
    f"{p} mean {np.mean(r):.1f}, max {np.max(r):.1f}" for p, r in regx.items()))
txt = "\n".join(lines)
open(os.path.join(ROOT, "results/ev1_tl_regret.txt"), "w").write(txt + "\n")
print(txt)
