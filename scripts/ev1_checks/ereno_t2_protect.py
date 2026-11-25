"""ERENO T2: per round, which nodes are protected against every block arrived so far, for
ReSIDS, FL and GL (same rule as scripts/ev1_tl_faithful_summary.py). Read-only."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo/scripts"))
import numpy as np
import ev1_tl_faithful_summary as S

dom = sys.argv[1] if len(sys.argv) > 1 else "ereno"
runs = {(tr, p): S.node_eval(dom, tr, p) for tr, _ in S.TRACES for p, _ in S.POLS if S.have(dom, tr, p)}
ref = {a: max(np.nanmax([r[a] for r in rows]) for rows in runs.values()) for a in S.BLOCKS}
print("ref", {a: round(100 * v, 2) for a, v in ref.items()})
for p, lab in S.POLS:
    rows = runs[("T2", p)]
    print(f"--- {lab}")
    prev = None
    for t in sorted({r["t"] for r in rows}):
        if t < 1800:
            continue
        rt = sorted((r for r in rows if r["t"] == t), key=lambda r: r["node"])
        prot = [r["node"] for r in rt if all(r[a] >= S.REL * ref[a] for a in S.BLOCKS if a <= t)]
        b1 = [r["node"] for r in rt if r[1800] >= S.REL * ref[1800]]
        b2 = [r["node"] for r in rt if t >= 3600 and r[3600] >= S.REL * ref[3600]]
        key = (tuple(prot), tuple(b1), tuple(b2))
        if key != prev:
            print(f"  {t/60:5.1f} min  share {100*len(prot)/14:5.1f}%  protected {prot}  "
                  f"| block30 {b1} | block60 {b2}")
            prev = key
