#!/usr/bin/env python3
"""CICIoT2023 restricted to the categories whose labels carry their attack protocol
(DDoS, DoS, Mirai; see _ciciot_label_protocol_check.py). Same protocol as
cross_domain_results.py (N = 2 per category -> 6 specialists, fixed 0.5 threshold).
Out: results/ciciot2023_volumetric_subset.txt
"""
import glob
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402
from inspect_ciciot2023 import category  # noqa: E402

CATS = ["DDoS", "DoS", "Mirai"]


def main():
    parts = sorted(glob.glob(C.D + "ciciot2023/part-*.csv"))
    tr = pd.concat([pd.read_csv(p) for p in parts[:7]], ignore_index=True)
    te = pd.concat([pd.read_csv(p) for p in parts[7:]], ignore_index=True)
    lab = tr.columns[-1]
    for d in (tr, te):
        d["_cat"] = d[lab].map(category)
    keep = CATS + ["Benign"]
    tr, te = tr[tr["_cat"].isin(keep)], te[te["_cat"].isin(keep)]
    feats = [c for c in tr.columns if c not in (lab, "_cat", "IAT")]
    C.N = 2 * len(CATS)
    b, _ = C.train_specialists(tr[feats].to_numpy(np.float32), tr["_cat"].to_numpy(), CATS)
    V = C.votes(b, te[feats].to_numpy(np.float32))
    atk = te["_cat"].to_numpy() != "Benign"
    print(f"test: {int(atk.sum()):,} attack / {int((~atk).sum()):,} benign; specialists={C.N}")
    for k in (1, 2):
        s = C.stats(V.sum(1) >= k, atk)
        print(f"  GL union k>={k}: recall {s['recall']:.2f}%  FPR {s['FPR']:.3f}%  "
              f"F1 {s['F1']:.2f}  F1@13.75 {s['F1@13.75']:.2f}")
    for n in range(C.N, 0, -1):
        s = C.stats(V[:, :n].sum(1) >= 2, atk)
        print(f"  FL k>=2 with {n} nodes: recall {s['recall']:.2f}%  FPR {s['FPR']:.3f}%")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/ciciot2023_volumetric_subset.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
