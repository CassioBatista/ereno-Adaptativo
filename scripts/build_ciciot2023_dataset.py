#!/usr/bin/env python3
"""Build the CICIoT2023 sample (10 of the 169 CSV parts) in the pipeline's ARFF layout:
part-00000..06 -> TRAIN (the only part GRASP may see), part-00007..09 -> TEST.
Label set ciciot2023-7 (+ Benign); IAT excluded (it behaves as a capture clock; audit).

Outputs (repo root, not versioned):
  all_in_one_ciciot2023_{train,test}.csv     ARFF: F1..F45 numeric + @class@
  all_in_one_ciciot2023_trainbal.csv         TRAIN capped at 20k rows per class (global GRASP)
  features/ciciot2023_feature_names.json     F# -> column name
The first data row is Benign (util.load_arff takes it as the normal class).

  python scripts/build_ciciot2023_dataset.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_ciciot2023 import category  # noqa: E402

CLASSES = ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]
BAL_CAP = 20_000          # rows per class in the balanced copy used by the global GRASP


def write_arff(path, X, cat, n_feat):
    first = int(np.where(cat == "Benign")[0][0])
    order = np.r_[first, np.delete(np.arange(len(cat)), first)]
    with open(path, "w") as f:
        f.write("@relation ciciot2023\n\n")
        for i in range(1, n_feat + 1):
            f.write(f"@attribute F{i} numeric\n")
        f.write("@attribute @class@ {" + ", ".join(CLASSES) + "}\n\n@data\n")
        Xo, co = X[order], cat[order]
        for s in range(0, len(Xo), 200_000):
            block = pd.DataFrame(Xo[s:s + 200_000])
            block["c"] = co[s:s + 200_000]
            block.to_csv(f, header=False, index=False, na_rep="nan", float_format="%.6g")
    print(f"  {path}: {len(X):,} rows", flush=True)


def main():
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    parts = sorted(glob.glob(os.path.expanduser("~/datasets/ciciot2023/part-*.csv")))
    tr = pd.concat([pd.read_csv(p) for p in parts[:7]], ignore_index=True)
    te = pd.concat([pd.read_csv(p) for p in parts[7:]], ignore_index=True)
    lab = tr.columns[-1]
    feats = [c for c in tr.columns if c not in (lab, "IAT")]
    json.dump({"source": "CICIoT2023 CSV parts 0-9 (Kaggle mirror madhavmalhotra), IAT excluded",
               "names": {str(i + 1): n for i, n in enumerate(feats)}},
              open("features/ciciot2023_feature_names.json", "w"), indent=1)
    # class-balanced copy of TRAIN for the global (multiclass) GRASP: benign is 2.4 % of the
    # release, so an unbalanced multiclass objective would all but ignore it.
    tr["_c"] = tr[lab].map(category)
    bal = pd.concat([g.sample(min(len(g), BAL_CAP), random_state=42)
                     for _, g in tr.groupby("_c")])
    print("balanced train per class:", bal["_c"].value_counts().to_dict())
    tr = tr.drop(columns="_c")
    bal = bal.drop(columns="_c")
    for name, d in (("train", tr), ("test", te), ("trainbal", bal)):
        if os.path.exists(f"all_in_one_ciciot2023_{name}.csv"):
            continue                                       # already built
        X = d[feats].to_numpy(np.float64)
        X[~np.isfinite(X)] = np.nan
        write_arff(f"all_in_one_ciciot2023_{name}.csv", X, d[lab].map(category).to_numpy(),
                   len(feats))


if __name__ == "__main__":
    main()
