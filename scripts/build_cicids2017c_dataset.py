#!/usr/bin/env python3
"""Build the corrected CICIDS2017 (Engelen et al.) in the pipeline's ARFF layout, with the
temporal split used by scripts/cross_domain_results.py:

  * per day and attack category, the last 30 % of the episode's time span -> TEST
    (every flow inside it, benign included);
  * Monday (benign only): first 70 % -> CALIBRATION, last 30 % -> TEST;
  * everything else -> TRAIN (the only part GRASP may see).

Conditions of use from the audit (docs/PROFILES.md, it-flow): Heartbleed dropped,
"Infiltration - Portscan" merged into PortScan, Attempted -> Benign, identifiers and
FWD/Bwd Init Win Bytes excluded. Label set cicids2017c-7 (+ Benign).

Outputs (in the repo root, not versioned, like the other all_in_one_*.csv):
  all_in_one_cicids2017c_{train,test,calib}.csv   ARFF: F1..F80 numeric + @class@
  features/cicids2017c_feature_names.json          F# -> column name
The first data row of each file is Benign (util.load_arff takes it as the normal class).

  python scripts/build_cicids2017c_dataset.py
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402

CLASSES = ["Benign"] + C.CATS_C


def write_arff(path, X, cat, n_feat):
    first = int(np.where(cat == "Benign")[0][0])          # a Benign row goes first
    order = np.r_[first, np.delete(np.arange(len(cat)), first)]
    with open(path, "w") as f:
        f.write("@relation cicids2017c\n\n")
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
    z = zipfile.ZipFile(C.D + "cicids2017_improved/CICIDS2017_improved.zip")
    df = pd.concat([pd.read_csv(z.open(f"{d}.csv"), low_memory=False,
                                dtype={c: "str" for c in C.TEXT}).assign(_day=d)
                    for d in ["monday", "tuesday", "wednesday", "thursday", "friday"]],
                   ignore_index=True)
    df["_cat"] = df["Label"].map(C.cat_cicids)
    df = df[df["_cat"] != "drop"].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="mixed")
    df = df.sort_values("_ts", kind="stable").reset_index(drop=True)
    feats = [c for c in df.columns
             if c not in C.TEXT + C.IDENT + C.FINGERPRINT + ["_day", "_cat", "_ts"]]
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float64").to_numpy()
    X[~np.isfinite(X)] = np.nan
    test = np.zeros(len(df), bool)
    for d, g in df.groupby("_day"):
        if d == "monday":
            continue
        for c, gc in g.groupby("_cat"):
            if c != "Benign":
                t70, tmax = gc["_ts"].quantile(0.7), gc["_ts"].max()
                test |= ((df["_day"] == d) & (df["_ts"] >= t70) & (df["_ts"] <= tmax)).to_numpy()
    mon = (df["_day"] == "monday").to_numpy()
    t70m = df.loc[mon, "_ts"].quantile(0.7)
    calib = mon & (df["_ts"] < t70m).to_numpy()
    test |= mon & (df["_ts"] >= t70m).to_numpy()
    train = ~test & ~calib
    cat = df["_cat"].to_numpy()
    json.dump({"source": "CICIDS2017_improved.zip (Engelen et al.), audit exclusions applied",
               "names": {str(i + 1): n for i, n in enumerate(feats)}},
              open("features/cicids2017c_feature_names.json", "w"), indent=1)
    print(f"features: {len(feats)}; train {int(train.sum()):,} / test {int(test.sum()):,} / "
          f"calib {int(calib.sum()):,}")
    for name, m in (("train", train), ("test", test), ("calib", calib)):
        write_arff(f"all_in_one_cicids2017c_{name}.csv", X[m], cat[m], len(feats))
    # timestamps of the test and calibration rows, in file order, for window analyses
    for name, m in (("test", test), ("calib", calib)):
        c = cat[m]
        first = np.where(c == "Benign")[0][0]
        order = np.r_[first, np.delete(np.arange(len(c)), first)]
        secs = (df.loc[m, "_ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()[order]
        np.save(f"all_in_one_cicids2017c_{name}_ts.npy", secs)


if __name__ == "__main__":
    main()
