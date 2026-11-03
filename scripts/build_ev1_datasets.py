#!/usr/bin/env python3
"""Dataset preparation of docs/EVALUATION_PROTOCOL.md (tag evaluation-v1), section 2.1, for
the three domains that need new files (ERENO uses its existing all_in_one_ereno_*.csv):

  ev1_cicids   CICIDS2017, improved: 26 Label types; per label, its first 70 % in time TRAIN
               and the rest TEST; benign likewise, day by day (correction of 2026-11-03,
               protocol change log: the interval rule left types without training rows)
  ev1_morris4  Morris-4 (gas pipeline, 2015): 30 specific attacks (5 excluded by Q2a); for
               each label its first 70 % in time TRAIN (amendment of 2026-11-03: replaces
               Morris-3, which its authors superseded)
  ev1_morris3  Morris-3 (water tank), kept for the record: 6 attack categories (CMRI
               excluded by Q2a); file order, first 70 % of each label TRAIN
  ev1_ciciot   CICIoT2023, 10 screened part files: 30 labels (three excluded by Q2b); part
               files 0-6 TRAIN, 7-9 TEST

Common rule: rows of the types excluded by the screening (results/protocol_v1/<release>.json)
and misaligned rows (Q1a, scripts/apply_protocol.py) are removed; features are the content
columns of the screening loaders (identifiers removed; no model-based exclusion); labels are
the types, written as safe names (the mapping is saved). Outputs in the repo root (not
versioned, like the other all_in_one_*.csv):

  all_in_one_<ds>_{train,test}.csv         ARFF: F1..Fn numeric + @class@ (Benign first)
  all_in_one_<ds>_trainbal.csv             (ev1_ciciot) TRAIN capped at 20,000 rows per
                                           type, for the global GRASP (benign < 10 %)
  all_in_one_<ds>_test_order.npy           test rows' position in time / file order, in the
                                           written row order (windows; not ev1_ciciot)
  features/<ds>_feature_names.json         F# -> column name, type -> safe label name

  python scripts/build_ev1_datasets.py [ev1_cicids] [ev1_morris3] [ev1_ciciot]
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BAL_CAP = 20_000


def safe(name):
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
    return s or "unnamed"


def write_arff(path, X, lab, classes, relation):
    first = int(np.where(lab == "Benign")[0][0])
    order = np.r_[first, np.delete(np.arange(len(lab)), first)]
    with open(path, "w") as f:
        f.write(f"@relation {relation}\n\n")
        for i in range(1, X.shape[1] + 1):
            f.write(f"@attribute F{i} numeric\n")
        f.write("@attribute @class@ {" + ", ".join(classes) + "}\n\n@data\n")
        Xo, lo = X[order], lab[order]
        for s in range(0, len(Xo), 200_000):
            block = pd.DataFrame(Xo[s:s + 200_000])
            block["c"] = lo[s:s + 200_000]
            block.to_csv(f, header=False, index=False, na_rep="nan", float_format="%.6g")
    print(f"  {path}: {len(X):,} rows", flush=True)
    return order


def excluded(release):
    j = json.load(open(f"results/protocol_v1/{release}.json"))
    best = next(t for t in j["tables"] if t["table"] == j["best_table"])
    return {t for t, v in best["types"].items() if v["excluded_by"]}


def prepare(rel, release, typ):
    """Drop excluded types and misaligned rows; numeric content matrix; safe labels."""
    import apply_protocol as A
    df = rel.df
    typ = np.asarray(typ, dtype=object)
    bad, _ = A.misaligned(df, rel.feats + rel.ids)
    ex = excluded(release)
    keep = ~bad & ~np.isin(typ, list(ex))
    print(f"{rel.name}: {len(df):,} rows; excluded types {sorted(ex) or 'none'}; misaligned rows "
          f"{int(bad.sum()):,}; kept {int(keep.sum()):,}", flush=True)
    X = df.loc[keep, rel.feats].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).astype("category").cat.codes      # categorical -> codes
    X = np.array(X.to_numpy(np.float64))                # writable copy (pandas copy-on-write)
    X[~np.isfinite(X)] = np.nan
    t = typ[keep]
    lab = np.where(t == rel.benign, "Benign", np.array([safe(v) for v in t], dtype=object))
    return keep, X, lab


def save_names(ds, feats, typ_map, note):
    json.dump({"source": note, "names": {str(i + 1): n for i, n in enumerate(feats)},
               "labels": typ_map}, open(f"features/{ds}_feature_names.json", "w"), indent=1)


def classes_of(lab):
    return ["Benign"] + sorted(set(lab) - {"Benign"})


def build_cicids():
    import audit_loaders as L
    rel = L.LOADERS["cicids2017c"]()
    df = rel.df
    typ = np.where(df["_y"].astype(str) == "Benign", df["_type"].astype(str), df["_type"].astype(str))
    btype = pd.Series(typ[df["_y"].astype(str).values == "Benign"]).mode().iloc[0]   # BENIGN
    typ = np.where(typ == btype, rel.benign, typ)
    keep, X, lab = prepare(rel, "cicids2017c", typ)
    d = df.loc[keep, ["_t"]].copy()
    d["_lab"] = lab
    d["_day"] = pd.to_datetime(d["_t"], unit="s").dt.day_name().str.lower()
    # per label, its first 70 % in time train and the rest test; benign likewise per day
    # (change log 2026-11-03: the interval rule left DDoS and DoS GoldenEye without training)
    test = np.zeros(len(d), bool)
    t, day = d["_t"].to_numpy(), d["_day"].to_numpy()
    for c in np.unique(lab):
        for dd in (np.unique(day) if c == "Benign" else [None]):
            ii = np.flatnonzero((lab == c) & ((day == dd) if dd is not None else True))
            ii = ii[np.argsort(t[ii], kind="stable")]
            test[ii[int(0.7 * len(ii)):]] = True
    return finish("ev1_cicids", rel, X, lab, ~test, test, d["_t"].to_numpy(), typ[keep],
                  "CICIDS2017_improved.zip (Liu et al., 27 April 2023); evaluation-v1")


def build_morris3():
    import audit_loaders as L
    rel = L.LOADERS["morris3"]()
    typ = rel.df["_type"].astype(str).values
    keep, X, lab = prepare(rel, "morris3", typ)
    pos = np.arange(len(rel.df))[keep]                 # file order
    train = np.zeros(len(lab), bool)
    for c in set(lab):
        ii = np.where(lab == c)[0]
        train[ii[: int(0.7 * len(ii))]] = True
    return finish("ev1_morris3", rel, X, lab, train, ~train, pos.astype(np.float64), typ[keep],
                  "water_final.arff (Morris and Gao 2014, Web Archive copy); evaluation-v1")


def build_morris4():
    """Morris-4 (amendment of 2026-11-03, replacing Morris-3): 30 specific attacks; it has a
    clock, so for each label its first 70 % in time train."""
    import audit_loaders as L
    rel = L.LOADERS["morris4"]()
    typ = rel.df["_type"].astype(str).values
    keep, X, lab = prepare(rel, "morris4", typ)
    t = rel.df["_t"].to_numpy(np.float64)[keep]
    train = np.zeros(len(lab), bool)
    for c in set(lab):
        ii = np.flatnonzero(lab == c)
        ii = ii[np.argsort(t[ii], kind="stable")]
        train[ii[: int(0.7 * len(ii))]] = True
    return finish("ev1_morris4", rel, X, lab, train, ~train, t, typ[keep],
                  "IanArffDataset.arff (Morris, Thornton and Turnipseed 2015, Web Archive copy); "
                  "evaluation-v1, amendment of 2026-11-03")


def build_ciciot():
    parts = sorted(glob.glob(os.path.expanduser("~/datasets/ciciot2023/part-*.csv")))
    fr = []
    for k, p in enumerate(parts):
        d = pd.read_csv(p, low_memory=False)
        d["_part"] = k
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    lab0 = [c for c in df.columns if c != "_part"][-1]
    feats = [c for c in df.columns if c not in (lab0, "_part")]
    ex = excluded("ciciot2023")
    typ = df[lab0].astype(str).values
    keep = ~np.isin(typ, list(ex))
    print(f"CICIoT2023: {len(df):,} rows; excluded types {sorted(ex)}; kept {int(keep.sum()):,}")
    X = np.array(df.loc[keep, feats].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64))
    X[~np.isfinite(X)] = np.nan
    t = typ[keep]
    btype = "BenignTraffic" if "BenignTraffic" in set(t) else pd.Series(t).mode().iloc[0]
    lab = np.where(t == btype, "Benign", np.array([safe(v) for v in t], dtype=object))
    part = df.loc[keep, "_part"].to_numpy()

    class R:                                            # minimal stand-in for a Release
        name, benign = "CICIoT2023", btype
    R.feats = feats
    out = finish("ev1_ciciot", R, X, lab, part < 7, part >= 7, None, t,
                 "CICIoT2023 CSV parts 0-9 (the 10 screened files); evaluation-v1")
    tr = part < 7
    bal = np.zeros(len(lab), bool)
    rng = np.random.default_rng(42)
    for c in set(lab[tr]):
        ii = np.where(tr & (lab == c))[0]
        bal[rng.choice(ii, min(len(ii), BAL_CAP), replace=False)] = True
    write_arff("all_in_one_ev1_ciciot_trainbal.csv", X[bal], lab[bal], classes_of(lab), "ev1_ciciot")
    return out


def finish(ds, rel, X, lab, train, test, order, typ, note):
    classes = classes_of(lab)
    typ_map = {str(v): (str(s)) for v, s in sorted(set(zip(typ, lab)))}
    save_names(ds, rel.feats, typ_map, note)
    print(f"{ds}: {X.shape[1]} features; train {int(train.sum()):,} / test {int(test.sum()):,}; "
          f"{len(classes) - 1} attack types")
    print("  train per label:", pd.Series(lab[train]).value_counts().to_dict())
    print("  test  per label:", pd.Series(lab[test]).value_counts().to_dict())
    write_arff(f"all_in_one_{ds}_train.csv", X[train], lab[train], classes, ds)
    o = write_arff(f"all_in_one_{ds}_test.csv", X[test], lab[test], classes, ds)
    if order is not None:
        np.save(f"all_in_one_{ds}_test_order.npy", order[test][o])
    return ds


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    todo = sys.argv[1:] or ["ev1_morris4", "ev1_ciciot", "ev1_cicids"]
    for ds in todo:
        {"ev1_cicids": build_cicids, "ev1_morris3": build_morris3, "ev1_morris4": build_morris4,
         "ev1_ciciot": build_ciciot}[ds]()
