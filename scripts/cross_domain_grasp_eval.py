#!/usr/bin/env python3
"""Effect of the GRASP selections on CICIoT2023 and the corrected CICIDS2017, with the
protocol of cross_domain_results.py (14 specialists, two per category, k-of-n fusion,
same split, same XGBoost settings). Three feature modes:

  all        every eligible feature (the earlier run)
  per-attack each specialist uses the binary GRASP selection of its own category
  aggregate  every specialist uses the union (penalized global U per-category supersets)

Reports GL (retained union of the 14) recall / FPR / F1 at the ERENO prevalence for k>=1
and k>=2, per-category recall at k>=2, and FL k>=2 with 7 and 3 nodes.

  python scripts/cross_domain_grasp_eval.py [ciciot2023|cicids2017c]
Out: results/cross_domain_grasp_<ds>.txt
"""
import glob
import io
import json
import os
import sys
import zipfile
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402
from inspect_ciciot2023 import category  # noqa: E402


def train_fs(X, cat, cats, cols_by_cat):
    """cross_domain_results.train_specialists with a feature subset per category."""
    rng = np.random.default_rng(C.SEED)
    ben = np.where(cat == "Benign")[0]
    shares = np.array_split(rng.permutation(ben), C.N)
    halves = {c: np.array_split(rng.permutation(np.where(cat == c)[0]), 2) for c in cats}
    models = []
    for i in range(C.N):
        c = cats[i % len(cats)]
        pos, neg = halves[c][i // len(cats)], shares[i]
        rows = np.r_[pos, neg]
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        p = dict(C.XGB)
        if len(pos) and len(neg):
            p["scale_pos_weight"] = len(neg) / len(pos)
        cols = cols_by_cat[c]
        models.append((xgb.train(p, xgb.DMatrix(X[np.ix_(rows, cols)], label=y),
                                 num_boost_round=10), cols))
    return models


def votes(models, X):
    return np.column_stack([(m.predict(xgb.DMatrix(X[:, cols])) >= 0.5).astype(np.int8)
                            for m, cols in models])


def report(tag, V, cat_te, cats):
    atk = cat_te != "Benign"
    out = []
    for k in (1, 2):
        g = C.stats(V.sum(1) >= k, atk)
        out.append(f"GL k>={k}: recall {g['recall']:6.2f}%  FPR {g['FPR']:7.3f}%  "
                   f"F1@13.75 {g['F1@13.75']:6.2f}")
    f = V.sum(1) >= 2
    pc = "  ".join(f"{c}={100 * f[cat_te == c].mean():.1f}" for c in cats)
    fl = "  ".join(f"FL k>=2 {n} nodes: recall {C.stats(V[:, :n].sum(1) >= 2, atk)['recall']:.1f}%"
                   for n in (7, 3))
    print(f"\n[{tag}]\n  " + "\n  ".join(out) + f"\n  per category (k>=2): {pc}\n  {fl}")


def load(ds):
    if ds == "ciciot2023":
        parts = sorted(glob.glob(C.D + "ciciot2023/part-*.csv"))
        tr = pd.concat([pd.read_csv(p) for p in parts[:7]], ignore_index=True)
        te = pd.concat([pd.read_csv(p) for p in parts[7:]], ignore_index=True)
        lab = tr.columns[-1]
        names = [c for c in tr.columns if c not in (lab, "IAT")]
        return (tr[names].to_numpy(np.float32), tr[lab].map(category).to_numpy(),
                te[names].to_numpy(np.float32), te[lab].map(category).to_numpy(), names,
                ["DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"])
    z = zipfile.ZipFile(C.D + "cicids2017_improved/CICIDS2017_improved.zip")
    df = pd.concat([pd.read_csv(z.open(f"{d}.csv"), low_memory=False,
                                dtype={c: "str" for c in C.TEXT}).assign(_day=d)
                    for d in ["monday", "tuesday", "wednesday", "thursday", "friday"]],
                   ignore_index=True)
    df["_cat"] = df["Label"].map(C.cat_cicids)
    df = df[df["_cat"] != "drop"].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="mixed")
    names = [c for c in df.columns if c not in C.TEXT + C.IDENT + C.FINGERPRINT + ["_day", "_cat", "_ts"]]
    X = df[names].apply(pd.to_numeric, errors="coerce").astype("float32").to_numpy()
    X[~np.isfinite(X)] = np.nan
    test = np.zeros(len(df), bool)
    for d, g in df.groupby("_day"):
        if d != "monday":
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
    return X[train], cat[train], X[test], cat[test], names, C.CATS_C


def selections(ds):
    """Per-category and aggregate names; the global GRASP joins the aggregate when present."""
    f = f"features/all_in_one_{ds}_train_combined.json"
    if os.path.exists(f):
        return json.load(open(f))
    fn = json.load(open(f"features/{ds}_feature_names.json"))["names"]
    s = json.load(open(f"features/por_ataque/grasppen005_{ds}_SUPERSET.json"))
    g = f"features/global_penalizado_{ds}_L0.05.json"
    glob_ = json.load(open(g))["features"] if os.path.exists(g) else []
    print("(global GRASP not finished: aggregate = union of the per-category selections)"
          if not glob_ else "")
    return {"names": [fn[str(i)] for i in sorted(set(s["superset_grasp"]) | set(glob_))],
            "per_category_names": {k: [fn[str(i)] for i in v] for k, v in s["per_attack"].items()}}


def main(ds):
    Xtr, ctr, Xte, cte, names, cats = load(ds)
    comb = selections(ds)
    idx = {n: i for i, n in enumerate(names)}
    allc = list(range(len(names)))
    per = {c: [idx[n] for n in comb["per_category_names"].get(c, [])] or allc for c in cats}
    agg = [idx[n] for n in comb["names"]]
    print(f"{ds}: train {len(ctr):,}, test {len(cte):,}; features all={len(allc)}, "
          f"aggregate={len(agg)}; per category: " + ", ".join(f"{c}={len(per[c])}" for c in cats))
    for c in cats:
        print(f"  {c:<13} {comb['per_category_names'].get(c, '(none -> all)')}")
    print(f"  aggregate: {comb['names']}")
    for tag, cbc in (("all features", {c: allc for c in cats}),
                     ("per-attack GRASP", per),
                     ("aggregate GRASP", {c: agg for c in cats})):
        report(tag, votes(train_fs(Xtr, ctr, cats, cbc), Xte), cte, cats)


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    for ds in (sys.argv[1:] or ["ciciot2023", "cicids2017c"]):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(ds)
        open(f"results/cross_domain_grasp_{ds}.txt", "w").write(buf.getvalue())
        print(buf.getvalue())
