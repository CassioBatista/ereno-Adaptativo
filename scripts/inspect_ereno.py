#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on the ERENO release used
by the pipeline: all_in_one_ereno_{train,test}.csv (ARFF, 58 numeric features F1..F58 +
class, converted by scripts/build_ereno_dataset.py; names in features/feature_names.json).

  (i)   STRUCTURE   fields per data line, non-numeric/NaN cells, value kinds per column
  (ii)  LABELS      duplicates, duplicates without absolute time, vectors with >1 label,
                    train/test overlap
  (iii) TIME        absolute-time columns, per-class spans, interleaving in file order
  (iv)  ATTRIBUTION identity fields (removed by the conversion; reported as such)
  (v)   PREVALENCE  class shares, benign span
  (vi)  SHORTCUT    XGBoost with/without absolute time and protocol counters, on a random
                    split of train and on the authors' train/test split

  python scripts/inspect_ereno.py
Out: results/ereno_audit.txt
"""
import io
import json
import os
import sys
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

NAMES = {int(k): v for k, v in json.load(open("features/feature_names.json"))["names"].items()}
ABS_TIME = [1, 38, 39]                 # Time, t, GooseTimestamp
COUNTERS = [40, 41]                    # SqNum, StNum


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def read_arff(path):
    """Header -> column names; data lines checked for field count before parsing."""
    cols, n_fields = [], Counter()
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("@attribute"):
                cols.append(s.split()[1])
            elif s.lower().startswith("@data"):
                break
        for line in f:
            if line.strip():
                n_fields[line.count(",") + 1] += 1
    df = pd.read_csv(path, skiprows=0, header=None, names=cols, comment="@",
                     skip_blank_lines=True, low_memory=False,
                     dtype={c: "float32" for c in cols[:-1]})
    df = df.rename(columns={f"F{i}": f"F{i}_{NAMES[i]}" for i in NAMES})
    return df, len(cols), n_fields


def col(i):
    return f"F{i}_{NAMES[i]}"


def main():
    tr, ncol, fc_tr = read_arff("all_in_one_ereno_train.csv")
    te, _, fc_te = read_arff("all_in_one_ereno_test.csv")
    lab = tr.columns[-1]
    feats = list(tr.columns[:-1])

    section("(i) STRUCTURE")
    for name, df, fc in (("train", tr, fc_tr), ("test", te, fc_te)):
        print(f"{name}: {len(df):,} rows; header fields={ncol}; fields per data line="
              f"{dict(fc)}; NaN cells={int(df[feats].isna().sum().sum())}")
    print("value kinds (train): absolute-time and counter columns")
    for i in ABS_TIME + COUNTERS + [43, 45, 58]:
        s = tr[col(i)]
        print(f"  {col(i):<26} min={s.min():.6g} max={s.max():.6g} "
              f"integer={bool((s.dropna() == s.dropna().round()).all())} "
              f"monotonic-in-file={s.is_monotonic_increasing}")

    section("(ii) LABELS")
    for name, df in (("train", tr), ("test", te)):
        h = pd.util.hash_pandas_object(df[feats], index=False)
        nt = [c for c in feats if int(c.split("_")[0][1:]) not in ABS_TIME]
        h2 = pd.util.hash_pandas_object(df[nt], index=False)
        g = df.groupby(h2.values)[lab].nunique()
        multi = g[g > 1]
        rows = h2.isin(multi.index).values
        print(f"{name}: exact duplicates {int(df.duplicated().sum()):,}; feature duplicates "
              f"{int(h.duplicated().sum()):,}; without absolute time "
              f"{int(h2.duplicated().sum()):,} ({pct(int(h2.duplicated().sum()), len(df))}); "
              f"vectors (no abs. time) with >1 label {len(multi):,}, rows involved "
              f"{int(rows.sum()):,}: {df.loc[rows, lab].value_counts().to_dict()}")
    nt = [c for c in feats if int(c.split("_")[0][1:]) not in ABS_TIME]
    htr = set(pd.util.hash_pandas_object(tr[nt + [lab]], index=False))
    hte = pd.util.hash_pandas_object(te[nt + [lab]], index=False)
    print(f"test rows whose (features without abs. time, label) also occur in train: "
          f"{pct(int(hte.isin(htr).sum()), len(te))}")

    section("(iii) TIME")
    t = col(1)
    for name, df in (("train", tr), ("test", te)):
        runs = int(df[lab].ne(df[lab].shift()).sum())
        print(f"{name}: label runs in file order {runs:,} (classes={df[lab].nunique()}); "
              f"{t} monotonic: {df[t].is_monotonic_increasing}")
        sp = df.groupby(lab)[t].agg(["min", "max", "count"])
        nmin, nmax = sp.loc["normal", "min"], sp.loc["normal", "max"]
        for k, r in sp.iterrows():
            inside = ((df.loc[df[lab] == k, t] >= nmin) & (df.loc[df[lab] == k, t] <= nmax)).mean()
            print(f"  {k:<24} {t} [{r['min']:.3f}, {r['max']:.3f}]  rows={int(r['count']):>9,}  "
                  f"inside benign span {100 * inside:5.1f}%")

    section("(iv) ATTRIBUTION")
    print("identity fields (ethSrc, ethDst, gocbRef, goID, gooseAppid, datSet) were removed by "
          "scripts/build_ereno_dataset.py; this copy cannot attribute an alarm to a "
          "publisher. Auditing attribution needs the original release.")

    section("(v) PREVALENCE")
    for name, df in (("train", tr), ("test", te)):
        vc = df[lab].value_counts()
        print(f"{name}: " + ", ".join(f"{k} {pct(v, len(df))}" for k, v in vc.items()))
        s = df.loc[df[lab] == "normal", t]
        print(f"  benign span of {t}: {s.min():.3f} .. {s.max():.3f} ({s.max() - s.min():.1f} units)")

    section("(vi) SHORTCUT PROBE — XGBoost macro-F1")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    rng = np.random.RandomState(42)
    sets = {
        "all features": feats,
        "without absolute time": [c for c in feats if int(c.split("_")[0][1:]) not in ABS_TIME],
        "without abs. time and SqNum/StNum": [c for c in feats if int(c.split("_")[0][1:])
                                              not in ABS_TIME + COUNTERS],
    }
    yall, names = pd.factorize(pd.concat([tr[lab], te[lab]]), sort=True)
    ytr, yte = yall[:len(tr)], yall[len(tr):]
    sub = rng.choice(len(tr), size=min(600_000, len(tr)), replace=False)
    tsub = rng.choice(len(te), size=min(600_000, len(te)), replace=False)
    for split in ("random 70/30 within train", "authors' train -> test"):
        print(f"-- {split}")
        for sname, cols in sets.items():
            m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist",
                                  n_jobs=4, random_state=42)
            if split.startswith("random"):
                a, b = train_test_split(sub, test_size=0.3, stratify=ytr[sub], random_state=42)
                m.fit(tr.iloc[a][cols], ytr[a])
                p, yt = m.predict(tr.iloc[b][cols]), ytr[b]
            else:
                m.fit(tr.iloc[sub][cols], ytr[sub])
                p, yt = m.predict(te.iloc[tsub][cols]), yte[tsub]
            per = f1_score(yt, p, average=None, labels=range(len(names)), zero_division=0)
            print(f"  {sname:<36} macro-F1={per.mean():.3f}  "
                  + "  ".join(f"{n}={v:.2f}" for n, v in zip(names, per)))


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    os.makedirs("results", exist_ok=True)
    open("results/ereno_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
