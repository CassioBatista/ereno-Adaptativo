#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on a sample of the
CICIoT2023 CSV release (Neto et al., Sensors 2023): the first N of its 169 part files.

  (i)   STRUCTURE   fields per line, header agreement across parts, NaN/inf, value kinds
  (ii)  LABELS      classes and 7 categories, duplicates, vectors with >1 label
  (iii) TIME        no timestamp column; does IAT behave like a capture clock?
  (iv)  ATTRIBUTION no address column (reported)
  (v)   PREVALENCE  class/category shares in the sample
  (vi)  SHORTCUT    XGBoost with/without IAT and the window-size columns (Number, Weight)

  python scripts/inspect_ciciot2023.py ~/datasets/ciciot2023
Out: results/ciciot2023_audit.txt
"""
import csv
import glob
import io
import os
import sys
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import pandas as pd


def category(lbl):
    l = lbl.lower()
    if l.startswith("benign"):
        return "Benign"
    if l.startswith("ddos"):
        return "DDoS"
    if l.startswith("dos"):
        return "DoS"
    if l.startswith("recon") or l.startswith("vulnerabilityscan"):
        return "Recon"
    if l.startswith("mirai"):
        return "Mirai"
    if l.startswith("mitm") or l.startswith("dns_spoofing"):
        return "Spoofing"
    if l.startswith("dictionarybruteforce"):
        return "BruteForce"
    return "Web"


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main(d):
    parts = sorted(glob.glob(os.path.join(d, "part-*.csv")))
    section(f"(i) STRUCTURE — {len(parts)} part files (sample of the 169 in the release)")
    headers, fc, frames = set(), Counter(), []
    for p in parts:
        with open(p, newline="") as f:
            r = csv.reader(f)
            h = next(r)
            headers.add(tuple(h))
            for row in r:
                fc[len(row)] += 1
        frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    lab = df.columns[-1]
    feats = list(df.columns[:-1])
    n = len(df)
    print(f"rows={n:,}; columns={df.shape[1]}; distinct headers across parts={len(headers)}; "
          f"fields per line={dict(fc)}")
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    print(f"non-numeric cells={int(num.isna().sum().sum() - df[feats].isna().sum().sum())}; "
          f"NaN={int(df[feats].isna().sum().sum())}; inf={int(np.isinf(num.values).sum())}")
    print("columns:", feats)
    for c in ("flow_duration", "IAT", "Number", "Weight", "Tot size", "Rate"):
        if c in df:
            s = num[c]
            print(f"  {c:<14} min={s.min():.6g} median={s.median():.6g} max={s.max():.6g} "
                  f"distinct={s.nunique():,}")

    section("(ii) LABELS")
    df["_cat"] = df[lab].map(category)
    vc = df[lab].value_counts()
    print(f"{df[lab].nunique()} labels:")
    for k, v in vc.items():
        print(f"  {k:<28}{v:>9,}  {pct(v, n):>7}  {category(k)}")
    print("exact duplicate rows:", f"{int(df.duplicated(subset=feats + [lab]).sum()):,}")
    h = pd.util.hash_pandas_object(df[feats], index=False)
    print(f"feature duplicates: {int(h.duplicated().sum()):,} ({pct(int(h.duplicated().sum()), n)})")
    for key in (lab, "_cat"):
        g = df.groupby(h.values)[key].nunique()
        multi = g[g > 1]
        rows = h.isin(multi.index).values
        print(f"vectors with >1 {('label' if key == lab else 'category')}: {len(multi):,} "
              f"(rows {int(rows.sum()):,}); top: "
              f"{df.loc[rows, key].value_counts().head(6).to_dict()}")
    print("label runs in file order:", f"{int(df[lab].ne(df[lab].shift()).sum()):,}",
          "(n = shuffled; #classes = blocks)")

    section("(iii) TIME")
    print("no timestamp column in the release")
    iat = num["IAT"]
    g = df.assign(_iat=iat).groupby(lab)["_iat"].agg(["min", "median", "max"])
    print("IAT per label (if IAT were an inter-arrival time, ranges would overlap across "
          "classes; a capture clock gives one narrow band per capture):")
    print(g.sort_values("median").to_string(float_format=lambda x: f"{x:.6g}"))
    maj = df.assign(_b=iat.round(-3)).groupby("_b")[lab].agg(lambda s: s.value_counts().iloc[0])
    print(f"IAT rounded to 1e3 -> label (majority per bin): accuracy {pct(int(maj.sum()), n)}")

    section("(iv) ATTRIBUTION")
    print("no source/destination address column: alarms cannot be attributed to a device")

    section("(v) PREVALENCE")
    for k, v in df["_cat"].value_counts().items():
        print(f"  {k:<12}{v:>9,}  {pct(v, n)}")

    section("(vi) SHORTCUT PROBE — XGBoost macro-F1 (dedup, stratified 70/30, 7 categories)")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    ded = df.loc[~h.duplicated().values]
    y, names = pd.factorize(ded["_cat"], sort=True)
    X = num.loc[ded.index].replace([np.inf, -np.inf], np.nan).fillna(-1)
    sets = {"all features": feats,
            "without IAT": [c for c in feats if c != "IAT"],
            "without IAT, Number, Weight": [c for c in feats if c not in ("IAT", "Number", "Weight")]}
    a, b = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=42)
    for sname, cols in sets.items():
        m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist", n_jobs=4,
                              random_state=42)
        m.fit(X.iloc[a][cols], y[a])
        p = m.predict(X.iloc[b][cols])
        per = f1_score(y[b], p, average=None, labels=range(len(names)), zero_division=0)
        print(f"  {sname:<30} macro-F1={per.mean():.3f}  "
              + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, per)))
        if sname == "all features":
            gsc = pd.Series(m.get_booster().get_score(importance_type="gain"))
            print("    top gain:", list(gsc.sort_values(ascending=False).head(6).index))
    # Single-feature check: IAT alone
    m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist", n_jobs=4,
                          random_state=42)
    m.fit(X.iloc[a][["IAT"]], y[a])
    per = f1_score(y[b], m.predict(X.iloc[b][["IAT"]]), average=None,
                   labels=range(len(names)), zero_division=0)
    print(f"  {'IAT alone':<30} macro-F1={per.mean():.3f}  "
          + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, per)))


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/ciciot2023")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    os.makedirs("results", exist_ok=True)
    open("results/ciciot2023_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
