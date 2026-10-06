#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on the corrected CICIDS2017
(Engelen et al., SPW 2021; Liu et al., CNS 2022): CICIDS2017_improved.zip, five days.

  (i)   STRUCTURE   fields per line, header per day, NaN/inf, value kinds (IPs, ports,
                    protocol, timestamps)
  (ii)  LABELS      labels and 'Attempted' flows, duplicates, vectors with >1 label
  (iii) TIME        timestamps per day; attacks interleaved with benign flows?
  (iv)  ATTRIBUTION source addresses per class; does the source alone give the class?
  (v)   PREVALENCE  per day and overall; benign-only day
  (vi)  SHORTCUT    XGBoost with/without identifiers and the initial-window fingerprint,
                    on a random split and on a time-ordered split within each class

  python scripts/inspect_cicids2017c.py ~/datasets/cicids2017_improved/CICIDS2017_improved.zip
Out: results/cicids2017c_audit.txt
"""
import csv
import io
import os
import re
import sys
import zipfile
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
IDENT = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]
TEXT = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "Attempted Category"]
FINGERPRINT = ["FWD Init Win Bytes", "Bwd Init Win Bytes"]     # host/OS artefacts


def category(lbl):
    l = lbl.lower()
    if "attempted" in l or l == "benign":
        return "Benign"
    if l.startswith("ddos"):
        return "DDoS"
    if l == "heartbleed":
        return "Heartbleed"
    if l.startswith("dos"):
        return "DoS"
    if l.startswith("infiltration - portscan"):
        return "Infiltration-Portscan"
    if l.startswith("infiltration"):
        return "Infiltration"
    if "portscan" in l:
        return "PortScan"
    if "patator" in l:
        return "BruteForce"
    if l.startswith("web attack"):
        return "Web"
    if l.startswith("bot"):
        return "Bot"
    return "??" + lbl


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main(zpath):
    z = zipfile.ZipFile(zpath)
    section("(i) STRUCTURE")
    frames, headers = [], {}
    for d in DAYS:
        raw = z.read(f"{d}.csv").decode("utf-8", "replace")
        rows = csv.reader(io.StringIO(raw))
        h = next(rows)
        headers[d] = tuple(h)
        fc = Counter(len(r) for r in rows)
        df = pd.read_csv(io.StringIO(raw), low_memory=False,
                         dtype={c: "str" for c in TEXT})
        df["_day"] = d
        frames.append(df)
        print(f"{d:<10} rows={len(df):>9,}  fields per line={dict(fc)}")
        del raw
    print("same header on all days:", len(set(headers.values())) == 1)
    df = pd.concat(frames, ignore_index=True)
    del frames
    n = len(df)
    feats = [c for c in df.columns if c not in TEXT + ["_day"]]
    num = df[feats].apply(pd.to_numeric, errors="coerce").astype("float64")
    print(f"total rows={n:,}; numeric columns={len(feats)}; NaN={int(num.isna().sum().sum())}; "
          f"inf={int(np.isinf(num.values).sum())}")
    ipv4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    for c in ("Src IP", "Dst IP"):
        print(f"  {c}: IPv4-shaped {pct(int(df[c].str.match(ipv4).sum()), n)}")
    for c in ("Src Port", "Dst Port"):
        print(f"  {c}: within 0..65535 {pct(int(num[c].between(0, 65535).sum()), n)}")
    print("  Protocol values:", num["Protocol"].value_counts().to_dict())
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed")
    print(f"  Timestamp parsed: {pct(int(ts.notna().sum()), n)}; e.g. {df['Timestamp'].iloc[0]}")
    df["_ts"] = ts

    section("(ii) LABELS")
    df["_cat"] = df["Label"].map(category)
    vc = df["Label"].value_counts()
    for k, v in vc.items():
        print(f"  {k:<44}{v:>10,}  {category(k)}")
    att = df["Label"].str.contains("Attempted", case=False)
    print(f"Attempted flows: {int(att.sum()):,}; 'Attempted Category' values: "
          f"{df.loc[att, 'Attempted Category'].value_counts().head(8).to_dict()}")
    print("exact duplicate rows (all columns):", f"{int(df.drop(columns=['_ts']).duplicated().sum()):,}")
    fe = [c for c in feats if c not in IDENT]
    h = pd.util.hash_pandas_object(num[fe], index=False)
    print(f"duplicates without identifiers: {int(h.duplicated().sum()):,} ({pct(int(h.duplicated().sum()), n)})")
    for key in ("Label", "_cat"):
        g = df.groupby(h.values)[key].nunique()
        multi = g[g > 1]
        rows = h.isin(multi.index).values
        print(f"vectors with >1 {'label' if key == 'Label' else 'category'}: {len(multi):,} "
              f"(rows {int(rows.sum()):,}); top: {df.loc[rows, key].value_counts().head(6).to_dict()}")

    section("(iii) TIME")
    for d in DAYS:
        s = df[df["_day"] == d]
        print(f"{d:<10} {s['_ts'].min()} .. {s['_ts'].max()}  label runs in file order "
              f"{int(s['Label'].ne(s['Label'].shift()).sum()):,}")
    print("per attack category: time window and share of benign flows inside it (same day)")
    for k in sorted(df["_cat"].unique()):
        if k == "Benign":
            continue
        s = df[df["_cat"] == k]
        lo, hi = s["_ts"].min(), s["_ts"].max()
        days = s["_day"].unique()
        inwin = df[(df["_day"].isin(days)) & (df["_ts"] >= lo) & (df["_ts"] <= hi)]
        print(f"  {k:<22} days={list(days)}  [{lo} .. {hi}]  flows in window={len(inwin):,}  "
              f"benign share {pct(int((inwin['_cat'] == 'Benign').sum()), len(inwin))}")

    section("(iv) ATTRIBUTION")
    for k in sorted(df["_cat"].unique()):
        s = df[df["_cat"] == k]
        top = s["Src IP"].value_counts().head(3)
        print(f"  {k:<22} src={s['Src IP'].nunique():>6,}  top: "
              + ", ".join(f"{i} ({pct(v, len(s))})" for i, v in top.items()))
    benign_src = set(df.loc[df["_cat"] == "Benign", "Src IP"])
    for k in sorted(df["_cat"].unique()):
        if k != "Benign":
            s = df[df["_cat"] == k]
            print(f"  {k:<22} attack flows whose source also sends benign flows: "
                  f"{pct(int(s['Src IP'].isin(benign_src).sum()), len(s))}")
    maj = df.groupby("Src IP")["_cat"].agg(lambda s: s.value_counts().iloc[0])
    print(f"source address alone -> category (majority per address): {pct(int(maj.sum()), n)}")

    section("(v) PREVALENCE")
    for d in DAYS:
        s = df[df["_day"] == d]
        print(f"  {d:<10} benign {pct(int((s['_cat'] == 'Benign').sum()), len(s))} of {len(s):,}")
    print("  overall:", {k: pct(v, n) for k, v in df["_cat"].value_counts().items()})
    mon = df[df["_day"] == "monday"]
    print(f"  monday: {mon['_cat'].value_counts().to_dict()}; span {mon['_ts'].min()} .. {mon['_ts'].max()}")

    section("(vi) SHORTCUT PROBE — XGBoost macro-F1 (dedup; all attacks + 400k benign)")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    keep = ~h.duplicated().values & (df["_cat"] != "Heartbleed").values
    d2 = df.loc[keep]
    ben = d2.index[d2["_cat"] == "Benign"]
    rng = np.random.RandomState(42)
    idx = np.r_[d2.index[d2["_cat"] != "Benign"], rng.choice(ben, min(400_000, len(ben)), replace=False)]
    d2 = d2.loc[idx]
    X = num.loc[idx].replace([np.inf, -np.inf], np.nan).fillna(-1)
    y, names = pd.factorize(d2["_cat"], sort=True)
    sets = {"with identifiers (ports, id)": [c for c in feats],
            "without identifiers": fe,
            "without identifiers and Init Win Bytes": [c for c in fe if c not in FINGERPRINT]}
    # random split
    a, b = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=42)
    # time-ordered split: first 70 % of each category (by timestamp) trains, last 30 % tests
    order = d2["_ts"].values
    ta, tb = [], []
    for c in range(len(names)):
        ii = np.where(y == c)[0]
        ii = ii[np.argsort(order[ii], kind="stable")]
        cut = int(0.7 * len(ii))
        ta += list(ii[:cut]); tb += list(ii[cut:])
    for split, (tr, te) in (("random 70/30", (a, b)), ("time-ordered 70/30 per category", (np.array(ta), np.array(tb)))):
        print(f"-- {split}")
        for sname, cols in sets.items():
            m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist", n_jobs=4,
                                  random_state=42)
            m.fit(X.iloc[tr][cols], y[tr])
            p = m.predict(X.iloc[te][cols])
            per = f1_score(y[te], p, average=None, labels=range(len(names)), zero_division=0)
            print(f"  {sname:<40} macro-F1={per.mean():.3f}  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, per)))
            if split.startswith("random") and sname == "without identifiers":
                g = pd.Series(m.get_booster().get_score(importance_type="gain"))
                print("    top gain:", list(g.sort_values(ascending=False).head(6).index))


if __name__ == "__main__":
    p = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else
                           "~/datasets/cicids2017_improved/CICIDS2017_improved.zip")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(p)
    os.makedirs("results", exist_ok=True)
    open("results/cicids2017c_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
