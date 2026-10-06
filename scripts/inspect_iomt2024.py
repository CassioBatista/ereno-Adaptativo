#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on CIC-BCCC-NRC-IoMT-2024,
the CICFlowMeter re-extraction of CICIoMT2024 (Dadkhah et al., Internet of Things 2024)
in the CIC-BCCC-NRC TabularIoTAttacks-2024 collection. Audited copy: Kaggle mirror
kabeleswarpe/cic-bccc-nrc-tabulariotattacks-2024 (15 CSVs, one per class; MD5SUMS kept
next to the files).

  (i)   STRUCTURE   fields per line, header per file, NaN/inf, value kinds
  (ii)  LABELS      'Attack Name'/'Label' vs file, duplicates, vectors with >1 label
  (iii) TIME        timestamps per class; are attacks interleaved with benign flows?
  (iv)  ATTRIBUTION source addresses per class
  (v)   PREVALENCE  class shares, benign volume and span
  (vi)  SHORTCUT    XGBoost with/without identifiers and Init Win Bytes, random and
                    time-ordered splits

  python scripts/inspect_iomt2024.py ~/datasets/cic_bccc_nrc_iomt2024
Out: results/iomt2024_audit.txt
"""
import csv
import glob
import io
import os
import re
import sys
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

IDENT = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]
TEXT = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Attack Name", "Label"]
FINGERPRINT = ["FWD Init Win Bytes", "Bwd Init Win Bytes"]


def category(name):
    l = name.lower()
    if l.startswith("benign"):
        return "Benign"
    if l.startswith("mqtt"):
        return "MQTT"
    if l.startswith("ddos"):
        return "DDoS"
    if l.startswith("dos"):
        return "DoS"
    if l.startswith("recon"):
        return "Recon"
    if "spoof" in l:
        return "Spoofing"
    return "??" + name


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main(d):
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    section(f"(i) STRUCTURE — {len(files)} files")
    frames, headers = [], set()
    for p in files:
        with open(p, newline="", encoding="latin-1") as f:
            r = csv.reader(f)
            h = next(r)
            headers.add(tuple(h))
            fc = Counter(len(row) for row in r)
        cols = h
        df = pd.read_csv(p, low_memory=False, encoding="latin-1",
                         dtype={c: ("str" if c in TEXT else "float32") for c in cols})
        df["_file"] = os.path.basename(p)[:-4]
        frames.append(df)
        print(f"  {df['_file'].iloc[0]:<28} rows={len(df):>9,}  fields per line={dict(fc)}")
    print("same header in all files:", len(headers) == 1)
    df = pd.concat(frames, ignore_index=True)
    del frames
    n = len(df)
    feats = [c for c in df.columns if c not in TEXT + ["_file"]]
    num = df[feats]
    print(f"total rows={n:,}; numeric columns={len(feats)}; NaN={int(num.isna().sum().sum())}; "
          f"inf={int(np.isinf(num.values).sum())}")
    ipv4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    for c in ("Src IP", "Dst IP"):
        print(f"  {c}: IPv4-shaped {pct(int(df[c].str.match(ipv4).sum()), n)}")
    for c in ("Src Port", "Dst Port"):
        print(f"  {c}: within 0..65535 {pct(int(num[c].between(0, 65535).sum()), n)}")
    print("  Protocol values:", num["Protocol"].value_counts().to_dict())
    ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    print(f"  Timestamp parsed (dd/mm/yyyy hh:mm:ss AM/PM): {pct(int(ts.notna().sum()), n)}; "
          f"e.g. {df['Timestamp'].iloc[0]}")
    df["_ts"] = ts

    section("(ii) LABELS")
    df["_cat"] = df["_file"].map(category)
    print("file -> 'Attack Name' / 'Label' values:")
    for f, g in df.groupby("_file"):
        print(f"  {f:<28} {g['Attack Name'].value_counts().head(3).to_dict()}  "
              f"{g['Label'].value_counts().head(3).to_dict()}")
    print("exact duplicate rows:", f"{int(df.drop(columns=['_ts']).duplicated().sum()):,}")
    fe = [c for c in feats if c not in IDENT]
    h = pd.util.hash_pandas_object(num[fe], index=False)
    print(f"duplicates without identifiers: {int(h.duplicated().sum()):,} ({pct(int(h.duplicated().sum()), n)})")
    for key in ("_file", "_cat"):
        g = df.groupby(h.values)[key].nunique()
        multi = g[g > 1]
        rows = h.isin(multi.index).values
        print(f"vectors with >1 {'class' if key == '_file' else 'category'}: {len(multi):,} "
              f"(rows {int(rows.sum()):,}); top: {df.loc[rows, key].value_counts().head(6).to_dict()}")
    ded = df.loc[~h.duplicated().values]
    print("unique rows per class:", ded["_file"].value_counts().to_dict())

    section("(iii) TIME")
    sp = df.groupby("_file")["_ts"].agg(["min", "max", "count"])
    print(sp.to_string())
    b = df[df["_cat"] == "Benign"]
    for f, g in df[df["_cat"] != "Benign"].groupby("_file"):
        lo, hi = g["_ts"].min(), g["_ts"].max()
        inside = int(((b["_ts"] >= lo) & (b["_ts"] <= hi)).sum())
        print(f"  {f:<28} benign flows inside its window: {inside:,} "
              f"(benign share {pct(inside, inside + len(g))})")
    print("distinct capture dates:", sorted(df["_ts"].dt.date.dropna().unique().astype(str))[:20])

    section("(iv) ATTRIBUTION")
    for f, g in df.groupby("_file"):
        top = g["Src IP"].value_counts().head(3)
        print(f"  {f:<28} src={g['Src IP'].nunique():>6,} dst={g['Dst IP'].nunique():>6,}  top src: "
              + ", ".join(f"{i} ({pct(v, len(g))})" for i, v in top.items()))
    bsrc = set(b["Src IP"])
    print(f"benign: {b['Src IP'].nunique()} source and {b['Dst IP'].nunique()} destination addresses; "
          f"private 192.168.x sources: {sum(1 for x in bsrc if str(x).startswith('192.168.'))}")
    maj = df.groupby("Src IP")["_cat"].agg(lambda s: s.value_counts().iloc[0])
    print(f"source address alone -> category (majority per address): {pct(int(maj.sum()), n)}")

    section("(v) PREVALENCE")
    for k, v in df["_cat"].value_counts().items():
        print(f"  {k:<10}{v:>10,}  {pct(v, n)}")
    print(f"  benign span: {b['_ts'].min()} .. {b['_ts'].max()}")

    section("(vi) SHORTCUT PROBE — XGBoost macro-F1 (dedup; per class up to 100k rows)")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    rng = np.random.RandomState(42)
    idx = np.concatenate([rng.choice(g.index, min(100_000, len(g)), replace=False)
                          for _, g in ded.groupby("_cat")])
    d2 = df.loc[idx]
    X = num.loc[idx].replace([np.inf, -np.inf], np.nan).fillna(-1)
    y, names = pd.factorize(d2["_cat"], sort=True)
    sets = {"with identifiers (ports)": feats,
            "without identifiers": fe,
            "without identifiers and Init Win Bytes": [c for c in fe if c not in FINGERPRINT]}
    a, bb = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=42)
    order = d2["_ts"].values
    ta, tb = [], []
    for c in range(len(names)):
        ii = np.where(y == c)[0]
        ii = ii[np.argsort(order[ii], kind="stable")]
        cut = int(0.7 * len(ii))
        ta += list(ii[:cut]); tb += list(ii[cut:])
    for split, (tr, te) in (("random 70/30", (a, bb)),
                            ("time-ordered 70/30 per category", (np.array(ta), np.array(tb)))):
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
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/cic_bccc_nrc_iomt2024")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    os.makedirs("results", exist_ok=True)
    open("results/iomt2024_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
