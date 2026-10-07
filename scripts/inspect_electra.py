#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on Electra Modbus
(Perales Gomez et al., 2019; electric traction substation of a high-speed railway, 5 PLCs
and a SCADA system; perception.inf.um.es/ICS-datasets, checksums in
results/electra_md5sums.txt). One row per Modbus packet:
Time, smac, dmac, sip, dip, request, fc, error, address, data, label.

  (i)   STRUCTURE   fields per line, time column, value ranges
  (ii)  LABELS      label set, duplicates and vectors with >1 label on the content fields
                    (request, fc, error, address, data); MITM_UNALTERED vs NORMAL
  (iii) TIME        order of the Time column, span per label, interleaving
  (iv)  ATTRIBUTION MAC/IP pairs per label (the MitM host)
  (v)   PREVALENCE  label shares
  (vi)  SHORTCUT    XGBoost with/without MAC/IP and time, random and time-ordered splits

  python scripts/inspect_electra.py ~/datasets/electra/electra_modbus.csv
Out: results/electra_audit.txt
"""
import io
import os
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

CONTENT = ["request", "fc", "error", "address", "data"]
IDENT = ["smac", "dmac", "sip", "dip"]


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main(path):
    section("(i) STRUCTURE")
    nf = subprocess.run(["awk", "-F,", "NR>1{c[NF]++} END{for(k in c) print k\":\"c[k]}", path],
                        capture_output=True, text=True).stdout.split()
    print("fields per line:", nf)
    df = pd.read_csv(path, dtype={c: "category" for c in IDENT + ["label"]}
                     | {c: "int64" for c in CONTENT} | {"Time": "int64"})
    n = len(df)
    print(f"rows={n:,}")
    for c in ["Time"] + CONTENT:
        print(f"  {c:<8} min={df[c].min()} max={df[c].max()} distinct={df[c].nunique():,}")
    dt = np.diff(df["Time"].values)
    print(f"  Time non-decreasing in file order: {pct(int((dt >= 0).sum()), len(dt))} of steps; "
          f"rows with Time==0: {int((df['Time'] == 0).sum()):,}")
    print("  first values:", df["Time"].head(8).tolist())

    section("(ii) LABELS")
    vc = df["label"].value_counts()
    print("labels:", vc.to_dict())
    h = pd.util.hash_pandas_object(df[CONTENT], index=False)
    print(f"distinct content vectors: {h.nunique():,}; duplicates: {pct(int(h.duplicated().sum()), n)}")
    pairs = pd.DataFrame({"h": h.values, "l": df["label"].cat.codes.values}).drop_duplicates()
    multi = pairs["h"][pairs["h"].duplicated()].unique()
    rows = np.isin(h.values, multi)
    print(f"vectors with >1 label: {len(multi):,} (rows {int(rows.sum()):,}, {pct(int(rows.sum()), n)})")
    for l in vc.index:
        m = (df["label"] == l).values
        print(f"  {l:<24} rows {int(m.sum()):>10,}  in multi-label vectors {pct(int((m & rows).sum()), int(m.sum()))}")
    nh = set(h[(df["label"] == "NORMAL").values])
    for l in vc.index:
        if l != "NORMAL":
            m = (df["label"] == l).values
            print(f"  {l:<24} rows whose content vector also occurs as NORMAL: "
                  f"{pct(int(h[m].isin(nh).sum()), int(m.sum()))}")

    section("(iii) TIME")
    order = np.argsort(df["Time"].values, kind="stable")
    t = df["Time"].values
    print(df.groupby("label", observed=True)["Time"].agg(["min", "max", "count"]).to_string())
    tn = np.sort(t[(df["label"] == "NORMAL").values])
    for l in vc.index:
        if l == "NORMAL":
            continue
        g = t[(df["label"] == l).values]
        lo, hi = g.min(), g.max()
        ins = int(np.searchsorted(tn, hi, "right") - np.searchsorted(tn, lo, "left"))
        print(f"  {l:<24} NORMAL rows inside its span: {ins:,} (share {pct(ins, ins + len(g))})")

    section("(iv) ATTRIBUTION")
    for l in vc.index:
        g = df[df["label"] == l]
        sm = g["smac"].value_counts().head(3)
        dm = g["dmac"].value_counts().head(3)
        print(f"  {l:<24} src MAC: " + ", ".join(f"{k} {pct(v, len(g))}" for k, v in sm.items())
              + " | dst MAC: " + ", ".join(f"{k} {pct(v, len(g))}" for k, v in dm.items()))
    for c in ("smac", "dmac", "sip", "dip"):
        mj = pd.crosstab(df[c], df["label"]).max(axis=1).sum()
        print(f"  {c} alone -> label (majority per value): {pct(int(mj), n)}")
    mac = df["smac"].cat.codes.astype(np.int64) * 10_000 + df["dmac"].cat.codes.astype(np.int64)
    mj = pd.crosstab(mac.values, df["label"].values).max(axis=1).sum()
    print(f"  (smac, dmac) pair alone -> label: {pct(int(mj), n)}")

    section("(v) PREVALENCE")
    for k, v in vc.items():
        print(f"  {k:<24}{v:>12,}  {pct(v, n)}")

    section("(vi) SHORTCUT PROBE — XGBoost macro-F1 (dedup per label on all fields; <= 100k rows per label)")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    X = df[CONTENT].copy()
    for c in IDENT:
        X[c] = df[c].cat.codes
    X["Time"] = df["Time"]
    hall = pd.util.hash_pandas_object(X, index=False)
    ded = ~hall.duplicated().values
    rng = np.random.RandomState(42)
    idx = np.sort(np.concatenate([rng.choice(np.where(ded & (df["label"] == l).values)[0],
                                             min(100_000, int((ded & (df["label"] == l).values).sum())),
                                             replace=False) for l in vc.index]))
    Xs = X.iloc[idx].reset_index(drop=True)
    y, names = pd.factorize(df["label"].iloc[idx].astype(str), sort=True)
    sets = {"content only": CONTENT, "content + MAC/IP": CONTENT + IDENT,
            "content + MAC/IP + Time": CONTENT + IDENT + ["Time"]}
    a, b = train_test_split(np.arange(len(Xs)), test_size=0.3, stratify=y, random_state=42)
    ta, tb = [], []
    for c in range(len(names)):
        ii = np.where(y == c)[0]
        ii = ii[np.argsort(Xs["Time"].values[ii], kind="stable")]
        cut = int(0.7 * len(ii)); ta += list(ii[:cut]); tb += list(ii[cut:])
    for split, (tr, te) in (("random 70/30", (a, b)),
                            ("time-ordered 70/30 per label", (np.array(ta), np.array(tb)))):
        print(f"-- {split}")
        for sname, cols in sets.items():
            prm = {"objective": "multi:softmax", "num_class": len(names), "max_depth": 6,
                   "eta": 0.3, "tree_method": "hist", "nthread": 4, "seed": 42}
            m = xgb.train(prm, xgb.DMatrix(Xs.iloc[tr][cols], label=y[tr]), num_boost_round=100)
            p = m.predict(xgb.DMatrix(Xs.iloc[te][cols])).astype(int)
            per = f1_score(y[te], p, average=None, labels=range(len(names)), zero_division=0)
            print(f"  {sname:<26} macro-F1={per.mean():.3f}  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, per)))


if __name__ == "__main__":
    p = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/electra/electra_modbus.csv")
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(p)
    open("results/electra_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
