#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on CIC IIoT 2025 (DataSense;
Firouzi et al., Electronics 14:4095, 2025): processed per-device time windows that join
sensor logs (log_*) and network aggregates (network_*). Audited copy: Kaggle mirror
baderbotaiban/cic-iiot-2025-datasense-csv-mirror, 1 s and 10 s windows (MD5SUMS kept).

  (i)   STRUCTURE   fields per line, headers, NaN/inf, timestamps, MACs, list columns
  (ii)  LABELS      label1..4 consistency, categories, empty windows labelled attack,
                    duplicates, vectors with >1 label
  (iii) TIME        capture dates of benign vs attack; benign windows inside attack spans
  (iv)  ATTRIBUTION per-device rows; do attack windows show an emitter absent from benign?
  (v)   PREVALENCE  windows per class
  (vi)  SHORTCUT    XGBoost (network, sensor, both) on random and time-ordered splits
  (vii) SYNC        sensor logs present per class/device; 1 s vs 10 s consistency

  python scripts/inspect_cic_iiot2025.py ~/datasets/cic_iiot2025
Out: results/cic_iiot2025_audit.txt
"""
import ast
import csv
import io
import os
import re
import sys
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

csv.field_size_limit(10**9)
IDCOLS = ["device_name", "device_mac", "label_full", "label1", "label2", "label3", "label4",
          "timestamp", "timestamp_start", "timestamp_end"]


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = next(r)
        fc = Counter(len(row) for row in r)
    df = pd.read_csv(path, low_memory=False)
    return df, head, fc


def split_cols(df):
    lists = [c for c in df.columns if c not in IDCOLS and df[c].dtype == object
             or (c not in IDCOLS and not pd.api.types.is_numeric_dtype(df[c]))]
    num = [c for c in df.columns if c not in IDCOLS and c not in lists]
    return lists, num


def main(d):
    data = {}
    section("(i) STRUCTURE")
    for w in ("1sec", "10sec"):
        for kind in ("benign", "attack"):
            df, head, fc = load(os.path.join(d, f"{kind}_samples_{w}.csv"))
            df["_kind"] = kind
            data[(kind, w)] = df
            print(f"{kind}_{w}: rows={len(df):,} header fields={len(head)} fields per line={dict(fc)}")
    b1, a1 = data[("benign", "1sec")], data[("attack", "1sec")]
    print("same header benign/attack:", list(b1.columns) == list(a1.columns))
    df = pd.concat([b1, a1], ignore_index=True)
    n = len(df)
    lists, num = split_cols(df)
    print("list/text columns:", lists)
    X = df[num].apply(pd.to_numeric, errors="coerce")
    print(f"numeric columns={len(num)}; NaN={int(X.isna().sum().sum())}; inf={int(np.isinf(X.values).sum())}")
    for c in ("timestamp_start", "timestamp_end"):
        t = pd.to_datetime(df[c], errors="coerce", utc=True)
        print(f"  {c} parsed {pct(int(t.notna().sum()), n)}")
    df["_t0"] = pd.to_datetime(df["timestamp_start"], errors="coerce", utc=True)
    df["_t1"] = pd.to_datetime(df["timestamp_end"], errors="coerce", utc=True)
    dur = (df["_t1"] - df["_t0"]).dt.total_seconds()
    print("  window length (s), 1 s files:", dur.round(3).value_counts().head(5).to_dict())
    mac = re.compile(r"^([0-9a-f]{2}:){5}[0-9a-f]{2}$", re.I)
    print(f"  device_mac well-formed {pct(int(df['device_mac'].astype(str).str.match(mac).sum()), n)}; "
          f"devices={df['device_name'].nunique()}")

    section("(ii) LABELS (1 s windows)")
    print("label1:", df["label1"].value_counts().to_dict())
    print("label2 (category):", df["label2"].value_counts().to_dict())
    print(f"label4 (attack): {df['label4'].nunique()} distinct")
    bad = df[(df["label1"] == "benign") != (df["label2"] == "benign")]
    print("label1/label2 inconsistent rows:", len(bad))
    net = [c for c in num if c.startswith("network_")]
    log = [c for c in num if c.startswith("log_")]
    empty = (X[net].abs().sum(axis=1) == 0) & (X[log].abs().sum(axis=1) == 0)
    emp_net = X[net].abs().sum(axis=1) == 0
    for k in ("benign", "attack"):
        m = df["label1"] == k
        print(f"  {k}: windows with no network and no sensor data {pct(int((empty & m).sum()), int(m.sum()))}; "
              f"with no network data {pct(int((emp_net & m).sum()), int(m.sum()))}")
    print("  empty windows by attack category:",
          df.loc[empty & (df['label1'] == 'attack'), 'label2'].value_counts().to_dict())
    h = pd.util.hash_pandas_object(X, index=False)
    print(f"feature duplicates: {int(h.duplicated().sum()):,} ({pct(int(h.duplicated().sum()), n)})")
    for key in ("label1", "label2"):
        g = df.groupby(h.values)[key].nunique()
        multi = g[g > 1]
        rows = h.isin(multi.index).values
        print(f"vectors with >1 {key}: {len(multi):,} (rows {int(rows.sum()):,}); "
              f"top: {df.loc[rows, key].value_counts().head(6).to_dict()}")

    section("(iii) TIME")
    for k, g in df.groupby("label2"):
        print(f"  {k:<10} {g['_t0'].min()} .. {g['_t0'].max()}  dates={sorted(g['_t0'].dt.date.astype(str).unique())[:6]}")
    ben = df[df["label1"] == "benign"]
    for k, g in df[df["label1"] == "attack"].groupby("label2"):
        inside = int(((ben["_t0"] >= g["_t0"].min()) & (ben["_t0"] <= g["_t0"].max())).sum())
        print(f"  {k:<10} benign windows inside its span: {inside:,}")

    section("(iv) ATTRIBUTION")
    print("rows per device (benign | attack):")
    print(pd.crosstab(df["device_name"], df["label1"]).to_string())

    def ips(s):
        try:
            return set(ast.literal_eval(s)) if isinstance(s, str) and s.startswith("[") else set()
        except Exception:
            return set()
    bsrc = set().union(*ben["network_ips_src"].map(ips))
    att = df[df["label1"] == "attack"]
    srcs = att["network_ips_src"].map(ips)
    nonempty = srcs.map(len) > 0
    new = srcs.map(lambda s: len(s - bsrc) > 0)
    print(f"benign source IPs: {len(bsrc)}; attack windows with any source IP: "
          f"{pct(int(nonempty.sum()), len(att))}; of those, with a source never seen in benign: "
          f"{pct(int((new & nonempty).sum()), int(nonempty.sum()))}")
    print("  by category:", {k: pct(int((new & nonempty)[att['label2'] == k].sum()),
                                     int(nonempty[att['label2'] == k].sum()))
                             for k in att["label2"].unique()})

    section("(v) PREVALENCE (1 s windows)")
    for k, v in df["label2"].value_counts().items():
        print(f"  {k:<10}{v:>9,}  {pct(v, n)}")
    print(f"  benign span: {ben['_t0'].min()} .. {ben['_t0'].max()}")

    section("(vi) SHORTCUT PROBE — XGBoost macro-F1 over label2 (1 s windows, dedup)")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    keep = ~h.duplicated().values
    d2 = df.loc[keep]
    Xd = X.loc[keep].replace([np.inf, -np.inf], np.nan).fillna(-1)
    Xd = Xd.assign(_dev=d2["device_name"].astype("category").cat.codes.values)
    y, names = pd.factorize(d2["label2"], sort=True)
    sets = {"network + sensor + device id": net + log + ["_dev"],
            "network + sensor": net + log,
            "network only": net,
            "sensor only": log}
    a, b = train_test_split(np.arange(len(Xd)), test_size=0.3, stratify=y, random_state=42)
    order = d2["_t0"].values
    ta, tb = [], []
    for c in range(len(names)):
        ii = np.where(y == c)[0]
        ii = ii[np.argsort(order[ii], kind="stable")]
        cut = int(0.7 * len(ii))
        ta += list(ii[:cut]); tb += list(ii[cut:])
    for split, (tr, te) in (("random 70/30", (a, b)),
                            ("time-ordered 70/30 per category", (np.array(ta), np.array(tb)))):
        print(f"-- {split}")
        for sname, cols in sets.items():
            m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist", n_jobs=4,
                                  random_state=42)
            m.fit(Xd.iloc[tr][cols], y[tr])
            p = m.predict(Xd.iloc[te][cols])
            per = f1_score(y[te], p, average=None, labels=range(len(names)), zero_division=0)
            print(f"  {sname:<30} macro-F1={per.mean():.3f}  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, per)))

    section("(vii) SYNC — sensor data and window consistency")
    has_log = X[log].abs().sum(axis=1) > 0
    print("windows with sensor data, by device (benign | attack):")
    print(pd.crosstab(df["device_name"], df["label1"], values=has_log, aggfunc="mean")
          .round(3).to_string())
    for kind in ("benign", "attack"):
        r1, r10 = len(data[(kind, "1sec")]), len(data[(kind, "10sec")])
        print(f"  {kind}: 1 s windows={r1:,}; 10 s windows={r10:,}; ratio={r1 / max(r10, 1):.2f} "
              f"(10 expected if 10 s windows tile the same time)")
    for kind in ("benign", "attack"):
        a1_, a10 = data[(kind, "1sec")], data[(kind, "10sec")]
        s1 = a1_.groupby("label_full").size()
        s10 = a10.groupby("label_full").size()
        j = pd.concat([s1.rename("w1"), s10.rename("w10")], axis=1).fillna(0)
        j["ratio"] = j["w1"] / j["w10"].replace(0, np.nan)
        print(f"  {kind}: scenarios={len(j)}; ratio 1s/10s per scenario: "
              f"median={j['ratio'].median():.2f} min={j['ratio'].min():.2f} max={j['ratio'].max():.2f}; "
              f"scenarios only in one file: {int(((j['w1'] == 0) | (j['w10'] == 0)).sum())}")


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/cic_iiot2025")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    os.makedirs("results", exist_ok=True)
    open("results/cic_iiot2025_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
