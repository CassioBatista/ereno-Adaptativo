#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on X-IIoTID
(Al-Hawawreh, Sitnikova and Aboutorab, IEEE IoT Journal 2022). Copy: the first author's
Kaggle upload munaalhawawreh/xiiotid-iiot-intrusion-dataset (checksums in
results/xiiotid_md5sums.txt). 820,834 records, 68 columns: Date/Timestamp, addresses and
ports, Zeek-like flow fields, host-resource statistics (CPU, I/O, memory, processes),
OSSEC and anomaly alerts, log-derived fields, and labels class1 (19), class2 (10), class3.

Feature groups checked separately, because host statistics drift with time (a possible
capture clock) and alert columns may come from a detector that already saw the attack:
  FLOW  flow fields;  HOST  Avg_/Std_ host statistics;  ALERT  alert and log fields.

  (i)   STRUCTURE   fields per line, Date vs Timestamp, value kinds
  (ii)  LABELS      class1 -> class2 mapping, duplicates, vectors with >1 class
  (iii) TIME        span and dates per class, interleaving with normal traffic
  (iv)  ATTRIBUTION source addresses per class
  (v)   PREVALENCE  class shares
  (vi)  SHORTCUT    XGBoost per feature group, random and time-ordered splits

  python scripts/inspect_xiiotid.py ~/datasets/xiiotid/xiiotid.csv
Out: results/xiiotid_audit.txt
"""
import io
import os
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

IDENT = ["Date", "Timestamp", "Scr_IP", "Scr_port", "Des_IP", "Des_port"]
LABELS = ["class1", "class2", "class3"]


def groups(cols):
    host = [c for c in cols if c.lower().startswith(("avg_", "std_"))]
    alert = ["anomaly_alert", "OSSEC_alert", "OSSEC_alert_level", "Login_attempt",
             "Succesful_login", "File_activity", "Process_activity",
             "read_write_physical.process", "is_privileged"]
    flow = [c for c in cols if c not in IDENT + LABELS + host + alert]
    return flow, host, alert


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main(path):
    print("=== (i) STRUCTURE ===")
    nf = subprocess.run(["awk", "-F,", "NR>1{c[NF]++} END{for(k in c) print k\":\"c[k]}", path],
                        capture_output=True, text=True).stdout.split()
    print("fields per line:", nf)
    df = pd.read_csv(path, low_memory=False)
    n = len(df)
    flow, host, alert = groups(df.columns)
    print(f"rows={n:,}; flow={len(flow)}, host={len(host)}, alert/log={len(alert)}")
    raw = df["Timestamp"].astype(str)
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    bad = df["Timestamp"].isna()
    print(f"  Timestamp non-numeric: {int(bad.sum()):,} rows; examples {raw[bad].unique()[:5].tolist()}; "
          f"their classes {df.loc[bad, 'class2'].value_counts().to_dict()}")
    df["Timestamp"] = df["Timestamp"].fillna(-1)
    d1 = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    ts = pd.to_datetime(df["Timestamp"].where(~bad), unit="s", errors="coerce")
    print(f"  Date parsed {pct(int(d1.notna().sum()), n)}; Timestamp parsed {pct(int(ts.notna().sum()), n)}; "
          f"Date == Timestamp's day (UTC) {pct(int((d1.dt.date == ts.dt.date).sum()), n)}, "
          f"(+10 h, Australia) {pct(int((d1.dt.date == (ts + pd.Timedelta(hours=10)).dt.date).sum()), n)}")
    print(f"  Timestamp range {ts.min()} .. {ts.max()}; rows in time order: "
          f"{pct(int((np.diff(df['Timestamp'].values) >= 0).sum()), n - 1)}")
    for c in ("Scr_port", "Des_port"):
        v = pd.to_numeric(df[c], errors="coerce")
        print(f"  {c}: numeric {pct(int(v.notna().sum()), n)}, within 0..65535 {pct(int(v.between(0, 65535).sum()), n)}")
    obj = [c for c in flow + host + alert if df[c].dtype == object]
    print("  non-numeric feature columns:", {c: df[c].nunique() for c in obj})

    print("\n=== (ii) LABELS ===")
    m = df.groupby("class1")["class2"].nunique()
    print("class1 with more than one class2:", m[m > 1].to_dict())
    print("class1 -> class2:", df.groupby("class1")["class2"].first().to_dict())
    X = df[flow + host + alert].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.factorize(X[c])[0]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(-1)
    for gname, cols in (("flow", flow), ("flow+host+alert", flow + host + alert)):
        h = pd.util.hash_pandas_object(X[cols], index=False)
        pairs = pd.DataFrame({"h": h.values, "y": df["class2"].values}).drop_duplicates()
        multi = pairs["h"][pairs["h"].duplicated()].unique()
        rows = np.isin(h.values, multi)
        print(f"  [{gname}] duplicates {pct(int(h.duplicated().sum()), n)}; rows in vectors with >1 class2 "
              f"{pct(int(rows.sum()), n)}; classes: {df.loc[rows, 'class2'].value_counts().head(6).to_dict()}")
    print(f"exact duplicate rows: {int(df.duplicated().sum()):,}")

    print("\n=== (iii) TIME ===")
    df["_ts"] = df["Timestamp"]
    sp = df.groupby("class2")["_ts"].agg(["min", "max", "count"])
    sp["from"] = pd.to_datetime(sp["min"], unit="s"); sp["to"] = pd.to_datetime(sp["max"], unit="s")
    sp["days"] = df.groupby("class2")["Date"].nunique()
    print(sp[["from", "to", "count", "days"]].to_string())
    nt = np.sort(df.loc[df["class2"] == "Normal", "_ts"].values)
    for k, g in df[df["class2"] != "Normal"].groupby("class2"):
        lo, hi = g["_ts"].min(), g["_ts"].max()
        ins = int(np.searchsorted(nt, hi, "right") - np.searchsorted(nt, lo, "left"))
        at = np.unique(g["_ts"].values)
        i = np.searchsorted(at, nt)
        near = np.zeros(len(nt), bool)
        for j in (i - 1, np.minimum(i, len(at) - 1)):
            ok = j >= 0
            near[ok] |= np.abs(nt[ok] - at[j[ok]]) <= 60
        print(f"  {k:<18} normal inside its span {ins:,} (share {pct(ins, ins + len(g))}); "
              f"normal within 60 s of one of its records {int(near.sum()):,}")
    print("normal records per date:", df[df["class2"] == "Normal"]["Date"].value_counts().sort_index().to_dict())
    print("attack records per date:", df[df["class2"] != "Normal"]["Date"].value_counts().sort_index().to_dict())

    print("\n=== (iv) ATTRIBUTION ===")
    for k, g in df.groupby("class2"):
        top = g["Scr_IP"].value_counts().head(3)
        print(f"  {k:<18} src={g['Scr_IP'].nunique():>6,} dst={g['Des_IP'].nunique():>6,}  top src: "
              + ", ".join(f"{i} ({pct(v, len(g))})" for i, v in top.items()))
    mj = pd.crosstab(df["Scr_IP"], df["class2"]).max(axis=1).sum()
    print(f"source address alone -> class2: {pct(int(mj), n)}")

    print("\n=== (v) PREVALENCE ===")
    for k, v in df["class2"].value_counts().items():
        print(f"  {k:<18}{v:>9,}  {pct(v, n)}")

    print("\n=== (vi) SHORTCUT PROBE — XGBoost macro-F1 over class2 (dedup on all features) ===")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    Xi = X.copy()
    Xi["Timestamp"] = df["Timestamp"]
    for c in ("Scr_IP", "Des_IP"):
        Xi[c] = pd.factorize(df[c])[0]
    for c in ("Scr_port", "Des_port"):
        Xi[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1)
    ded = ~pd.util.hash_pandas_object(Xi[flow + host + alert], index=False).duplicated().values
    idx = np.where(ded)[0]
    y, names = pd.factorize(df["class2"].values, sort=True)
    sets = {"flow": flow, "host only": host, "alert/log only": alert, "flow + host": flow + host,
            "flow + host + alert": flow + host + alert,
            "all + identifiers": flow + host + alert + ["Timestamp", "Scr_IP", "Des_IP", "Scr_port", "Des_port"]}
    a, b = train_test_split(idx, test_size=0.3, stratify=y[idx], random_state=42)
    ta, tb = [], []
    for c in range(len(names)):
        ii = idx[y[idx] == c]
        ii = ii[np.argsort(df["Timestamp"].values[ii], kind="stable")]
        cut = int(0.7 * len(ii)); ta += list(ii[:cut]); tb += list(ii[cut:])
    for split, (tr, te) in (("random 70/30", (a, b)),
                            ("time-ordered 70/30 per class", (np.array(ta), np.array(tb)))):
        print(f"-- {split}")
        for sname, cols in sets.items():
            prm = {"objective": "multi:softmax", "num_class": len(names), "max_depth": 6,
                   "eta": 0.3, "tree_method": "hist", "nthread": 4, "seed": 42}
            mdl = xgb.train(prm, xgb.DMatrix(Xi.iloc[tr][cols], label=y[tr]), num_boost_round=100)
            p = mdl.predict(xgb.DMatrix(Xi.iloc[te][cols])).astype(int)
            per = f1_score(y[te], p, average=None, labels=range(len(names)), zero_division=0)
            print(f"  {sname:<22} macro-F1={per.mean():.3f}  "
                  + "  ".join(f"{k[:8]}={v:.2f}" for k, v in zip(names, per)))
            if split.startswith("random") and sname == "flow + host + alert":
                g = pd.Series(mdl.get_score(importance_type="gain"))
                print("    top gain:", list(g.sort_values(ascending=False).head(8).index))


if __name__ == "__main__":
    p = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/xiiotid/xiiotid.csv")
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(p)
    open("results/xiiotid_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
