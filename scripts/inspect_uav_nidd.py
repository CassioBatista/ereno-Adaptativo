#!/usr/bin/env python3
"""Inspect UAV-NIDD (Hadi et al., IEEE TNSE 12(4), 2025) before any profile is defined.

Three labelled CSVs, one per compromise scenario (UAV, access point, ground control
station), plus Sample.csv. For each:
  1. SCHEMA      — rows, columns, label column, types, constants
  2. LABELS      — per-class counts
  3. ALIGNMENT   — does each column hold the kind of value its name says? Values are
                   classified (epoch, Wi-Fi frequency, dBm, IPv4, ...) and checked
                   against the header
  4. TIME        — epoch/relative columns, per-class spans, class blocks in file order
  5. ADDRESSES   — ip.src/ip.dst per class (attribution)
  6. DUPLICATES  — feature-level duplicates and vectors carrying >1 label
  7. SHORTCUTS   — XGBoost with and without position/time/identity columns

  python scripts/inspect_uav_nidd.py ~/datasets/uav_nidd
Out: results/uav_nidd_inspection.txt
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

FILES = {
    "case1_uav": "UAV-Case1-Label/UAV-Case1-Label.csv",
    "case2_ap": "Access Point Case2 Label/Access Point Case2 Label.csv",
    "case3_gcs": "GCS-Case 3/GCS-Case 3/GSC Case3 Label .csv",
    "sample": "Sample.csv",
}
# Columns that locate a packet in the capture or name an endpoint, not behaviour.
POSITION = ["frame.number", "frame.time_epoch", "frame.time_relative",
            "frame.time_delta_displayed", "radiotap.mactime", "radiotap.timestamp.ts",
            "wlan_radio.start_tsf", "wlan_radio.end_tsf", "wlan_radio.timestamp",
            "wlan.seq", "eapol.keydes.replay_counter", "http.date", "http.last_modified"]
IDENTITY = ["ip.src", "ip.dst", "wlan.bssid", "udp.srcport", "udp.dstport", "http.host",
            "http.location"]


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def kind(s):
    """Dominant kind of the non-zero values of a column."""
    v = s[s.astype(str).str.strip().ne("0") & s.notna()]
    if v.empty:
        return "all-zero"
    x = pd.to_numeric(v, errors="coerce")
    if x.isna().mean() > 0.5:
        sv = v.astype(str)
        if sv.str.match(r"^\d+\.\d+\.\d+\.\d+$").mean() > 0.5:
            return "IPv4"
        if sv.str.match(r"^0x[0-9a-fA-F]+$").mean() > 0.5:
            return "hex"
        return "text"
    x = x.dropna()
    med = x.median()
    if 1.2e9 < med < 2.0e9:
        return "epoch-s"
    if x.between(2400, 2500).mean() > 0.8 or x.between(5150, 5900).mean() > 0.8:
        return "wifi-freq-MHz"
    if x.between(-100, -10).mean() > 0.8:
        return "dBm"
    return f"num(med={med:.4g})"


EXPECT = {"frame.time_epoch": "epoch-s", "radiotap.channel.freq": "wifi-freq-MHz",
          "wlan_radio.frequency": "wifi-freq-MHz", "radiotap.dbm_antsignal": "dBm",
          "wlan_radio.signal_dbm": "dBm", "ip.src": "IPv4", "ip.dst": "IPv4"}


def field_counts(path):
    """Fields per line, parsed as CSV: rows with a count other than the header's
    cannot be aligned with it."""
    import csv
    from collections import Counter
    with open(path, newline="", encoding="latin-1") as f:
        r = csv.reader(f)
        head = len(next(r))
        return head, Counter(len(row) for row in r)


def inspect(name, path):
    head, fc = field_counts(path)
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    lab = df.columns[-1]
    n = len(df)
    section(f"{name}: {os.path.relpath(path)} — {n:,} rows x {df.shape[1]} columns")
    print(f"fields per line: header={head}, rows={dict(fc.most_common(6))}")
    print(f"label column (last): '{lab}'")
    nonnum = [c for c in df.columns[:-1] if not pd.api.types.is_numeric_dtype(df[c])]
    print("non-numeric columns:", nonnum)
    print("constant columns:", [c for c in df.columns if df[c].nunique(dropna=False) <= 1])
    print("NaN cells:", int(df.isna().sum().sum()))

    print("-- labels")
    vc = df[lab].value_counts()
    for k, v in vc.items():
        print(f"  {str(k):<28}{v:>10,}  {pct(v, n)}")
    runs = int(df[lab].ne(df[lab].shift()).sum())
    print(f"  label runs in file order: {runs:,} (= number of classes -> one block per class)")

    print("-- alignment (dominant value kind per column; '!!' = contradicts the name)")
    sample = df.sample(min(n, 50_000), random_state=0)
    for c in df.columns[:-1]:
        k = kind(sample[c])
        exp = EXPECT.get(c)
        flag = "  !!" if exp and k != exp and k != "all-zero" else ""
        hint = "" if not (k in ("epoch-s", "wifi-freq-MHz", "dBm", "IPv4") and exp is None) \
            else "  (named otherwise)"
        if flag or hint or c in EXPECT:
            print(f"  {c:<34}{k:<22}{('expected ' + exp) if exp else ''}{flag}{hint}")

    print("-- time")
    for c in ("frame.time_epoch", "frame.time_relative", "flow_duration"):
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            print(f"  {c}: min={x.min():.6g} max={x.max():.6g} zero={pct(int((x == 0).sum()), n)}")
    tc = next((c for c in ("frame.time_relative", "frame.time_epoch") if c in df.columns), None)
    if tc:
        x = pd.to_numeric(df[tc], errors="coerce")
        for k in vc.index:
            s = x[df[lab] == k]
            s = s[s > 1e9]
            if len(s):
                print(f"  {str(k):<28} {tc} as epoch: {pd.to_datetime(s.min(), unit='s')} .. "
                      f"{pd.to_datetime(s.max(), unit='s')}  ({len(s):,} rows)")

    if "ip.src" in df.columns:
        print("-- addresses")
        for k in vc.index:
            s = df[df[lab] == k]
            src = s["ip.src"].astype(str)
            has = src.str.match(r"^\d+\.\d+\.\d+\.\d+$")
            top = src[has].value_counts().head(3)
            print(f"  {str(k):<28} rows with IPv4 src {pct(int(has.sum()), len(s)):>8}  "
                  f"distinct={src[has].nunique():>4}  top: "
                  + ", ".join(f"{i} ({pct(v, int(has.sum()))})" for i, v in top.items()))

    print("-- duplicates")
    feats = [c for c in df.columns[:-1] if c not in POSITION]
    print(f"  exact duplicate rows: {int(df.duplicated().sum()):,}")
    h = pd.util.hash_pandas_object(df[feats], index=False)
    print(f"  duplicates without position columns: {int(h.duplicated().sum()):,} "
          f"({pct(int(h.duplicated().sum()), n)})")
    g = df.groupby(h.values)[lab].nunique()
    print(f"  vectors carrying >1 label: {int((g > 1).sum()):,}")

    print("-- shortcut hint: XGBoost macro-F1 (stratified 70/30, rows deduplicated)")
    try:
        import xgboost as xgb
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        d = df.loc[~h.duplicated().values & df[lab].notna().values]
        d = d[d[lab].map(d[lab].value_counts()) >= 5]        # stratify needs >= 2 per class
        X = d[df.columns[:-1]].copy()
        for c in nonnum:
            X[c] = X[c].astype(str).astype("category").cat.codes
        codes, names = pd.factorize(d[lab], sort=True)
        y = pd.Series(codes, index=d.index)
        sets = {"all columns": list(X.columns),
                "without position/time": [c for c in X.columns if c not in POSITION],
                "without position/time/identity": [c for c in X.columns
                                                   if c not in POSITION + IDENTITY]}
        tr, te = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y,
                                  random_state=42)
        for sname, cols in sets.items():
            m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist",
                                  n_jobs=4, random_state=42)
            m.fit(X.iloc[tr][cols], y.iloc[tr])
            p = m.predict(X.iloc[te][cols])
            per = f1_score(y.iloc[te], p, average=None)
            worst = sorted(zip(per, names[np.unique(np.r_[y.iloc[te].values, p])]))[:3]
            print(f"  {sname:<34} macro-F1={f1_score(y.iloc[te], p, average='macro'):.3f}  "
                  f"worst: " + ", ".join(f"{c} {v:.2f}" for v, c in worst))
            if sname == "all columns":
                gain = pd.Series(m.get_booster().get_score(importance_type="gain"))
                print("    top gain:", list(gain.sort_values(ascending=False).head(6).index))
    except Exception as e:
        print("  skipped:", repr(e))
    return df


def main(d):
    dfs = {k: inspect(k, os.path.join(d, f)) for k, f in FILES.items()}
    section("Sample.csv vs case 3 (same 45 flow columns?)")
    a, b = dfs["sample"], dfs["case3_gcs"]
    print("same header:", list(a.columns) == list(b.columns))
    hb = set(pd.util.hash_pandas_object(b, index=False))
    ha = pd.util.hash_pandas_object(a, index=False)
    print(f"Sample rows also in case 3: {pct(int(ha.isin(hb).sum()), len(a))}")


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/uav_nidd")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    os.makedirs("results", exist_ok=True)
    open("results/uav_nidd_inspection.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
