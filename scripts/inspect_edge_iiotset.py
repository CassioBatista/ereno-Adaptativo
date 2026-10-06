#!/usr/bin/env python3
"""Inspect Edge-IIoTset (Ferrag et al., IEEE Access 2022) before any profile is defined.

Answers, for the selected DNN file (the full 2.2 M-row release):
  1. SCHEMA      — columns, types, constant columns, identifier/payload columns
  2. LABELS      — Attack_type counts, Attack_label consistency, 5-category grouping
  3. DUPLICATES  — exact duplicates, and duplicates once identifiers are dropped,
                   including feature-identical rows that carry different labels
  4. TIME        — what frame.time actually holds, per-class spans, interleaving
  5. ADDRESSES   — source/destination hosts per class; does the source alone give
                   the class away (shortcut); how normal traffic splits per device
  6. SHORTCUTS   — a quick XGBoost on the remaining features (a hint, not a result)
  7. ML FILE     — what the 157 k-row "ML" file is relative to the DNN file

  python scripts/inspect_edge_iiotset.py ~/datasets/edge_iiotset
Out: results/edge_iiotset_inspection.txt
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

# The authors' own drop list (Kaggle readme, step 4): identifiers, ports, payloads.
AUTHOR_DROP = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
               "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
               "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
               "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"]
LABELS = ["Attack_label", "Attack_type"]
# Values that identify a packet or a connection rather than describe behaviour.
ARBITRARY = ["tcp.seq", "tcp.ack", "tcp.ack_raw", "tcp.checksum", "icmp.checksum",
             "icmp.seq_le", "udp.stream", "mbtcp.trans_id"]

# Five threat categories of the paper (Sec. on attacks); verify against the PDF.
CATEGORY = {
    "DDoS_UDP": "DoS/DDoS", "DDoS_ICMP": "DoS/DDoS", "DDoS_TCP": "DoS/DDoS",
    "DDoS_HTTP": "DoS/DDoS",
    "Port_Scanning": "Information gathering", "Fingerprinting": "Information gathering",
    "Vulnerability_scanner": "Information gathering",
    "MITM": "Man in the middle",
    "XSS": "Injection", "SQL_injection": "Injection", "Uploading": "Injection",
    "Backdoor": "Malware", "Password": "Malware", "Ransomware": "Malware",
    "Normal": "Normal",
}


def section(title):
    print(f"\n=== {title} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main(d):
    dnn = pd.read_csv(os.path.join(d, "DNN-EdgeIIoT-dataset.csv"), low_memory=False)
    n = len(dnn)

    # ---------------------------------------------------------------- 1. schema
    section(f"1. SCHEMA — DNN file: {n:,} rows x {dnn.shape[1]} columns")
    # pandas >= 3 reads text as the 'str' dtype, not object: test for numeric instead.
    obj = [c for c in dnn.columns
           if not pd.api.types.is_numeric_dtype(dnn[c]) and c not in LABELS]
    const = [c for c in dnn.columns if dnn[c].nunique(dropna=False) <= 1]
    print("non-numeric columns:", obj)
    print("constant columns:", const)
    print("author drop list present:", [c for c in AUTHOR_DROP if c in dnn.columns])
    print("NaN cells:", int(dnn.isna().sum().sum()))
    # Column alignment: frame.time must hold a time; where it does not, every column
    # of the row is shifted and its features describe the wrong fields.
    shifted = ~dnn["frame.time"].astype(str).str.strip().str.match(r"^\d{4} \d")
    print(f"rows with a shifted layout (frame.time is not a time): {int(shifted.sum()):,}")
    for k, v in dnn.loc[shifted, "Attack_type"].value_counts().items():
        tot = int((dnn["Attack_type"] == k).sum())
        ex = dnn.loc[shifted & (dnn["Attack_type"] == k), ["frame.time", "ip.src_host",
                                                         "ip.dst_host"]].iloc[0].tolist()
        print(f"  {k:<22}{v:>10,} of {tot:,}  e.g. frame.time/src/dst = {ex}")

    # ---------------------------------------------------------------- 2. labels
    section("2. LABELS")
    vc = dnn["Attack_type"].value_counts()
    print(f"{'Attack_type':<24}{'rows':>12}{'share':>9}  category")
    for k, v in vc.items():
        print(f"  {k:<22}{v:>12,}{pct(v, n):>9}  {CATEGORY.get(k, '?? unmapped')}")
    bad = dnn[(dnn["Attack_type"] == "Normal") != (dnn["Attack_label"] == 0)]
    print("Attack_label inconsistent with Attack_type:", len(bad))
    cat = dnn["Attack_type"].map(CATEGORY)
    print("per category:", cat.value_counts().to_dict())
    nn = int((dnn["Attack_type"] == "Normal").sum())
    print(f"prevalence: normal {pct(nn, n)}, attack {pct(n - nn, n)} "
          f"(normal:attack = {nn / max(n - nn, 1):.2f}:1)")

    # ------------------------------------------------------------ 3. duplicates
    section("3. DUPLICATES")
    print("exact duplicate rows:", f"{int(dnn.duplicated().sum()):,}")
    feats = [c for c in dnn.columns if c not in AUTHOR_DROP + LABELS]
    fdup = dnn.duplicated(subset=feats + ["Attack_type"])
    print(f"duplicates on features+label (identifiers dropped): {int(fdup.sum()):,} "
          f"({pct(int(fdup.sum()), n)})")
    h = pd.util.hash_pandas_object(dnn[feats], index=False)
    g = dnn.groupby(h.values)["Attack_type"].nunique()
    conflict = g[g > 1]
    print(f"feature vectors carrying >1 label: {len(conflict):,} distinct vectors")
    if len(conflict):
        key = h.isin(conflict.index).values
        print(f"  rows involved: {int(key.sum()):,}; labels among them:",
              dnn.loc[key, "Attack_type"].value_counts().to_dict())
    print("unique rows per class after dedup (features+label):")
    ded = dnn.loc[~fdup]
    for k, v in ded["Attack_type"].value_counts().items():
        print(f"  {k:<22}{v:>12,}  (kept {pct(v, vc[k])})")

    # ------------------------------------------------------------------ 4. time
    section("4. TIME (frame.time)")
    ft = dnn["frame.time"].astype(str)
    print("examples:", ft.head(3).tolist(), ft.sample(3, random_state=0).tolist())
    t = pd.to_datetime(ft.str.strip(), format="%Y %H:%M:%S.%f", errors="coerce")
    print(f"parsed as 'YYYY HH:MM:SS.f': {int(t.notna().sum()):,} / {n:,}  "
          "(the month/day is missing: the original 'Mon DD, YYYY' was split on its comma)")
    print("unparsed examples:", ft[t.isna()].value_counts().head(5).to_dict())
    tod = (t - t.dt.normalize()).dt.total_seconds()
    dnn["_tod"] = tod
    print(f"{'class':<22}{'first':>10}{'last':>10}{'span_s':>10}{'rows/s':>10}")
    for k in vc.index:
        s = dnn.loc[dnn["Attack_type"] == k, "_tod"].dropna()
        if len(s):
            span = s.max() - s.min()
            print(f"  {k:<20}{s.min():>10.0f}{s.max():>10.0f}{span:>10.0f}"
                  f"{len(s) / max(span, 1):>10.1f}")
    order = dnn["Attack_type"].ne(dnn["Attack_type"].shift()).sum()
    print(f"label runs in file order: {int(order):,} (1 run per class = blocks; "
          "many = interleaved)")
    print("time-of-day monotonic within file:", bool(dnn["_tod"].dropna().is_monotonic_increasing))

    # ------------------------------------------------------------- 5. addresses
    section("5. ADDRESSES")
    for c in ("ip.src_host", "ip.dst_host"):
        print(f"{c}: {dnn[c].nunique():,} distinct")
    print(f"{'class':<22}{'src':>6}{'dst':>6}  top sources")
    for k in vc.index:
        sub = dnn[dnn["Attack_type"] == k]
        top = sub["ip.src_host"].value_counts().head(3)
        tops = ", ".join(f"{i} ({pct(v, len(sub))})" for i, v in top.items())
        print(f"  {k:<20}{sub['ip.src_host'].nunique():>6}"
              f"{sub['ip.dst_host'].nunique():>6}  {tops}")
    # Shortcut: how well does the source address alone predict the class?
    maj = dnn.groupby("ip.src_host")["Attack_type"].agg(lambda s: s.value_counts().iloc[0])
    print(f"source address alone -> class (majority per address): accuracy "
          f"{pct(int(maj.sum()), n)}")
    normal = dnn[dnn["Attack_type"] == "Normal"]
    devs = normal["ip.src_host"].value_counts()
    print(f"normal traffic: {len(devs)} source addresses; top 12:")
    for i, v in devs.head(12).items():
        print(f"  {i:<18}{v:>10,}  {pct(v, len(normal))}")

    # ------------------------------------------------------------- 6. shortcuts
    section("6. SHORTCUT HINT — XGBoost on author-cleaned features (dedup, 70/30)")
    try:
        import xgboost as xgb
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        X = ded[feats].copy()
        nonnum = [c for c in feats if not pd.api.types.is_numeric_dtype(X[c])]
        for c in nonnum:                       # categorical protocol fields -> codes
            X[c] = X[c].astype(str).astype("category").cat.codes
        y = ded["Attack_type"].astype("category")
        Xtr, Xte, ytr, yte = train_test_split(X, y.cat.codes, test_size=0.3,
                                              stratify=y.cat.codes, random_state=42)
        m = xgb.XGBClassifier(n_estimators=60, max_depth=6, tree_method="hist",
                              n_jobs=4, random_state=42)
        m.fit(Xtr, ytr)
        print("categorical columns encoded as codes:", nonnum)
        print(classification_report(yte, m.predict(Xte), target_names=list(y.cat.categories),
                                    digits=4, zero_division=0))
        gain = pd.Series(m.get_booster().get_score(importance_type="gain"))
        print("top features by gain:", gain.sort_values(ascending=False).head(12).round(1).to_dict())
        # Second pass without per-packet arbitrary values (sequence/ack numbers,
        # checksums, stream ids): a model that leans on them memorises captures.
        arbitrary = [c for c in ARBITRARY if c in X.columns]
        m2 = xgb.XGBClassifier(n_estimators=60, max_depth=6, tree_method="hist",
                               n_jobs=4, random_state=42)
        m2.fit(Xtr.drop(columns=arbitrary), ytr)
        print(f"\nwithout arbitrary per-packet fields {arbitrary}:")
        print(classification_report(yte, m2.predict(Xte.drop(columns=arbitrary)),
                                    target_names=list(y.cat.categories), digits=4,
                                    zero_division=0))
    except Exception as e:                                   # never fail the inspection
        print("skipped:", repr(e))

    # --------------------------------------------------------------- 7. ML file
    section("7. ML FILE")
    ml = pd.read_csv(os.path.join(d, "ML-EdgeIIoT-dataset.csv"), low_memory=False)
    print(f"{len(ml):,} rows x {ml.shape[1]} columns; same header as DNN:",
          list(ml.columns) == [c for c in dnn.columns if c != "_tod"])
    print("frame.time examples:", ml["frame.time"].astype(str).head(3).tolist())
    print("Attack_type:", ml["Attack_type"].value_counts().to_dict())
    print("exact duplicates:", int(ml.duplicated().sum()))


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/edge_iiotset")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    os.makedirs("results", exist_ok=True)
    open("results/edge_iiotset_inspection.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
