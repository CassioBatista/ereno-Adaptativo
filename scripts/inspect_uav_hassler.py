#!/usr/bin/env python3
"""Inspect the UAV cyber-physical dataset of Hassler, Mughal & Ismail (IEEE T-ITS 2024;
GitHub uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks, MIT), Dataset_T-ITS.csv.

The CSV stacks ten blocks, each with its own header line (cyber or physical, per class).
For each block:
  1. SCHEMA      — which header it carries; blocks sharing a feature space
  2. LABELS      — classes inside the block
  3. TIME        — timestamp kind and range
  4. SYNC        — can the cyber and the physical block of a class be aligned in time?
  5. DUPLICATES  — duplicate rows
Then:
  6. BENIGN COVERAGE — which attack classes have benign rows in their own feature space
  7. SHORTCUT HINT   — XGBoost on the one cyber schema that holds benign + attacks

  python scripts/inspect_uav_hassler.py ~/datasets/uav_hassler/Dataset_T-ITS.csv
Out: results/uav_hassler_inspection.txt
"""
import csv
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

NUM = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def section(t):
    print(f"\n=== {t} ===")


def blocks(path):
    rows = list(csv.reader(open(path, encoding="latin-1")))
    out, cur = [], None
    for i, r in enumerate(rows):
        vals = [x for x in r if x != ""]
        if sum(1 for x in vals if not NUM.match(x)) >= 5:          # a header line
            cur = {"line": i + 1, "header": vals, "rows": []}
            out.append(cur)
        elif vals:
            cur["rows"].append(vals[:len(cur["header"])])
    for b in out:
        b["df"] = pd.DataFrame(b["rows"], columns=b["header"])
        for c in b["df"].columns[:-1]:
            b["df"][c] = pd.to_numeric(b["df"][c], errors="coerce")
    return out


def tkind(s):
    s = s.dropna()
    if s.empty:
        return "none"
    return "epoch" if s.median() > 1.2e9 else "relative-s"


def main(path):
    bl = blocks(path)
    section(f"1-3. BLOCKS ({len(bl)})")
    schemas = {}
    for b in bl:
        df, h = b["df"], b["header"]
        sig = tuple(h)
        schemas.setdefault(sig, len(schemas) + 1)
        kind = "cyber" if "frame.len" in h else "physical"
        tcol = next((c for c in ("timestamp_c", "timestamp_p") if c in h), None)
        if tcol:
            t = df[tcol]
            tk = tkind(t)
            rng = (f"{pd.to_datetime(t.min(), unit='s')} .. {pd.to_datetime(t.max(), unit='s')}"
                   if tk == "epoch" else f"{t.min():.1f} .. {t.max():.1f} s")
        else:
            tk, rng = "none", "(no timestamp column)"
        b.update(kind=kind, tcol=tcol, tk=tk)
        print(f"line {b['line']:>6}  {kind:<8} schema S{schemas[sig]} ({len(h) - 1} features)  "
              f"rows={len(df):>6,}  labels={df['class'].value_counts().to_dict()}  "
              f"time[{tk}] {rng}")
        print(f"        duplicate rows: {int(df.duplicated().sum()):,}; "
              f"without timestamp: {int(df.drop(columns=[tcol] if tcol else []).duplicated().sum()):,}")
    print("\nschemas:")
    for sig, k in schemas.items():
        print(f"  S{k}: {list(sig)[:8]} ... ({len(sig) - 1} features)")

    section("4. SYNC — cyber vs physical time ranges per class")
    by = {}
    for b in bl:
        lab = b["df"]["class"].iloc[0]
        by.setdefault(lab.split()[0].lower(), {})[b["kind"]] = b
    for lab, d in by.items():
        c, p = d.get("cyber"), d.get("physical")
        if not (c and p) or not (c["tcol"] and p["tcol"]):
            print(f"  {lab:<10} cannot align: "
                  f"{'physical block has no timestamp' if p and not p['tcol'] else 'missing block'}")
            continue
        tc, tp = c["df"][c["tcol"]], p["df"][p["tcol"]]
        lo, hi = max(tc.min(), tp.min()), min(tc.max(), tp.max())
        print(f"  {lab:<10} cyber [{tc.min():.1f}, {tc.max():.1f}]  physical [{tp.min():.1f}, "
              f"{tp.max():.1f}]  overlap {max(0, hi - lo):.1f} s  "
              f"(start offset {tp.min() - tc.min():+.1f} s)")

    section("6. BENIGN COVERAGE — classes per feature space")
    for sig, k in schemas.items():
        labs = {}
        for b in bl:
            if tuple(b["header"]) == sig:
                for x, v in b["df"]["class"].value_counts().items():
                    labs[x] = labs.get(x, 0) + v
        print(f"  S{k}: {labs}")

    section("7. SHORTCUT HINT — XGBoost on cyber schema S1 (benign, DoS, replay)")
    try:
        import xgboost as xgb
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        s1 = pd.concat([b["df"] for b in bl if b["kind"] == "cyber"
                        and tuple(b["header"]) == tuple(bl[0]["header"])], ignore_index=True)
        y, names = pd.factorize(s1["class"], sort=True)
        ident = ["timestamp_c", "frame.number", "wlan.seq", "ip.id", "tcp.seq_raw",
                 "tcp.ack_raw", "time_since_last_packet"]
        for label, drop in (("all features", ["class"]),
                            ("without position/sequence ids", ["class"] + ident)):
            X = s1.drop(columns=drop).fillna(-1)
            tr, te, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            m = xgb.XGBClassifier(n_estimators=100, max_depth=6, n_jobs=4, random_state=42)
            m.fit(tr, ytr)
            print(f"-- {label}")
            print(classification_report(yte, m.predict(te), target_names=list(names), digits=4,
                                        zero_division=0))
            g = pd.Series(m.get_booster().get_score(importance_type="gain"))
            print("   top gain:", list(g.sort_values(ascending=False).head(6).index))
    except Exception as e:
        print("skipped:", repr(e))


if __name__ == "__main__":
    p = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                           else "~/datasets/uav_hassler/Dataset_T-ITS.csv")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(p)
    os.makedirs("results", exist_ok=True)
    open("results/uav_hassler_inspection.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
