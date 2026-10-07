#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on the Mississippi State
ICS datasets of Morris et al., as listed in the CPS survey of Quincozes et al.:
  Morris-2  gas_final.arff        gas pipeline (2014), 7 attack categories
  Morris-3  water_final.arff      water storage tank (2014), 7 attack categories
  Morris-4  IanArffDataset.arff   new gas pipeline (2015), 7 categories / 35 attacks
The UAH page that hosted them is gone; the copies come from the Web Archive (checksums in
results/morris_ics_md5sums.txt). The authors' own report (MSU_SCADA_Final_Report.pdf)
warns that the 2014 sets "contain some unintended patterns".

  (i)   STRUCTURE   attributes, missing values ('?') per class, the 'time' column
  (ii)  LABELS      duplicates and vectors with >1 category (without time and crc rate)
  (iii) TIME        is 'time' an absolute clock? order, interleaving of attacks and normal
  (iv)  ATTRIBUTION address fields per category
  (v)   PREVALENCE  category shares
  (vi)  SHORTCUT    XGBoost with/without time, crc rate, missing-value pattern; random and
                    time-ordered (or row-ordered) splits

  python scripts/inspect_morris_ics.py ~/datasets/morris_ics
Out: results/morris_ics_audit.txt
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

SETS = [("Morris-2 (gas pipeline 2014)", "gas_final.arff", "result"),
        ("Morris-3 (water tank 2014)", "water_final.arff", "result"),
        ("Morris-4 (gas pipeline 2015)", "IanArffDataset.arff", "categorized result")]
NAMES = {0: "normal", 1: "NMRI", 2: "CMRI", 3: "MSCI", 4: "MPCI", 5: "MFCI", 6: "DoS", 7: "Recon"}


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def load_arff(p):
    cols, rows = [], []
    with open(p, errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if s.lower().startswith("@attribute"):
                cols.append(s.split("'")[1] if "'" in s else s.split()[1])
            elif s.lower().startswith("@data"):
                break
        df = pd.read_csv(f, header=None, names=cols, na_values=["?"], quotechar="'",
                         skipinitialspace=True, low_memory=False)
    return df


def audit(title, df, lab):
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    print(f"\n################ {title} ################")
    n = len(df)
    y = df[lab].astype(int)
    df = df.drop(columns=[c for c in ("binary result", "categorized result", "specific result", "result") if c in df])
    print(f"(i) rows={n:,}; attributes={list(df.columns)}")
    miss = df.isna()
    pat = miss.apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)
    print("    missing-value pattern -> category (share of rows explained by majority):",
          pct(int(pd.crosstab(pat, y).max(axis=1).sum()), n), f"; {pat.nunique()} patterns")
    t = df["time"]
    print(f"    time: min={t.min():.6g} max={t.max():.6g}; non-decreasing steps "
          f"{pct(int((np.diff(t.values) >= 0).sum()), n - 1)}")
    feats = [c for c in df.columns if c not in ("time",)]
    content = [c for c in feats if "crc" not in c]
    print("(v) categories:", {NAMES[k]: f"{v:,} ({pct(v, n)})" for k, v in y.value_counts().sort_index().items()})
    X = df[content].fillna(-1)
    h = pd.util.hash_pandas_object(X, index=False)
    print(f"(ii) duplicates without time/crc: {pct(int(h.duplicated().sum()), n)}")
    pairs = pd.DataFrame({"h": h.values, "y": y.values}).drop_duplicates()
    multi = pairs["h"][pairs["h"].duplicated()].unique()
    rows = np.isin(h.values, multi)
    print(f"     vectors with >1 category: {len(multi):,} (rows {int(rows.sum()):,}, {pct(int(rows.sum()), n)}); "
          f"categories involved: {pd.Series(y.values[rows]).map(NAMES).value_counts().to_dict()}")
    # time / order: is the time column a clock (monotonic) and are attacks interleaved?
    order = np.argsort(t.values, kind="stable") if (np.diff(t.values) >= 0).mean() > 0.99 else np.arange(n)
    pos = np.empty(n, int); pos[order] = np.arange(n)
    print("(iii) position of each category in", "time order:" if order is not None else "row order:")
    for k in sorted(y.unique()):
        p = pos[(y == k).values]
        print(f"     {NAMES[k]:<7} rows {p.min():>7}..{p.max():<7}  runs: "
              f"{int((np.diff(np.sort(p)) > 1).sum()) + 1}")
    addr = [c for c in df.columns if "address" in c]
    for c in addr:
        print(f"(iv) {c}: {pd.crosstab(df[c], y.map(NAMES)).to_dict('index')}")
    # probe: random vs ordered split; with/without time, crc, missing pattern
    sets = {"all incl. time and crc": feats + ["time"], "without time": feats,
            "without time and crc": content}
    Xa = df.fillna(-1)
    Xa["_pat"] = pd.factorize(pat)[0]
    sets["without time and crc, + missing pattern only"] = ["_pat"]
    yy, names = y.values, [NAMES[k] for k in sorted(y.unique())]
    yi = pd.Series(yy).map({k: i for i, k in enumerate(sorted(y.unique()))}).values
    ded = ~pd.util.hash_pandas_object(Xa[feats + ["time"]], index=False).duplicated().values
    idx = np.where(ded)[0]
    a, b = train_test_split(idx, test_size=0.3, stratify=yi[idx], random_state=42)
    ta, tb = [], []
    for c in range(len(names)):
        ii = idx[yi[idx] == c]
        ii = ii[np.argsort(pos[ii], kind="stable")]
        cut = int(0.7 * len(ii)); ta += list(ii[:cut]); tb += list(ii[cut:])
    print("(vi) probe, XGBoost macro-F1 (dedup):")
    for split, (tr, te) in (("random 70/30", (a, b)),
                            ("ordered 70/30 per category", (np.array(ta), np.array(tb)))):
        print(f"   -- {split}")
        for sname, cols in sets.items():
            prm = {"objective": "multi:softmax", "num_class": len(names), "max_depth": 6,
                   "eta": 0.3, "tree_method": "hist", "nthread": 4, "seed": 42}
            m = xgb.train(prm, xgb.DMatrix(Xa.iloc[tr][cols], label=yi[tr]), num_boost_round=100)
            p = m.predict(xgb.DMatrix(Xa.iloc[te][cols])).astype(int)
            per = f1_score(yi[te], p, average=None, labels=range(len(names)), zero_division=0)
            print(f"     {sname:<46} macro-F1={per.mean():.3f}  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, per)))


def main(d):
    for title, f, lab in SETS:
        audit(title, load_arff(os.path.join(d, f)), lab)


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/morris_ics")
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    open("results/morris_ics_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
