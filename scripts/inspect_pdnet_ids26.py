#!/usr/bin/env python3
"""Inspect PDNet-IDS26 (Lui et al., SBCAS 2026) before any profile is defined.

Answers:
  1. SCHEMA      — columns, types, constants
  2. LABELS      — per-class counts (the release is undersampled to 25 % per class)
  3. DUPLICATES  — exact and feature-level duplicates, conflicting labels
  4. PATIENTS    — subjects, age, sex and test_time per class: does the patient give
                   the class away (the paper's SHAP ranks Age high for FDI, yet the
                   attack does not touch age)?
  5. PAYLOAD     — DoS payloads, replayed vectors seen in benign traffic, FDI
                   plausibility (total_UPDRS >= motor_UPDRS)
  6. NETWORK     — ports, sequence numbers, TTL per class (session identifiers)
  7. SHORTCUTS   — XGBoost F1 per class under feature sets of decreasing leakage,
                   with stratified folds and with folds grouped by patient

  python scripts/inspect_pdnet_ids26.py ~/datasets/pdnet_ids26/PDNet-IDS26.csv
Out: results/pdnet_ids26_inspection.txt
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

NET = ["IAT", "Length", "IP_Len", "IP_TTL", "IP_Flags", "TCP_SrcPort", "TCP_DstPort",
       "TCP_Seq", "TCP_Ack", "TCP_HdrLen", "TCP_Len", "TCP_WinSize", "TCP_Flags",
       "MQTT_MsgType", "MQTT_QoS", "MQTT_Retain"]
PATIENT = ["subject_id", "age", "sex", "test_time"]
BIO = ["motor_UPDRS", "total_UPDRS", "Jitter_pct", "Jitter_Abs", "Jitter_RAP",
       "Jitter_PPQ5", "Jitter_DDP", "Shimmer", "Shimmer_dB", "Shimmer_APQ3",
       "Shimmer_APQ5", "Shimmer_APQ11", "Shimmer_DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]
SESSION = ["TCP_SrcPort", "TCP_Seq", "TCP_Ack"]          # identify a connection
LABEL = "Attack_Cat"


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main(path):
    df = pd.read_csv(path, low_memory=False)
    n = len(df)
    section(f"1. SCHEMA — {n:,} rows x {df.shape[1]} columns")
    nonnum = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    print("non-numeric:", {c: df[c].unique()[:6].tolist() for c in nonnum})
    print("constant:", [c for c in df.columns if df[c].nunique(dropna=False) <= 1])
    print("NaN cells:", int(df.isna().sum().sum()))
    print("columns not in NET/PATIENT/BIO:",
          [c for c in df.columns if c not in NET + PATIENT + BIO + [LABEL]])

    section("2. LABELS")
    vc = df[LABEL].value_counts()
    for k, v in vc.items():
        print(f"  {k:<10}{v:>8,}  {pct(v, n)}")
    runs = int(df[LABEL].ne(df[LABEL].shift()).sum())
    print(f"label runs in file order: {runs:,} (4 = class blocks; ~n = shuffled)")

    section("3. DUPLICATES")
    print("exact duplicate rows:", int(df.duplicated().sum()))
    for name, cols in (("bio only", BIO), ("network only (no session ids)",
                                            [c for c in NET if c not in SESSION])):
        g = df.groupby(cols, dropna=False)[LABEL].nunique()
        multi = g[g > 1]
        print(f"{name}: {df.duplicated(subset=cols).sum():,} duplicate vectors; "
              f"{len(multi):,} vectors shared by >1 class")

    section("4. PATIENTS")
    for k in vc.index:
        s = df[df[LABEL] == k]
        print(f"  {k:<8} subjects={s['subject_id'].nunique():>3}  "
              f"age mean={s['age'].mean():5.1f} [{s['age'].min():.0f}-{s['age'].max():.0f}]  "
              f"sex=1: {pct(int((s['sex'] == 1).sum()), len(s))}  "
              f"test_time [{s['test_time'].min():.1f}, {s['test_time'].max():.1f}]")
    real = df[df["subject_id"] > 0] if (df[LABEL] == "DOS").any() else df
    subj = {k: set(real.loc[real[LABEL] == k, "subject_id"]) for k in vc.index}
    keys = list(subj)
    print("subject overlap (Jaccard) between classes:")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            u = subj[a] | subj[b]
            if u:
                print(f"  {a:<8}{b:<8}{len(subj[a] & subj[b]) / len(u):.2f}  "
                      f"(shared {len(subj[a] & subj[b])})")
    # Does the subject alone predict the class?
    maj = df.groupby("subject_id")[LABEL].agg(lambda s: s.value_counts().iloc[0])
    print(f"subject_id alone -> class (majority per subject): accuracy {pct(int(maj.sum()), n)}")
    maj = df.groupby("age")[LABEL].agg(lambda s: s.value_counts().iloc[0])
    print(f"age alone -> class (majority per age value): accuracy {pct(int(maj.sum()), n)}")

    section("5. PAYLOAD")
    dos = df[df[LABEL] == "DOS"]
    if len(dos):
        zero = (dos[BIO + PATIENT].abs().sum(axis=1) == 0).mean()
        print(f"DoS rows with an all-zero biomedical payload: {100 * zero:.1f}%")
    ben = df[df[LABEL].str.lower() == "benign"]
    bkeys = set(map(tuple, ben[PATIENT + BIO].round(6).values))
    for k in vc.index:
        s = df[df[LABEL] == k]
        inb = np.mean([tuple(r) in bkeys for r in s[PATIENT + BIO].round(6).values])
        bad = (s["total_UPDRS"] < s["motor_UPDRS"]).mean()
        print(f"  {k:<8} payload vector also seen in benign: {100 * inb:5.1f}%   "
              f"total_UPDRS < motor_UPDRS: {100 * bad:5.1f}%")
    # Original telemonitoring records: is a given (subject, test_time) seen with
    # different UPDRS values across classes (i.e., perturbed copies)?
    key = df[df["subject_id"] > 0].groupby(["subject_id", "test_time"])
    var = key["motor_UPDRS"].nunique()
    print(f"(subject, test_time) records: {len(var):,}; with >1 motor_UPDRS value: "
          f"{int((var > 1).sum()):,}")

    section("6. NETWORK")
    for k in vc.index:
        s = df[df[LABEL] == k]
        print(f"  {k:<8} src ports={s['TCP_SrcPort'].nunique():>5}  "
              f"dst ports={s['TCP_DstPort'].nunique():>3}  TTL={sorted(s['IP_TTL'].unique())[:4]}  "
              f"Length median={s['Length'].median():.0f}  IAT median={s['IAT'].median():.4f}  "
              f"Seq range=[{s['TCP_Seq'].min()}, {s['TCP_Seq'].max()}]")
    maj = df.groupby("TCP_SrcPort")[LABEL].agg(lambda s: s.value_counts().iloc[0])
    print(f"TCP_SrcPort alone -> class: accuracy {pct(int(maj.sum()), n)}")

    section("7. SHORTCUTS — XGBoost macro/per-class F1 (5 folds)")
    try:
        import xgboost as xgb
        from sklearn.metrics import f1_score
        from sklearn.model_selection import GroupKFold, StratifiedKFold
        X = df.drop(columns=[LABEL]).copy()
        for c in nonnum:
            if c != LABEL:
                X[c] = X[c].astype(str).astype("category").cat.codes
        y = df[LABEL].astype("category")
        classes = list(y.cat.categories)
        yc = y.cat.codes.values
        sets = {
            "A all features": list(X.columns),
            "B paper (no subject_id, no src port)": [c for c in X.columns
                                                     if c not in ("subject_id", "TCP_SrcPort")],
            "C B minus session ids (seq/ack)": [c for c in X.columns
                                               if c not in ["subject_id"] + SESSION],
            "D C minus patient identity (age/sex/test_time)": [c for c in X.columns
                                                                if c not in PATIENT + SESSION],
        }
        groups = df["subject_id"].values
        for split in ("stratified", "grouped by patient"):
            print(f"\n-- folds: {split}")
            print(f"{'feature set':<50}{'macro':>7}" + "".join(f"{c:>8}" for c in classes))
            for name, cols in sets.items():
                pred = np.empty(n, dtype=int)
                if split == "stratified":
                    it = StratifiedKFold(5, shuffle=True, random_state=42).split(X, yc)
                else:
                    it = GroupKFold(5).split(X, yc, groups)
                for tr, te in it:
                    m = xgb.XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist",
                                          n_jobs=4, random_state=42)
                    m.fit(X.iloc[tr][cols], yc[tr])
                    pred[te] = m.predict(X.iloc[te][cols])
                f = f1_score(yc, pred, average=None)
                print(f"{name:<50}{f1_score(yc, pred, average='macro'):>7.3f}"
                      + "".join(f"{v:>8.3f}" for v in f))
        print("\n(grouped folds: DoS rows carry subject 0 and fall in one fold; the "
              "patient-grouped split tests whether detection survives unseen patients)")
    except Exception as e:
        print("skipped:", repr(e))


if __name__ == "__main__":
    p = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                           else "~/datasets/pdnet_ids26/PDNet-IDS26.csv")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(p)
    os.makedirs("results", exist_ok=True)
    open("results/pdnet_ids26_inspection.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
