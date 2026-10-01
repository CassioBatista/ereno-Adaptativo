#!/usr/bin/env python3
"""Inspect the corrected CIC-IDS-2017 (Liu, Engelen et al., CNS 2022) against the
`all_in_one_cicids_v2` this project uses (original MachineLearningCVE, 3 days).

Answers three questions before any migration is decided:
  1. COLUMNS   — does the corrected layout map onto our F1..F78 (canonical
                 MachineLearningCVE order, docs/DATASETS.md)? Which indices move,
                 which features are missing, which are new?
  2. LABELS    — label distribution over the same 3 days, including the new
                 'Attempted' labels, and how they relate to our 10 attack classes.
  3. TAILS     — counts of the weak classes (Heartbleed 11, sql 21, xss 652, brute
                 1,507 in v2) after the correction.

  python scripts/inspect_cicids2017_improved.py ~/datasets/cicids2017_improved/CICIDS2017_improved.zip
Out: results/cicids2017_improved_inspection.{txt,json}
"""
import json
import re
import sys
import zipfile
from collections import Counter

import pandas as pd

# Canonical MachineLearningCVE order: F1 = Destination Port ... F78 = Idle Min
# (docs/DATASETS.md proves our F-indices follow it, by column sums).
MLCVE = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min",
]
assert len(MLCVE) == 78

# Name drift between CICFlowMeter generations (ISCX 2017 export vs the newer tool).
ALIASES = {
    "total fwd packet": "total fwd packets",
    "total bwd packets": "total backward packets",
    "total length of fwd packet": "total length of fwd packets",
    "total length of bwd packet": "total length of bwd packets",
    "packet length min": "min packet length",
    "packet length max": "max packet length",
    "cwr flag count": "cwe flag count",
    "fwd segment size avg": "avg fwd segment size",
    "bwd segment size avg": "avg bwd segment size",
    "fwd bytes/bulk avg": "fwd avg bytes/bulk",
    "fwd packet/bulk avg": "fwd avg packets/bulk",
    "fwd bulk rate avg": "fwd avg bulk rate",
    "bwd bytes/bulk avg": "bwd avg bytes/bulk",
    "bwd packet/bulk avg": "bwd avg packets/bulk",
    "bwd bulk rate avg": "bwd avg bulk rate",
    "fwd init win bytes": "init_win_bytes_forward",
    "bwd init win bytes": "init_win_bytes_backward",
    "fwd act data pkts": "act_data_pkt_fwd",
    "fwd seg size min": "min_seg_size_forward",
    "dst port": "destination port",
}

# our 10 attack classes (scripts/regenerate_cicids_v2.py) and their original labels
OURS = {
    "ftp": "FTP-Patator", "ssh": "SSH-Patator", "hulk": "DoS Hulk",
    "GoldenEye": "DoS GoldenEye", "slowloris": "DoS slowloris",
    "Slowhttptest": "DoS Slowhttptest", "Heartbleed": "Heartbleed",
    "brute": "Web Attack - Brute Force", "xss": "Web Attack - XSS",
    "sql": "Web Attack - SQL Injection",
}
V2_COUNTS = {"BENIGN": 1_039_547, "hulk": 230_124, "GoldenEye": 10_293, "ftp": 7_935,
             "ssh": 5_897, "slowloris": 5_796, "Slowhttptest": 5_499, "brute": 1_507,
             "xss": 652, "sql": 21, "Heartbleed": 11}
DAYS = ["tuesday.csv", "wednesday.csv", "thursday.csv"]


def norm(s):
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return ALIASES.get(s, s)


def classify(label):
    """Map a corrected-dataset label onto our class names, keeping Attempted apart."""
    lb = str(label).strip()
    low = lb.lower()
    attempted = "attempted" in low
    base = re.sub(r"\s*-\s*attempted.*$", "", lb, flags=re.I).strip()
    bl = base.lower()
    hit = None
    if bl == "benign":
        hit = "BENIGN"
    elif "ftp" in bl:
        hit = "ftp"
    elif "ssh" in bl:
        hit = "ssh"
    elif "hulk" in bl:
        hit = "hulk"
    elif "goldeneye" in bl:
        hit = "GoldenEye"
    elif "slowloris" in bl:
        hit = "slowloris"
    elif "slowhttptest" in bl:
        hit = "Slowhttptest"
    elif "heartbleed" in bl:
        hit = "Heartbleed"
    elif "xss" in bl:
        hit = "xss"
    elif "sql" in bl:
        hit = "sql"
    elif "brute" in bl and "web" in bl:
        hit = "brute"
    return hit, attempted


def main():
    zpath = sys.argv[1]
    z = zipfile.ZipFile(zpath)
    out, rep = [], {}

    def say(s=""):
        print(s)
        out.append(s)

    # ---------------- 1. columns ----------------
    with z.open(DAYS[0]) as fh:
        head = pd.read_csv(fh, nrows=5, encoding="latin1")
    cols = [c.strip() for c in head.columns]
    say(f"=== 1. COLUMNS ({len(cols)} in the corrected export) ===")
    ncols = {norm(c): i for i, c in enumerate(cols)}
    mapping, missing = {}, []
    for f, name in enumerate(MLCVE, 1):
        n = norm(name)
        if n in ncols:
            mapping[f] = cols[ncols[n]]
        else:
            missing.append((f, name))
    canon = {norm(n) for n in MLCVE}
    extra = [c for c in cols if norm(c) not in canon]
    say(f"our F1..F78 found by name: {len(mapping)}/78")
    say(f"missing ({len(missing)}): {missing}")
    say(f"present only in the corrected export ({len(extra)}): {extra}")
    feat_cols = [c for c in cols if norm(c) in canon]
    same_order = [norm(c) for c in feat_cols] == [norm(n) for n in MLCVE if norm(n) in
                                                   {norm(c) for c in feat_cols}]
    say(f"relative order of the shared features preserved: {same_order}")
    # where do our GRASP-selected features land?
    sel = json.load(open("features/all_in_one_cicids_combined.json"))["features"]
    lost = [f for f in sel if f not in mapping]
    say(f"our combined GRASP selection ({len(sel)} features): "
        f"{len(sel) - len(lost)} map by name, lost {lost}")
    rep["columns"] = {"n_cols": len(cols), "mapped": len(mapping), "missing": missing,
                      "extra": extra, "same_order": same_order, "grasp_lost": lost,
                      "header": cols}

    # ---------------- 2/3. labels and tails ----------------
    lab_col = next(c for c in cols if c.lower() == "label")
    say()
    say("=== 2. LABELS, same 3 days (tuesday, wednesday, thursday) ===")
    raw = Counter()
    per_day = {}
    for d in DAYS:
        with z.open(d) as fh:
            s = pd.read_csv(fh, usecols=[lambda c: c.strip().lower() == "label"][0],
                            encoding="latin1")
        s = s.iloc[:, 0].astype(str).str.strip()
        c = Counter(s)
        per_day[d] = dict(c)
        raw.update(c)
        say(f"  {d}: {len(s):,} flows")
    say()
    say(f"{'label':45s} {'flows':>12s}")
    for lb, n in raw.most_common():
        say(f"  {lb:43s} {n:12,}")

    mapped, attempted, other = Counter(), Counter(), Counter()
    for lb, n in raw.items():
        hit, att = classify(lb)
        if hit is None:
            other[lb] += n
        elif att:
            attempted[hit] += n
        else:
            mapped[hit] += n
    say()
    say("=== 3. PER CLASS: v2 (original) vs corrected ===")
    say(f"{'class':14s} {'v2 orig':>11s} {'corrected':>11s} {'delta':>9s} {'+Attempted':>11s}")
    rows = []
    for k in ["BENIGN", "hulk", "GoldenEye", "ftp", "ssh", "slowloris", "Slowhttptest",
              "brute", "xss", "sql", "Heartbleed"]:
        a, b, t = V2_COUNTS[k], mapped.get(k, 0), attempted.get(k, 0)
        d = f"{100 * (b - a) / a:+.1f}%" if a else "n/a"
        say(f"{k:14s} {a:11,} {b:11,} {d:>9s} {t:11,}")
        rows.append({"class": k, "v2": a, "corrected": b, "attempted": t})
    tot_att = sum(attempted.values())
    tot_atk = sum(v for k, v in mapped.items() if k != "BENIGN")
    say()
    say(f"Attempted flows total: {tot_att:,} — would be counted as attack in the original, "
        f"they are NOT an attack payload")
    say(f"  share vs proper attack flows of the same classes: "
        f"{100 * tot_att / max(tot_atk + tot_att, 1):.1f}%")
    if other:
        say(f"labels outside our 10 classes (e.g. Thursday-afternoon Infiltration): "
            f"{dict(other)}")
    rep["labels"] = {"raw": dict(raw), "per_day": per_day, "per_class": rows,
                     "attempted_by_class": dict(attempted), "outside": dict(other)}

    with open("results/cicids2017_improved_inspection.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    with open("results/cicids2017_improved_inspection.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("\n-> results/cicids2017_improved_inspection.{txt,json}")


if __name__ == "__main__":
    main()
