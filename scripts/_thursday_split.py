#!/usr/bin/env python3
"""Where does the corrected thursday.csv split into morning (web attacks, which is what
our v2 used) and afternoon (Infiltration)? Needed to compare BENIGN like for like."""
import sys
import zipfile

import pandas as pd

z = zipfile.ZipFile(sys.argv[1])
d = pd.read_csv(z.open("thursday.csv"), usecols=["Timestamp", "Label"])
d["Label"] = d["Label"].str.strip()
d["t"] = pd.to_datetime(d["Timestamp"], errors="coerce")
print("timestamp example:", d["Timestamp"].iloc[0], "| unparsed:", int(d["t"].isna().sum()))
print("day:", d["t"].min(), "->", d["t"].max())
for k in ("Web Attack", "Infiltration"):
    s = d[d["Label"].str.contains(k)]["t"]
    print(f"{k:14s} {s.min()} -> {s.max()}  n={len(s):,}")

web_end = d[d["Label"].str.contains("Web Attack")]["t"].max()
inf_start = d[d["Label"].str.contains("Infiltration")]["t"].min()
cut = web_end + (inf_start - web_end) / 2
morning = d[d["t"] < cut]
print(f"\ncut at midpoint {cut} (web ends {web_end}, infiltration starts {inf_start})")
print("morning labels:")
for lb, n in morning["Label"].value_counts().items():
    print(f"  {lb:42s} {n:10,}")
print(f"afternoon BENIGN: {int((d[d['t'] >= cut]['Label'] == 'BENIGN').sum()):,}")
