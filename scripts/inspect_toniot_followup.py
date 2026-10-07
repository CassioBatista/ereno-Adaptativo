#!/usr/bin/env python3
"""Follow-up checks after inspect_toniot.py (low memory: a few columns per file).
  1. header of the file whose lines have 47 fields;
  2. the tagging rule: the published range 192.168.159.30-39 never appears; the attack
     sources are 192.168.1.30-39. Share of attack and normal flows involving that range;
  3. normal flows whose feature vector also carries an attack type: which types, which
     connection states, and whether they involve the attacker range.
Out: results/toniot_followup.txt
"""
import glob
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_toniot import CATF, NUM, read  # noqa: E402

RANGE = {f"192.168.1.{i}" for i in range(30, 40)}


def main(d):
    files = sorted(glob.glob(os.path.join(d, "Network_dataset_*.csv")),
                   key=lambda p: int(re.findall(r"(\d+)\.csv", p)[0]))
    heads = {}
    for p in files:
        with open(p, encoding="utf-8-sig") as f:
            heads[os.path.basename(p)] = f.readline().rstrip("\n").split(",")
    ref = heads["Network_dataset_1.csv"]
    for k, h in heads.items():
        if h != ref:
            extra = [c for c in h if c not in ref]
            print(f"{k}: {len(h)} columns; not in file 1: {extra}; first columns {h[:3]}")
    hs, ty, att, st = [], [], [], []
    for p in files:
        df = read(p)
        hs.append(pd.util.hash_pandas_object(df[NUM + CATF].fillna(-1), index=False).values)
        ty.append(df["type"].astype(str).values)
        att.append((df["src_ip"].astype(str).isin(RANGE) | df["dst_ip"].astype(str).isin(RANGE)).values)
        st.append(df["conn_state"].astype(str).values)
        del df
    h, t, a, s = map(np.concatenate, (hs, ty, att, st))
    print("\nflows involving 192.168.1.30-39 (source or destination):")
    for k in pd.unique(t):
        m = t == k
        print(f"  {k:<12} {100 * a[m].mean():6.2f}%  of {int(m.sum()):,}")
    pairs = pd.DataFrame({"h": h, "t": t}).drop_duplicates()
    att_h = set(pairs.loc[pairs["t"] != "normal", "h"])
    nm = t == "normal"
    coll = nm & pd.Series(h).isin(att_h).values
    print(f"\nnormal flows sharing a vector with an attack type: {int(coll.sum()):,} "
          f"({100 * coll.sum() / nm.sum():.1f}% of normal)")
    print("  their connection states:", pd.Series(s[coll]).value_counts().head(6).to_dict())
    print(f"  involving the attacker range: {100 * a[coll].mean():.1f}%")
    other = pairs[pairs["h"].isin(set(h[coll])) & (pairs["t"] != "normal")]["t"].value_counts()
    print("  attack types sharing those vectors (distinct vectors):", other.head(8).to_dict())
    ab = ~nm
    print(f"\nattack flows NOT involving the attacker range: {int((ab & ~a).sum()):,} "
          f"({100 * (ab & ~a).sum() / ab.sum():.2f}% of attack flows)")


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/toniot")
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    open("results/toniot_followup.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
