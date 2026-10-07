#!/usr/bin/env python3
"""ReSIDS evaluation protocol on ToN_IoT (network, processed Zeek flows): the decisive test
after the audit (scripts/inspect_toniot.py). The audit found that 20.5 % of the normal
flows share their feature vector with attack flows (degenerate OTH/S0/SH connections from
non-attacker hosts); the question is whether the specialists turn those into false
positives, as CICIoT2023's session labels did (23.4 % FPR).

Protocol as cross_domain_results.py: XGBoost specialists (depth 4, eta 0.1, 10 rounds,
scale_pos_weight, seed 42), two per category, k-of-n fusion, GL = retained union of all
specialists, FL with fewer nodes. Categories = the six that hold on a time-ordered split
in the audit (scanning, dos, injection, ddos, password, xss) -> N = 12. Backdoor,
ransomware and mitm are left out (172 to 1,144 distinct vectors; they do not generalize).
Features: Zeek numeric + protocol fields; no addresses, ports, timestamps, uid or
free-text session fields (dns_query, ssl_subject/issuer, http_uri/referrer/user_agent).
Split: time-ordered per type, first 70 % train / last 30 % test. Train sample <= 500k
rows per attack type (all normal); test <= 300k per attack type, ALL test normal flows.

  python scripts/cross_domain_toniot.py
Out: results/cross_domain_toniot.txt
"""
import glob
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402
from inspect_toniot import CATF, NUM, read  # noqa: E402

CATS = ["scanning", "dos", "injection", "ddos", "password", "xss"]
KEEP = CATS + ["normal"]
TRAIN_CAP, TEST_CAP = 500_000, 300_000
SPLIT = os.environ.get("TONIOT_SPLIT", "time")      # time | session (see normal_group)


def main(d):
    files = sorted(glob.glob(os.path.join(d, "Network_dataset_*.csv")),
                   key=lambda p: int(re.findall(r"(\d+)\.csv", p)[0]))
    # pass 1: timestamps and types only -> per-type 70 % time cut and sampling rates
    lt = pd.concat([pd.read_csv(p, usecols=["ts", "type"], dtype={"type": "category"})
                    for p in files], ignore_index=True)
    # start of the sessions of the excluded types (backdoor, ransomware, mitm): Apr 28
    late = float(lt.loc[~lt["type"].isin(KEEP), "ts"].min())
    lt = lt[lt["type"].isin(KEEP)]
    cut = lt.groupby("type", observed=True)["ts"].quantile(0.7)
    tmax = lt.groupby("type", observed=True)["ts"].max()
    wins = np.array([(cut[c], tmax[c]) for c in CATS])

    def normal_group(ts):
        """SPLIT=session: normal flows inside an attack's test interval -> 'in-session'
        test, after the excluded sessions began -> 'late' test, otherwise train.
        SPLIT=time (default): last 30 % of normal by time is the test."""
        if SPLIT == "session":
            ins = ((ts[:, None] >= wins[:, 0]) & (ts[:, None] <= wins[:, 1])).any(1)
            return np.where(ins, "in-session", np.where(ts >= late, "late", "train"))
        return np.where(ts >= cut["normal"], "late", "train")

    if SPLIT == "session":
        cut["normal"] = np.inf          # normal train/test is decided by normal_group
    ntr = lt[lt["ts"] < lt["type"].map(cut).astype(float)].groupby("type", observed=True).size()
    nte = lt.groupby("type", observed=True).size() - ntr
    ptr = {t: (1.0 if t == "normal" else min(1.0, TRAIN_CAP / ntr[t])) for t in KEEP}
    pte = {t: (1.0 if t == "normal" else min(1.0, TEST_CAP / nte[t])) for t in KEEP}
    del lt
    print(f"split={SPLIT}; 70 % time cut per type:",
          {t: str(pd.to_datetime(cut[t], unit="s")) for t in KEEP if np.isfinite(cut[t])},
          f"; late sessions from {pd.to_datetime(late, unit='s')}")
    # pass 2: features of the sampled rows
    rng = np.random.default_rng(C.SEED)
    parts = []
    for p in files:
        df = read(p)
        df = df[df["type"].isin(KEEP)]
        t = df["type"].astype(str).values
        tsv = df["ts"].values.astype(float)
        grp = np.where(t == "normal", normal_group(tsv), "attack")
        tr = np.where(t == "normal", grp == "train", tsv < pd.Series(t).map(cut).values)
        pr = np.where(tr, pd.Series(t).map(ptr).values, pd.Series(t).map(pte).values)
        keep = rng.random(len(df)) < pr
        q = df.loc[keep, NUM + CATF + ["ts"]].copy()       # conn_state is already in CATF
        for c in CATF:
            q[c] = q[c].astype(str)
        q["_type"], q["_train"], q["_grp"] = t[keep], tr[keep], grp[keep]
        q["_zero"] = (df.loc[keep, "src_bytes"].fillna(0).values == 0) & (df.loc[keep, "dst_bytes"].values == 0)
        q["_state"] = q["conn_state"].values
        parts.append(q)
        del df
    s = pd.concat(parts, ignore_index=True); del parts
    for c in CATF:
        s[c] = pd.factorize(s[c])[0].astype(np.int32)
    X = s[NUM + CATF].to_numpy(np.float32)
    cat = s["_type"].values
    trm = s["_train"].values
    print("train per type:", pd.Series(cat[trm]).value_counts().to_dict())
    print("test  per type:", pd.Series(cat[~trm]).value_counts().to_dict())
    # specialists: node i serves CATS[i % 6], half i // 6 of its train rows, 1/12 of normal
    N = 2 * len(CATS)
    Xtr, ctr = X[trm], cat[trm]
    ben = np.where(ctr == "normal")[0]
    shares = np.array_split(rng.permutation(ben), N)
    halves = {c: np.array_split(rng.permutation(np.where(ctr == c)[0]), 2) for c in CATS}
    models = []
    for i in range(N):
        c = CATS[i % len(CATS)]
        pos, neg = halves[c][i // len(CATS)], shares[i]
        rows = np.r_[pos, neg]
        p = dict(C.XGB); p["scale_pos_weight"] = len(neg) / max(len(pos), 1)
        models.append(xgb.train(p, xgb.DMatrix(Xtr[rows], label=np.r_[np.ones(len(pos)), np.zeros(len(neg))]),
                                num_boost_round=10))
    Xte, cte = X[~trm], cat[~trm]
    V = np.column_stack([(m.predict(xgb.DMatrix(Xte)) >= 0.5).astype(np.int8) for m in models])
    atk = cte != "normal"
    for k in (1, 2):
        g = C.stats(V.sum(1) >= k, atk)
        print(f"GL (union of {N}) k>={k}: recall {g['recall']:.2f}%  FPR {g['FPR']:.3f}%  "
              f"F1 {g['F1']:.2f}  F1@13.75 {g['F1@13.75']:.2f}")
    f2 = V.sum(1) >= 2
    print("per type recall (GL k>=2):", {c: round(100 * f2[cte == c].mean(), 1) for c in CATS})
    for n in (6, 3):
        g = C.stats(V[:, :n].sum(1) >= 2, atk)
        print(f"FL k>=2 with {n} nodes: recall {g['recall']:.1f}%  FPR {g['FPR']:.3f}%")
    f2 = V.sum(1) >= 2
    gte = s["_grp"].values[~trm]
    for gname in ("in-session", "late"):
        m = gte == gname
        if m.any():
            sel = atk | m
            g = C.stats(f2[sel], atk[sel])
            print(f"normal test group '{gname}': {int(m.sum()):,} flows, FPR (k>=2) {100 * f2[m].mean():.3f}%"
                  f"  -> with all attack test flows: F1 {g['F1']:.2f}, F1@13.75 {g['F1@13.75']:.2f}")
    nm = ~atk
    st = s["_state"].values[~trm]
    zero = s["_zero"].values[~trm]
    print(f"\nnormal test flows: {int(nm.sum()):,}; flagged (k>=2): {int(f2[nm].sum()):,}")
    print("  FPR by connection state:",
          {k: f"{100 * f2[nm & (st == k)].mean():.2f}% of {int((nm & (st == k)).sum()):,}"
           for k in pd.Series(st[nm]).value_counts().head(6).index})
    print(f"  FPR on zero-byte normal flows: {100 * f2[nm & zero].mean():.2f}% of {int((nm & zero).sum()):,}; "
          f"on the others: {100 * f2[nm & ~zero].mean():.2f}% of {int((nm & ~zero).sum()):,}")
    print("  specialists firing on false positives:",
          {CATS[i % 6] + f"#{i // 6}": int(V[nm & f2, i].sum()) for i in range(N)})


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/toniot")
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    open(f"results/cross_domain_toniot{'' if SPLIT == 'time' else '_' + SPLIT}.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
