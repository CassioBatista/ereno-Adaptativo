#!/usr/bin/env python3
"""Audit and evaluation protocol on the UNSW IoT attack traces (Hamza et al., ACM SOSR
2019; iotanalytics.unsw.edu.au/attack-data, MIT-0; checksums in
results/unsw_iot_attack_md5sums.txt).

Data: for 10 consumer IoT devices, one row per device and minute with packet and byte
counters of the flows of the device's MUD profile, over 32 to 49 days; annotations give,
per device, the start and end of each attack (10 minutes), the MUD flows it affects and its
type and rate (1, 10, 100 packets/s; launched from the local network or the Internet).

Common feature space: each device has its own MUD columns, so counters are summed per
direction (From/To), reach (Local/Internet) and protocol (Udp, Tcp, Icmp, Arp, other),
for packets and bytes, plus NoOfFlows -> 41 features.

Label set unsw-iot-7: TcpSynReflection, TcpSynDevice, ArpSpoof, UdpReflection (Ssdp +
Snmp), UdpDevice, PingOfDeath, Smurf. A device-minute is labelled with the attack whose
interval covers it.

  (i)   STRUCTURE   rows, minute spacing, gaps per device
  (ii)  LABELS      do the counters of the affected protocol rise inside each interval,
                    against the hour before? (per type and rate)
  (iii) TIME        attacks spread over days, interleaved with normal minutes
  (iv)  ATTRIBUTION victim device known; attacker local or remote
  (v)   PREVALENCE  attack minutes
  (vi)  SHORTCUT    XGBoost with/without device and time, random vs interval-grouped
                    time-ordered split
Then the ReSIDS protocol (14 specialists, k-of-n, GL union vs FL with fewer nodes),
split by time with attack intervals kept whole.

  python scripts/inspect_unsw_iot.py
Out: results/unsw_iot_audit.txt
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

D = os.path.expanduser("~/datasets/unsw_iot_attack")
PROTO = ["Udp", "Tcp", "Icmp", "Arp"]
CATS = ["TcpSynReflection", "TcpSynDevice", "ArpSpoof", "UdpReflection", "UdpDevice",
        "PingOfDeath", "Smurf"]
FAMILY = {"TcpSynReflection": "Tcp", "TcpSynDevice": "Tcp", "ArpSpoof": "Arp",
          "UdpReflection": "Udp", "UdpDevice": "Udp", "PingOfDeath": "Icmp", "Smurf": "Icmp"}


def category(t):
    base = re.sub(r"\d+[LW]2D.*$", "", t)
    return "UdpReflection" if base in ("Ssdp", "Snmp") else base


def features(df):
    out = pd.DataFrame(index=df.index)
    for d in ("From", "To"):
        for r in ("Local", "Internet"):
            for unit in ("Packet", "Byte"):
                cols = {p: [c for c in df.columns if c.startswith(f"{d}{r}{p}") and c.endswith(unit)]
                        for p in PROTO}
                other = [c for c in df.columns if c.startswith(f"{d}{r}") and c.endswith(unit)
                         and not any(c.startswith(f"{d}{r}{p}") for p in PROTO)]
                for p in PROTO:
                    out[f"{d}{r}{p}{unit}"] = df[cols[p]].sum(axis=1) if cols[p] else 0
                out[f"{d}{r}Other{unit}"] = df[other].sum(axis=1) if other else 0
    out["NoOfFlows"] = df["NoOfFlows"] if "NoOfFlows" in df else 0
    return out


def load():
    rows, ann = [], []
    for f in sorted(glob.glob(f"{D}/flowdata/flowdata/*_flowstats.csv")):
        dev = os.path.basename(f).split("_")[0]
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        x = features(df)
        x["_dev"], x["_ts"] = dev, df["Timestamp"].values / 1000.0
        rows.append(x)
        # some flow lists contain a comma: start, end, flows..., type
        recs = []
        for line in open(f"{D}/annotations/annotations/{dev}.csv"):
            p = line.strip().split(",")
            if len(p) >= 4:
                recs.append({"start": int(p[0]), "end": int(p[1]), "flows": ",".join(p[2:-1]),
                             "type": p[-1]})
        a = pd.DataFrame(recs)
        a["_dev"] = dev
        ann.append(a)
    X, A = pd.concat(rows, ignore_index=True), pd.concat(ann, ignore_index=True)
    A["cat"] = A["type"].map(category)
    A["rate"] = A["type"].str.extract(r"(\d+)[LW]2D")[0].astype(int)
    A["origin"] = A["type"].str.extract(r"\d+([LW])2D")[0].map({"L": "local", "W": "internet"})
    A["iid"] = np.arange(len(A))
    X["_cat"], X["_iid"] = "Normal", -1
    for _, r in A.iterrows():
        m = (X["_dev"] == r["_dev"]) & (X["_ts"] >= r["start"]) & (X["_ts"] < r["end"])
        X.loc[m, "_cat"], X.loc[m, "_iid"] = r["cat"], r["iid"]
    return X, A


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main():
    X, A = load()
    feats = [c for c in X.columns if not c.startswith("_")]
    n = len(X)
    print("=== (i) STRUCTURE ===")
    for d, g in X.groupby("_dev"):
        dt = np.diff(np.sort(g["_ts"].values))
        print(f"  {d}: {len(g):,} minutes, {pd.to_datetime(g['_ts'].min(), unit='s'):%Y-%m-%d} .. "
              f"{pd.to_datetime(g['_ts'].max(), unit='s'):%Y-%m-%d}; gaps > 2 min: {int((dt > 120).sum())}")
    print(f"total device-minutes {n:,}; features {len(feats)}")
    print("\n=== (ii) LABELS: affected protocol counters inside each interval vs the hour before ===")
    res = []
    for _, r in A.iterrows():
        g = X[X["_dev"] == r["_dev"]]
        fam = FAMILY.get(r["cat"])
        if fam is None:
            continue
        cols = [c for c in feats if fam in c and c.endswith("Packet")]
        ins = g[(g["_ts"] >= r["start"]) & (g["_ts"] < r["end"])][cols].sum(axis=1)
        pre = g[(g["_ts"] >= r["start"] - 3600) & (g["_ts"] < r["start"] - 60)][cols].sum(axis=1)
        if len(ins) and len(pre):
            res.append({"cat": r["cat"], "rate": r["rate"], "origin": r["origin"], "minutes": len(ins),
                        "ratio": (ins.mean() + 1) / (pre.mean() + 1),
                        "rise": ins.mean() - pre.mean(), "expected": r["rate"] * 60})
    R = pd.DataFrame(res)
    print(R.groupby(["cat", "rate"]).agg(intervals=("ratio", "size"), median_ratio=("ratio", "median"),
                                         median_rise_pkt_min=("rise", "median"),
                                         expected_pkt_min=("expected", "first"),
                                         visible_share=("ratio", lambda s: (s > 2).mean())).round(2).to_string())
    print(f"annotated minutes found in the data: {int((X['_iid'] >= 0).sum()):,} "
          f"(intervals {len(A)}, x10 min = {10 * len(A):,})")
    print("\n=== (iii) TIME ===")
    X["_day"] = pd.to_datetime(X["_ts"], unit="s").dt.date
    days = X.groupby("_day")["_cat"].agg(lambda s: (s != "Normal").sum())
    print(f"days with data {len(days)}, with attacks {int((days > 0).sum())}; "
          f"normal minutes on attack days {int(X[X['_day'].isin(days[days > 0].index) & (X['_cat'] == 'Normal')].shape[0]):,}")
    print("intervals per category:", A["cat"].value_counts().to_dict())
    print("\n=== (iv) ATTRIBUTION ===")
    print("victim device per interval: known (annotation per device); origin:",
          A["origin"].value_counts().to_dict(), "; devices:", A["_dev"].nunique())
    print("\n=== (v) PREVALENCE ===")
    vc = X["_cat"].value_counts()
    for k, v in vc.items():
        print(f"  {k:<18}{v:>9,}  {pct(v, n)}")

    print("\n=== (vi) SHORTCUT PROBE — XGBoost macro-F1 over categories ===")
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    y, names = pd.factorize(X["_cat"], sort=True)
    Xf = X[feats].copy()
    Xf["_devc"] = pd.factorize(X["_dev"])[0]
    Xf["_ts"] = X["_ts"]
    # balanced probe sample: all attack minutes, 30k normal
    rng = np.random.RandomState(42)
    norm = np.where(X["_cat"].values == "Normal")[0]
    idx = np.sort(np.r_[np.where(X["_cat"].values != "Normal")[0], rng.choice(norm, 30_000, replace=False)])
    a, b = train_test_split(idx, test_size=0.3, stratify=y[idx], random_state=42)
    # time split with intervals kept whole: per category, intervals sorted by start
    tr = set()
    for c in names:
        if c == "Normal":
            continue
        iv = A[A["cat"] == c].sort_values("start")["iid"].values
        tr |= set(iv[: int(round(0.7 * len(iv)))])
    tcut = {d: g["_ts"].quantile(0.7) for d, g in X.groupby("_dev")}
    is_tr = np.where(X["_cat"].values == "Normal", X["_ts"].values < X["_dev"].map(tcut).values,
                     np.isin(X["_iid"].values, list(tr)))
    ta, tb = idx[is_tr[idx]], idx[~is_tr[idx]]
    for sname, cols in (("counters", feats), ("counters + device + time", feats + ["_devc", "_ts"])):
        for split, (p_tr, p_te) in (("random 70/30", (a, b)), ("time, intervals whole", (ta, tb))):
            prm = {"objective": "multi:softmax", "num_class": len(names), "max_depth": 6, "eta": 0.3,
                   "tree_method": "hist", "nthread": 4, "seed": 42}
            m = xgb.train(prm, xgb.DMatrix(Xf.iloc[p_tr][cols], label=y[p_tr]), num_boost_round=100)
            p = m.predict(xgb.DMatrix(Xf.iloc[p_te][cols])).astype(int)
            per = f1_score(y[p_te], p, average=None, labels=range(len(names)), zero_division=0)
            print(f"  {sname:<26} {split:<22} macro-F1={per.mean():.3f}  "
                  + "  ".join(f"{k[:9]}={v:.2f}" for k, v in zip(names, per)))

    # all seven categories (the run that fails), then the label set unsw-iot-3: the
    # categories whose attack shows in the affected counters in EVERY annotated interval
    # (section (ii): visible_share 1.00 for ArpSpoof, PingOfDeath, Smurf; 0.58-0.86 for the
    # TCP and UDP floods). Rows of the other categories are removed from the stream.
    # (A per-device baseline normalisation of the counters was also tried on all seven
    # categories: it raised the FPR to 16.0% and is not used.)
    protocol(X, A, feats, is_tr, CATS, "all seven categories")
    keep = np.isin(X["_cat"].values, CATS3 + ["Normal"])
    protocol(X[keep].reset_index(drop=True), A[A["cat"].isin(CATS3)], feats, is_tr[keep], CATS3,
             "label set unsw-iot-3")


CATS3 = ["ArpSpoof", "PingOfDeath", "Smurf"]


def protocol(X, A, feats, is_tr, cats, variant):
    print(f"\n=== ReSIDS PROTOCOL — {variant} ({2 * len(cats)} specialists, time split with intervals whole) ===")
    Xa = X[feats].to_numpy(np.float32)
    cat = X["_cat"].values
    Xtr, ctr = Xa[is_tr], cat[is_tr]
    N = 2 * len(cats)
    ben = np.where(ctr == "Normal")[0]
    shares = np.array_split(np.random.default_rng(C.SEED).permutation(ben), N)
    rng2 = np.random.default_rng(C.SEED)
    halves = {c: np.array_split(rng2.permutation(np.where(ctr == c)[0]), 2) for c in cats}
    models = []
    for i in range(N):
        c = cats[i % len(cats)]
        pos, neg = halves[c][i // len(cats)], shares[i]
        rows = np.r_[pos, neg]
        prm = dict(C.XGB); prm["scale_pos_weight"] = len(neg) / max(len(pos), 1)
        models.append(xgb.train(prm, xgb.DMatrix(Xtr[rows], label=np.r_[np.ones(len(pos)), np.zeros(len(neg))]),
                                num_boost_round=10))
    Xte, cte = Xa[~is_tr], cat[~is_tr]
    V = np.column_stack([(m.predict(xgb.DMatrix(Xte)) >= 0.5).astype(np.int8) for m in models])
    atk = cte != "Normal"
    print("test:", pd.Series(cte).value_counts().to_dict())
    for k in (1, 2):
        g = C.stats(V.sum(1) >= k, atk)
        print(f"  GL (union of {N}) k>={k}: recall {g['recall']:.2f}%  FPR {g['FPR']:.4f}% (FP {g['FP']})  "
              f"F1 {g['F1']:.2f}  F1@13.75 {g['F1@13.75']:.2f}")
    f2 = V.sum(1) >= 2
    print("  per category minute recall (GL k>=2):", {c: round(100 * f2[cte == c].mean(), 1) for c in cats})
    te_ann = A.set_index("iid")
    iid = X["_iid"].values[~is_tr]
    rate = pd.Series(iid).map(te_ann["rate"]).values
    print("  minute recall by attack rate (pkt/s):",
          {int(r): round(100 * f2[atk & (rate == r)].mean(), 1) for r in (1, 10, 100) if (atk & (rate == r)).any()})
    # operational view: an attack interval (10 min) is detected if any of its minutes alarms
    hit = pd.Series(f2[atk]).groupby(iid[atk]).max()
    icat = pd.Series(iid[atk]).map(te_ann["cat"]).groupby(iid[atk]).first()
    print(f"  attack intervals detected (any minute flagged): {int(hit.sum())}/{len(hit)} "
          f"({100 * hit.mean():.1f}%)  per category "
          + str({c: f"{int(hit[icat == c].sum())}/{int((icat == c).sum())}" for c in cats}))
    ndev_days = (~atk).sum() / 1440
    print(f"  false alarms per device-day: {int((f2 & ~atk).sum()) / ndev_days:.2f} "
          f"({int((f2 & ~atk).sum())} alarmed normal minutes over {ndev_days:.0f} device-days)")
    for nn in sorted({len(cats), 3, 2}, reverse=True):
        v = V[:, :nn].sum(1) >= 2
        g = C.stats(v, atk)
        print(f"  FL k>=2 with {nn} nodes: recall {g['recall']:.1f}%  FPR {g['FPR']:.4f}%  per category "
              + str({c[:9]: round(100 * v[cte == c].mean(), 1) for c in cats}))


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/unsw_iot_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
