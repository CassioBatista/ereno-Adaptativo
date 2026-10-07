#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on the network part of
ToN_IoT (Moustafa et al., UNSW Canberra): processed Zeek flows, Network_dataset_1..23.csv.
Audited copy: Kaggle mirror mohammedaddoun/ton-iot (checksums in results/toniot_md5sums.txt).
The authors label by "tagging IP addresses (192.168.159.30-39) and their timestamps".

  (i)   STRUCTURE   fields per line, header per file, value kinds, unset values
  (ii)  LABELS      label vs type, duplicates, vectors with >1 type, the tagging rule
                    (attacker range in attack and normal flows), behaviour per type
  (iii) TIME        span per type and per file; normal flows inside each attack window
  (iv)  ATTRIBUTION source addresses per type
  (v)   PREVALENCE  flows per type
  (vi)  SHORTCUT    XGBoost with/without identifiers and session strings, random and
                    time-ordered splits

Memory: 22 M rows do not fit as a full frame in the 9 GB WSL VM (a first version was
OOM-killed). Pass 1 reads one file at a time and keeps only row hashes and a few light
columns; pass 2 re-reads the files to fetch the rows sampled for the probe.

  python scripts/inspect_toniot.py ~/datasets/toniot
Out: results/toniot_audit.txt
"""
import glob
import io
import ipaddress
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

IDENT = ["ts", "src_ip", "src_port", "dst_ip", "dst_port"]
NUM = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes",
       "dst_pkts", "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
       "http_trans_depth", "http_request_body_len", "http_response_body_len",
       "http_status_code"]
# low-cardinality protocol fields
CATF = ["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
        "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "http_method",
        "http_version", "http_orig_mime_types", "http_resp_mime_types", "weird_name",
        "weird_addl", "weird_notice"]
# free-text fields that can name a session or a host (queries, certificates, URIs, agents)
SESSION = ["dns_query", "ssl_subject", "ssl_issuer", "http_uri", "http_referrer",
           "http_user_agent"]
LIGHT = ["ts", "src_port", "dst_port", "src_ip", "dst_ip", "proto", "service",
         "conn_state", "type", "label"]
ATT_LO = int(ipaddress.ip_address("192.168.159.30"))
ATT_HI = int(ipaddress.ip_address("192.168.159.39"))
IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def ip_to_int(s):
    p = s.astype(str).str.split(".", expand=True)
    if p.shape[1] < 4:
        return np.full(len(s), -1, np.int64)
    q = p.iloc[:, :4].apply(pd.to_numeric, errors="coerce")
    return (q[0] * 2**24 + q[1] * 2**16 + q[2] * 2**8 + q[3]).fillna(-1).astype(np.int64).values


def read(p):
    with open(p, encoding="utf-8-sig") as f:
        cols = f.readline().rstrip("\n").split(",")
    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False, na_filter=False,
                     dtype={c: "category" for c in cols if c not in ("ts", "src_port", "dst_port", "label")})
    for c in NUM:                                      # "-" marks an unset Zeek field
        df[c] = pd.to_numeric(df[c].astype(str), errors="coerce").astype("float32")
    for c in ("ts", "src_port", "dst_port", "label"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main(d):
    files = sorted(glob.glob(os.path.join(d, "Network_dataset_*.csv")),
                   key=lambda p: int(re.findall(r"(\d+)\.csv", p)[0]))
    section(f"(i) STRUCTURE — {len(files)} files")
    heads, parts, hx, hf = set(), [], [], []
    numst = defaultdict(lambda: [0, np.inf, -np.inf])
    ipok = Counter()
    for k, p in enumerate(files):
        with open(p, encoding="utf-8-sig") as f:
            heads.add(f.readline())
        nf = subprocess.run(["awk", "-F,", "NR>1{c[NF]++} END{for(k in c) print k\":\"c[k]}", p],
                            capture_output=True, text=True).stdout.split()
        df = read(p)
        # hashes of whole rows and of the feature vector (values, comparable across files)
        hx.append(pd.util.hash_pandas_object(df, index=False).values)
        hf.append(pd.util.hash_pandas_object(df[NUM + CATF].fillna(-1), index=False).values)
        for c in NUM:
            s = numst[c]
            s[0] += int(df[c].isna().sum()); s[1] = min(s[1], df[c].min()); s[2] = max(s[2], df[c].max())
        for c in ("src_ip", "dst_ip"):
            cats = pd.Series(df[c].cat.categories.astype(str))
            ipok[c] += int(df[c].isin(cats[cats.str.match(IPV4)]).sum())
        lt = df[LIGHT].copy()
        lt["src_i"], lt["dst_i"] = ip_to_int(lt["src_ip"]), ip_to_int(lt["dst_ip"])
        lt["_file"] = np.int8(k + 1)
        lt["_row"] = np.arange(len(lt), dtype=np.int32)
        parts.append(lt)
        print(f"  {os.path.basename(p):<24} rows={len(df):>9,}  fields/line={nf}", flush=True)
        del df
    print("identical header in all files:", len(heads) == 1)
    from pandas.api.types import union_categoricals
    cols = {}
    for c in parts[0].columns:
        if isinstance(parts[0][c].dtype, pd.CategoricalDtype):
            cols[c] = union_categoricals([q[c] for q in parts])
        else:
            cols[c] = np.concatenate([q[c].values for q in parts])
        for q in parts:
            del q[c]
    del parts
    df = pd.DataFrame(cols)
    del cols
    hx, hf = np.concatenate(hx), np.concatenate(hf)
    n = len(df)
    print(f"total rows={n:,}")
    print("  ts: epoch seconds, non-numeric", int(df["ts"].isna().sum()), "; range",
          pd.to_datetime(df["ts"].min(), unit="s"), "..", pd.to_datetime(df["ts"].max(), unit="s"))
    for c in ("src_port", "dst_port"):
        print(f"  {c}: within 0..65535 {pct(int(df[c].between(0, 65535).sum()), n)}")
    for c in ("src_ip", "dst_ip"):
        print(f"  {c}: IPv4-shaped {pct(ipok[c], n)}; distinct {df[c].nunique():,}")
    for c in NUM:
        s = numst[c]
        print(f"  {c:<24} unset={s[0]:,} min={s[1]:g} max={s[2]:g}")
    for c in ("service", "conn_state", "proto"):
        print(f"  {c}: {df[c].value_counts().head(8).to_dict()}")

    section("(ii) LABELS")
    print("label x type:")
    print(pd.crosstab(df["type"], df["label"]).to_string())
    inr = lambda v: (v >= ATT_LO) & (v <= ATT_HI)
    df["_att_src"] = inr(df["src_i"].values)
    df["_att"] = df["_att_src"].values | inr(df["dst_i"].values)
    print("\nflows involving the tagged attacker range 192.168.159.30-39:")
    for t, g in df.groupby("type", observed=True):
        print(f"  {t:<12} source or destination {pct(int(g['_att'].sum()), len(g)):>8}   "
              f"as source {pct(int(g['_att_src'].sum()), len(g)):>8}")
    print(f"\nexact duplicate rows: {int(pd.Series(hx).duplicated().sum()):,}")
    dupf = pd.Series(hf).duplicated().values
    print(f"duplicates without identifiers and session strings: {int(dupf.sum()):,} ({pct(int(dupf.sum()), n)})")
    tcode = df["type"].cat.codes.values
    pairs = pd.DataFrame({"h": hf, "t": tcode}).drop_duplicates()
    multi = pairs["h"][pairs["h"].duplicated()].unique()
    rows = np.isin(hf, multi)
    print(f"vectors with >1 type: {len(multi):,} (rows {int(rows.sum()):,}, {pct(int(rows.sum()), n)}); "
          f"types involved: {df.loc[rows, 'type'].value_counts().head(10).to_dict()}")
    print("\nbehaviour per type:")
    for t, g in df.groupby("type", observed=True):
        top = lambda c, m=3: ", ".join(f"{k} {pct(v, len(g))}" for k, v in g[c].value_counts().head(m).items())
        print(f"  {t:<12} proto: {top('proto', 2)} | state: {top('conn_state')} | service: "
              f"{top('service', 2)} | dst port: {top('dst_port')}")
    print("\nunique vectors per type:", df.loc[~dupf, "type"].value_counts().to_dict())

    section("(iii) TIME")
    sp = df.groupby("type", observed=True)["ts"].agg(["min", "max", "count"])
    sp["from"] = pd.to_datetime(sp["min"], unit="s"); sp["to"] = pd.to_datetime(sp["max"], unit="s")
    print(sp[["from", "to", "count"]].to_string())
    nts = np.sort(df.loc[df["type"] == "normal", "ts"].values)
    for t, gg in df[["type", "ts"]].groupby("type", observed=True):
        if t == "normal":
            continue
        lo, hi = gg["ts"].min(), gg["ts"].max()
        inside = int(np.searchsorted(nts, hi, "right") - np.searchsorted(nts, lo, "left"))
        ats = np.unique(gg["ts"].values)
        i = np.searchsorted(ats, nts)
        near = np.zeros(len(nts), bool)
        for j in (i - 1, np.minimum(i, len(ats) - 1)):
            ok = j >= 0
            near[ok] |= np.abs(nts[ok] - ats[j[ok]]) <= 60
        print(f"  {t:<12} normal flows inside its span: {inside:,} (share {pct(inside, inside + len(gg))}); "
              f"within 60 s of one of its flows: {int(near.sum()):,}")
    print("\ntypes per file (% of rows):")
    print((pd.crosstab(df["_file"], df["type"], normalize="index") * 100).round(1).to_string())
    fs = df.groupby("_file")["ts"].agg(["min", "max"])
    print("files in time order:", bool((fs["min"].values[1:] >= fs["max"].values[:-1] - 1).all()))

    section("(iv) ATTRIBUTION")
    for t, g in df.groupby("type", observed=True):
        top = g["src_ip"].value_counts().head(3)
        print(f"  {t:<12} src={g['src_ip'].nunique():>7,} dst={g['dst_ip'].nunique():>7,}  top src: "
              + ", ".join(f"{i} ({pct(v, len(g))})" for i, v in top.items()))
    mj = pd.crosstab(df["src_ip"], df["type"]).max(axis=1).sum()
    print(f"source address alone -> type (majority per address): {pct(int(mj), n)}")

    section("(v) PREVALENCE")
    for k, v in df["type"].value_counts().items():
        print(f"  {k:<12}{v:>12,}  {pct(v, n)}")

    section("(vi) SHORTCUT PROBE — XGBoost macro-F1 over types (dedup; <= 100k rows per type)")
    rng = np.random.RandomState(42)
    ded = df.index[~dupf]
    idx = np.sort(np.concatenate([rng.choice(g, min(100_000, len(g)), replace=False)
                                  for _, g in pd.Series(ded).groupby(df.loc[ded, "type"].values)]))
    want = df.loc[idx, ["_file", "_row"]]
    light = df.loc[idx, ["ts", "src_i", "dst_i", "type"]].reset_index(drop=True)
    del df
    rows = []
    for k, p in enumerate(files):
        r = want.loc[want["_file"] == k + 1, "_row"].values
        if len(r):
            q = read(p).iloc[r]
            q = q[NUM + CATF + SESSION + ["src_port", "dst_port"]].copy()
            for c in CATF + SESSION:
                q[c] = q[c].astype(str)
            rows.append(q)
    Xp = pd.concat(rows, ignore_index=True)
    for c in CATF + SESSION:
        Xp[c] = pd.factorize(Xp[c])[0]
    Xp["ts"], Xp["src_ip"], Xp["dst_ip"] = light["ts"].values, light["src_i"].values, light["dst_i"].values
    y, names = pd.factorize(light["type"].astype(str), sort=True)
    feats = NUM + CATF
    sets = {"with identifiers and session strings": feats + SESSION + IDENT,
            "with session strings, no identifiers": feats + SESSION,
            "without identifiers and session strings": feats}
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    a, b = train_test_split(np.arange(len(Xp)), test_size=0.3, stratify=y, random_state=42)
    ta, tb = [], []
    for c in range(len(names)):
        ii = np.where(y == c)[0]
        ii = ii[np.argsort(light["ts"].values[ii], kind="stable")]
        cut = int(0.7 * len(ii)); ta += list(ii[:cut]); tb += list(ii[cut:])
    for split, (tr, te) in (("random 70/30", (a, b)),
                            ("time-ordered 70/30 per type", (np.array(ta), np.array(tb)))):
        print(f"-- {split}")
        for sname, cl in sets.items():
            prm = {"objective": "multi:softmax", "num_class": len(names), "max_depth": 6,
                   "eta": 0.3, "tree_method": "hist", "nthread": 4, "seed": 42}
            m = xgb.train(prm, xgb.DMatrix(Xp.iloc[tr][cl], label=y[tr]), num_boost_round=100)
            p = m.predict(xgb.DMatrix(Xp.iloc[te][cl])).astype(int)
            per = f1_score(y[te], p, average=None, labels=range(len(names)), zero_division=0)
            print(f"  {sname:<42} macro-F1={per.mean():.3f}  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, per)), flush=True)
            if split.startswith("random") and sname.startswith("without"):
                gsc = pd.Series(m.get_score(importance_type="gain"))
                print("    top gain:", list(gsc.sort_values(ascending=False).head(6).index))


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/toniot")
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    open("results/toniot_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
