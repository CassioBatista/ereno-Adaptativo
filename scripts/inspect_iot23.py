#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on IoT-23 (Garcia, Parmisano
and Erquiaga, Stratosphere Laboratory, CTU Prague, 2020): 20 malware and 3 benign captures
from IoT devices, Zeek conn.log flows labelled per flow (bro/conn.log.labeled).
Audited copy: iot_23_datasets_small.tar.gz from mcfp.felk.cvut.cz (labelled flows only),
extracted under ~/datasets/iot23/.

  (i)   STRUCTURE   fields per line, header per file, timestamps, unset values
  (ii)  LABELS      label/detailed-label pairs, behaviour per label (protocol, state, port),
                    duplicates and vectors with >1 category (on the sample)
  (iii) TIME        span per capture; benign flows inside the malicious window
  (iv)  ATTRIBUTION source address of malicious flows per capture
  (v)   PREVALENCE  flows per category and capture
  (vi)  SHORTCUT    XGBoost with/without identifiers: random, time-ordered and
                    leave-capture-out splits

The files are read once, in chunks; statistics are exact, and checks that need rows
together (duplicates, interleaving, probe) use a uniform sample of up to K flows per
capture and detailed label (bottom-k by random key).

  python scripts/inspect_iot23.py ~/datasets/iot23
Out: results/iot23_audit.txt, results/iot23_sample.pkl.gz
"""
import glob
import io
import os
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from multiprocessing import Pool

import numpy as np
import pandas as pd

K = 20_000
CHUNK = 1_000_000
COLS = ["ts", "uid", "orig_h", "orig_p", "resp_h", "resp_p", "proto", "service", "duration",
        "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp", "missed_bytes",
        "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tail"]
NUM = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes",
       "resp_pkts", "resp_ip_bytes"]
CATF = ["proto", "service", "conn_state", "history"]
IDENT = ["ts", "orig_h", "orig_p", "resp_h", "resp_p"]


def category(d):
    if d in ("-", "Benign", "benign"):
        return "Benign"
    if d.startswith("C&C"):
        return "C&C"
    if d.startswith("Okiru"):
        return "Okiru"
    if "PortScan" in d:
        return "PortScan"
    return d


def capture(path):
    return path.split(os.sep)[-3]


def scan(path):
    cap = capture(path)
    nf = subprocess.run(["awk", "-F\t", "!/^#/{c[NF]++} END{for(k in c) print k, c[k]}", path],
                        capture_output=True, text=True).stdout.split()
    with open(path) as f:
        head = [next(f) for _ in range(8)]
    g = defaultdict(lambda: {"n": 0, "tmin": np.inf, "tmax": -np.inf, "label": Counter(),
                             "orig": Counter(), "state": Counter(), "proto": Counter(),
                             "port": Counter(), "service": Counter(), "unset": Counter()})
    rng = np.random.default_rng(zlib.crc32(cap.encode()))
    keep, bad_tail, bad_ts = None, 0, 0
    for ch in pd.read_csv(path, sep="\t", names=COLS, comment="#", dtype=str, chunksize=CHUNK,
                          na_filter=False, engine="c"):
        t = ch.pop("tail").str.split(expand=True)
        bad_tail += int(t.iloc[:, :3].isna().any(axis=1).sum())
        if t.shape[1] > 3:
            bad_tail += int(t[3].notna().sum())
        ch["label"], ch["dlabel"] = t[1], t[2]
        ts = pd.to_numeric(ch["ts"], errors="coerce")
        bad_ts += int(ts.isna().sum())
        ch["ts"] = ts
        for d, s in ch.groupby("dlabel", sort=False):
            r = g[d]
            r["n"] += len(s)
            r["tmin"] = min(r["tmin"], s["ts"].min()); r["tmax"] = max(r["tmax"], s["ts"].max())
            r["label"].update(s["label"].value_counts().to_dict())
            r["orig"].update(s["orig_h"].value_counts().head(50).to_dict())
            r["state"].update(s["conn_state"].value_counts().to_dict())
            r["proto"].update(s["proto"].value_counts().to_dict())
            r["port"].update(s["resp_p"].value_counts().head(50).to_dict())
            r["service"].update(s["service"].value_counts().to_dict())
            for c in ("duration", "orig_bytes"):
                r["unset"][c] += int((s[c] == "-").sum())
        ch["_r"] = rng.random(len(ch))
        keep = ch if keep is None else pd.concat([keep, ch])
        keep = keep.sort_values("_r").groupby("dlabel", sort=False).head(K)
    keep["_cap"] = cap
    return {"cap": cap, "nf": dict(zip(nf[::2], nf[1::2])), "head": head[:7],
            "groups": {k: dict(v) for k, v in g.items()}, "bad_tail": bad_tail,
            "bad_ts": bad_ts, "sample": keep.drop(columns="_r")}


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main(d, res):
    res.sort(key=lambda r: int(r["cap"].split("-")[-2]) + (1000 if "Honeypot" in r["cap"] else 0))
    section(f"(i) STRUCTURE — {len(res)} captures")
    heads = Counter(tuple(r["head"][6:7]) for r in res)
    print("identical #fields header:", len(heads) == 1)
    tot = 0
    for r in res:
        n = sum(v["n"] for v in r["groups"].values())
        tot += n
        unset = sum(v["unset"]["duration"] for v in r["groups"].values())
        print(f"  {r['cap']:<32} flows={n:>11,}  fields/line={r['nf']}  bad tail={r['bad_tail']}  "
              f"bad ts={r['bad_ts']}  duration unset={pct(unset, n)}")
    print(f"total flows={tot:,}")

    section("(ii) LABELS — exact counts")
    agg = defaultdict(lambda: {"n": 0, "caps": set(), "label": Counter(), "state": Counter(),
                               "proto": Counter(), "port": Counter(), "service": Counter()})
    for r in res:
        for dl, v in r["groups"].items():
            a = agg[dl]
            a["n"] += v["n"]; a["caps"].add(r["cap"])
            for k in ("label", "state", "proto", "port", "service"):
                a[k].update(v[k])
    for dl, a in sorted(agg.items(), key=lambda x: -x[1]["n"]):
        top = lambda c, m=3: ", ".join(f"{k} {pct(v, a['n'])}" for k, v in c.most_common(m))
        print(f"  {dl:<36} cat={category(dl):<12} flows={a['n']:>11,}  captures={len(a['caps']):>2}  "
              f"label={dict(a['label'])}")
        print(f"      proto: {top(a['proto'])} | state: {top(a['state'])} | resp port: "
              f"{top(a['port'])} | service: {top(a['service'], 2)}")

    s = pd.concat([r["sample"] for r in res], ignore_index=True)
    s["_cat"] = s["dlabel"].map(category)
    for c in NUM:
        s[c] = pd.to_numeric(s[c].replace("-", np.nan), errors="coerce")
    s.to_pickle("results/iot23_sample.pkl.gz")
    feats = NUM + CATF
    X = s[NUM].copy()
    for c in CATF:
        X[c] = pd.factorize(s[c])[0]
    h = pd.util.hash_pandas_object(X.fillna(-1), index=False)
    print(f"\nsample: {len(s):,} flows (<= {K:,} per capture and detailed label)")
    print(f"duplicates without identifiers (in sample): {pct(int(h.duplicated().sum()), len(s))}")
    g = s.groupby(h.values)["_cat"].nunique()
    multi = g[g > 1]
    rows = h.isin(multi.index).values
    print(f"vectors with >1 category: {len(multi):,} (rows {int(rows.sum()):,}, {pct(int(rows.sum()), len(s))}); "
          f"categories involved: {s.loc[rows, '_cat'].value_counts().to_dict()}")
    top = s.loc[rows].groupby(h[rows].values)["_cat"].agg(lambda x: tuple(sorted(set(x)))).value_counts().head(5)
    print("  most frequent conflicts:", top.to_dict())

    section("(iii) TIME")
    s["_ts"] = pd.to_datetime(s["ts"], unit="s")
    for r in res:
        gr = r["groups"]
        tmin = min(v["tmin"] for v in gr.values()); tmax = max(v["tmax"] for v in gr.values())
        mal = [v for k, v in gr.items() if category(k) != "Benign"]
        line = (f"  {r['cap']:<32} {pd.to_datetime(tmin, unit='s'):%Y-%m-%d %H:%M} .. "
                f"{pd.to_datetime(tmax, unit='s'):%Y-%m-%d %H:%M} ({(tmax - tmin) / 3600:.1f} h)")
        if mal:
            lo, hi = min(v["tmin"] for v in mal), max(v["tmax"] for v in mal)
            b = s[(s["_cap"] == r["cap"]) & (s["_cat"] == "Benign")]
            ins = ((b["ts"] >= lo) & (b["ts"] <= hi)).mean() if len(b) else float("nan")
            line += f"; benign in sample {len(b):,}, inside malicious window {100 * ins:.1f}%"
        print(line)

    section("(iv) ATTRIBUTION — source of malicious flows")
    for r in res:
        o = Counter()
        nmal = 0
        for k, v in r["groups"].items():
            if category(k) != "Benign":
                o.update(v["orig"]); nmal += v["n"]
        if nmal:
            ip, c = o.most_common(1)[0]
            bg = r["groups"].get("-", {})
            bo = Counter(bg.get("orig", {}))
            st = Counter(bg.get("state", {}))
            print(f"  {r['cap']:<32} malicious={nmal:>11,}  top source {ip} ({pct(c, nmal)}); "
                  f"that host's share of benign flows: {pct(bo.get(ip, 0), sum(bo.values()))}; "
                  f"benign flows unanswered (S0): {pct(st.get('S0', 0), bg.get('n', 0))}")

    section("(v) PREVALENCE")
    cats = Counter()
    for a_dl, a in agg.items():
        cats[category(a_dl)] += a["n"]
    for k, v in cats.most_common():
        print(f"  {k:<14}{v:>12,}  {pct(v, tot)}")
    for r in res:
        n = sum(v["n"] for v in r["groups"].values())
        b = sum(v["n"] for k, v in r["groups"].items() if category(k) == "Benign")
        print(f"  {r['cap']:<32} benign {b:>10,} ({pct(b, n)})")

    section("(vi) SHORTCUT PROBE — XGBoost macro-F1 over categories (sample, dedup per capture)")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    keep = ~pd.DataFrame({"h": h.values, "c": s["_cap"]}).duplicated().values
    t = s.loc[keep].reset_index(drop=True)
    Xf = X.loc[keep].reset_index(drop=True)
    for c in ("orig_p", "resp_p"):
        Xf[c] = pd.to_numeric(t[c], errors="coerce")
    for c in ("orig_h", "resp_h"):
        Xf[c] = pd.factorize(t[c])[0]
    y, names = pd.factorize(t["_cat"], sort=True)
    print("rows per category:", t["_cat"].value_counts().to_dict())
    sets = {"with identifiers (hosts, ports)": feats + ["orig_h", "resp_h", "orig_p", "resp_p"],
            "with ports only": feats + ["orig_p", "resp_p"],
            "without identifiers": feats}
    a, b = train_test_split(np.arange(len(t)), test_size=0.3, stratify=y, random_state=42)
    ta, tb = [], []
    for c in range(len(names)):
        ii = np.where(y == c)[0]
        ii = ii[np.argsort(t["ts"].values[ii], kind="stable")]
        cut = int(0.7 * len(ii)); ta += list(ii[:cut]); tb += list(ii[cut:])
    caps = t["_cap"].unique()
    fold = {c: i % 5 for i, c in enumerate(sorted(caps))}
    fo = t["_cap"].map(fold).values

    def fit_pred(tr, te, cols):
        # xgb.train with a fixed class count: a leave-capture-out fold may lack a category
        prm = {"objective": "multi:softmax", "num_class": len(names), "max_depth": 6,
               "eta": 0.3, "tree_method": "hist", "nthread": 4, "seed": 42}
        m = xgb.train(prm, xgb.DMatrix(Xf.iloc[tr][cols], label=y[tr]), num_boost_round=100)
        return m.predict(xgb.DMatrix(Xf.iloc[te][cols])).astype(int), m

    for sname, cols in sets.items():
        print(f"-- {sname}")
        for split, (tr, te) in (("random 70/30", (a, b)),
                                ("time-ordered 70/30 per category", (np.array(ta), np.array(tb)))):
            p, m = fit_pred(tr, te, cols)
            per = f1_score(y[te], p, average=None, labels=range(len(names)), zero_division=0)
            print(f"  {split:<34} macro-F1={per.mean():.3f}  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, per)))
            if sname == "without identifiers" and split.startswith("random"):
                gsc = pd.Series(m.get_score(importance_type="gain"))
                print("    top gain:", list(gsc.sort_values(ascending=False).head(6).index))
        p = np.empty(len(t), dtype=int)
        for k in range(5):
            te = np.where(fo == k)[0]; tr = np.where(fo != k)[0]
            if len(te):
                p[te] = fit_pred(tr, te, cols)[0] if len(tr) else -1
        ok = p >= 0
        per = f1_score(y[ok], p[ok], average=None, labels=range(len(names)), zero_division=0)
        print(f"  {'leave-capture-out (5 folds)':<34} macro-F1={per.mean():.3f}  "
              + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, per)))
        bi = list(names).index("Benign")
        mb = f1_score(y[ok] != bi, p[ok] != bi)
        print(f"    malicious-vs-benign F1 across captures: {mb:.3f}")
    print("captures per category:",
          t.groupby("_cat")["_cap"].nunique().to_dict())


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/iot23")
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    files = sorted(glob.glob(os.path.join(d, "**", "bro", "conn.log.labeled"), recursive=True))
    with Pool(int(os.environ.get("IOT23_PROCS", 2))) as pool:
        res = []
        for r in pool.imap_unordered(scan, sorted(files, key=os.path.getsize, reverse=True)):
            print(f"scanned {r['cap']}", file=sys.stderr, flush=True)
            res.append(r)
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d, res)
    open("results/iot23_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
