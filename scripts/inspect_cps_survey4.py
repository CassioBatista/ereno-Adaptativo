#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on the four smart-grid and
ICS datasets shortlisted from the CPS-dataset survey of Quincozes et al. (SSRN 5247519):

  Morris-1   power-system attack datasets (Adhikari, Pan, Morris; MSU/ORNL), binary set,
             Kaggle mirror bachirbarika/power-system (the UAH links return 404)
  HAI 20.07  HAI 1.0 as corrected by its authors (GitHub icsdataset/hai)
  Westermo   network traffic data set (GitHub westermo/network-traffic-dataset), flows
  PowerDuck  GOOSE data set of cyberattacks in substations (Zenodo 6974112), IPAL files

Each dataset gets the six checks: (i) structure, (ii) labels, (iii) time, (iv) attribution,
(v) prevalence, (vi) shortcut probe. Reports: results/{morris1,hai2007,westermo,powerduck}_audit.txt

  python scripts/inspect_cps_survey4.py
"""
import csv
import glob
import gzip
import io
import json
import os
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

D = os.path.expanduser("~/datasets/")


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def probe(X, y, groups=None, order=None, label=""):
    """Macro-F1 of a gradient-boosted probe on a random split and, when given, on an
    independent split (by group) and on a time-ordered split (by order)."""
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    y = np.asarray(y)
    names = np.unique(y)
    yi = np.searchsorted(names, y)
    out = {}
    splits = {"random 70/30": train_test_split(np.arange(len(X)), test_size=0.3,
                                               stratify=yi, random_state=42)}
    if groups is not None:
        g = np.asarray(groups)
        ug = np.unique(g)
        test_g = ug[int(0.7 * len(ug)):]
        te = np.isin(g, test_g)
        splits[f"independent groups ({len(ug) - len(test_g)} train / {len(test_g)} test)"] = \
            (np.where(~te)[0], np.where(te)[0])
    if order is not None:
        o = np.asarray(order)
        ta, tb = [], []
        for c in range(len(names)):
            ii = np.where(yi == c)[0]
            ii = ii[np.argsort(o[ii], kind="stable")]
            cut = int(0.7 * len(ii))
            ta += list(ii[:cut]); tb += list(ii[cut:])
        splits["time-ordered 70/30 per class"] = (np.array(ta), np.array(tb))
    for sname, (a, b) in splits.items():
        m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist", n_jobs=4,
                              random_state=42)
        ytr = yi[a]
        present = np.unique(ytr)
        remap = {c: i for i, c in enumerate(present)}
        m.fit(X.iloc[a], np.array([remap[c] for c in ytr]))
        p = present[m.predict(X.iloc[b])]
        per = f1_score(yi[b], p, average=None, labels=range(len(names)), zero_division=0)
        print(f"  [{label}] {sname:<42} macro-F1={per.mean():.3f}  "
              + "  ".join(f"{n}={v:.2f}" for n, v in zip(names, per)))
        out[sname] = per.mean()
    return out


# ------------------------------------------------------------------------------- Morris-1
def morris():
    files = sorted(glob.glob(D + "morris_power/data*.csv"),
                   key=lambda p: int(os.path.basename(p)[4:-4]))
    section(f"(i) STRUCTURE — {len(files)} files (binary set; 15 sets x 37 scenarios, 1% random sample)")
    frames, heads = [], set()
    for p in files:
        with open(p, newline="") as f:
            r = csv.reader(f)
            heads.add(tuple(next(r)))
            fc = Counter(len(row) for row in r)
        df = pd.read_csv(p, low_memory=False)
        df["_set"] = int(os.path.basename(p)[4:-4])
        frames.append(df)
        print(f"  {os.path.basename(p):<12} rows={len(df):>6,} fields per line={dict(fc)}")
    print("same header in all files:", len(heads) == 1)
    df = pd.concat(frames, ignore_index=True)
    feats = [c for c in df.columns if c not in ("marker", "_set")]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    print(f"rows={len(df):,} features={len(feats)} NaN={int(X.isna().sum().sum())} "
          f"inf={int(np.isinf(X.values).sum())} (columns with inf: "
          f"{[c for c in feats if np.isinf(X[c]).any()][:8]})")
    pmu = [c for c in feats if c.startswith("R")]
    logs = [c for c in feats if not c.startswith("R")]
    print(f"PMU columns={len(pmu)} (magnitudes/angles of four relays); log columns={logs}")

    section("(ii) LABELS")
    print("marker:", df["marker"].value_counts().to_dict())
    h = pd.util.hash_pandas_object(X, index=False)
    print(f"feature duplicates: {int(h.duplicated().sum()):,} ({pct(int(h.duplicated().sum()), len(df))})")
    g = df.groupby(h.values)["marker"].nunique()
    print(f"vectors with >1 label: {int((g > 1).sum()):,}")

    section("(iii) TIME")
    print("no timestamp column; the release states that each set is a 1% random sample of the "
          "scenario recordings, so the time order of the measurements is not available")
    print("label runs in file order (set 1):",
          int(frames[0]["marker"].ne(frames[0]["marker"].shift()).sum()), "of", len(frames[0]))

    section("(iv) ATTRIBUTION")
    print("no address columns; relay and Snort logs are per-relay flags:",
          {c: int((X[c] != 0).sum()) for c in logs})

    section("(v) PREVALENCE")
    for k, v in df["marker"].value_counts().items():
        print(f"  {k:<10}{v:>8,}  {pct(v, len(df))}")

    section("(vi) SHORTCUT PROBE (binary marker)")
    Xc = X.replace([np.inf, -np.inf], np.nan).fillna(-1)
    for name, cols in (("PMU + logs", feats), ("PMU only (physical)", pmu), ("logs only", logs)):
        probe(Xc[cols], df["marker"].values, groups=df["_set"].values, label=name)


# ------------------------------------------------------------------------------- HAI 20.07
def hai():
    section("(i) STRUCTURE")
    fr = {}
    for f in ("train1", "train2", "test1", "test2"):
        df = pd.read_csv(D + f"hai_2007/{f}.csv.gz", sep=None, engine="python")
        df["_file"] = f
        df["_t"] = pd.to_datetime(df["time"], errors="coerce")
        gaps = df["_t"].diff().dt.total_seconds()
        fr[f] = df
        print(f"  {f}: rows={len(df):,} cols={df.shape[1] - 2} NaN={int(df.isna().sum().sum())} "
              f"time parsed={pct(int(df['_t'].notna().sum()), len(df))} "
              f"{df['_t'].min()} .. {df['_t'].max()}  step(s)={gaps.value_counts().head(3).to_dict()}")
    df = pd.concat(fr.values(), ignore_index=True)
    labs = ["attack", "attack_P1", "attack_P2", "attack_P3"]
    sens = [c for c in df.columns if c not in labs + ["time", "_file", "_t"]]
    print(f"process variables={len(sens)}; label columns={labs}")

    section("(ii) LABELS")
    for f, g in df.groupby("_file"):
        a = g["attack"].values
        runs = int(((a[1:] == 1) & (a[:-1] == 0)).sum() + (a[0] == 1))
        anyp = (g[labs[1:]].sum(axis=1) > 0).astype(int)
        print(f"  {f}: attack rows={int(a.sum()):,} ({pct(int(a.sum()), len(g))}); attack segments={runs}; "
              f"attack == any(P1..P3): {pct(int((anyp == a).sum()), len(g))}; "
              f"per process: {{{', '.join(f'{c}: {int(g[c].sum())}' for c in labs[1:])}}}")
    print("labels name the attacked process, not the attack type: no per-attack label set")
    h = pd.util.hash_pandas_object(df[sens], index=False)
    print(f"duplicates of process vectors (time removed): {int(h.duplicated().sum()):,} "
          f"({pct(int(h.duplicated().sum()), len(df))})")
    g = df.groupby(h.values)["attack"].nunique()
    print(f"vectors with >1 label: {int((g > 1).sum()):,}")

    section("(iii) TIME")
    for f in ("test1", "test2"):
        g = fr[f]
        a = g["attack"].values
        print(f"  {f}: attacks inside continuous operation; normal rows between first and last "
              f"attack: {int(((g['_t'] >= g.loc[a == 1, '_t'].min()) & (g['_t'] <= g.loc[a == 1, '_t'].max()) & (a == 0)).sum()):,}")
    print("  train files are normal-only (attack rows:",
          int(fr['train1']['attack'].sum() + fr['train2']['attack'].sum()), ")")

    section("(iv) ATTRIBUTION")
    print("no network traffic; the per-process labels locate the attacked subsystem (P1..P3)")

    section("(v) PREVALENCE")
    test = pd.concat([fr["test1"], fr["test2"]])
    print(f"  test: attack {pct(int(test['attack'].sum()), len(test))} of {len(test):,} s; "
          f"all files: attack {pct(int(df['attack'].sum()), len(df))}; "
          f"normal operation available: {int((df['attack'] == 0).sum()) / 3600:.1f} h")

    section("(vi) SHORTCUT PROBE (attack vs normal, test files; time excluded)")
    X = test[sens].reset_index(drop=True)
    probe(X, test["attack"].values, groups=test["_file"].values,
          order=test["_t"].values, label="process variables")


# ------------------------------------------------------------------------------- Westermo
def westermo():
    labs = ["IT_B_Label", "IT_M_Label", "NST_B_Label", "NST_M_Label"]
    ident = ["sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs", "startDate", "endDate",
             "start", "end", "startOffset", "endOffset"]
    for v in ("extended", "reduced"):
        section(f"(i) STRUCTURE — {v}")
        fr = []
        for s in ("bottom", "left", "right"):
            p = D + f"westermo/{v}/flows/output_{s}.csv"
            with open(p, newline="") as f:
                r = csv.reader(f)
                head = next(r)
                fc = Counter(len(row) for row in r)
            df = pd.read_csv(p, low_memory=False)
            df["_point"] = s
            fr.append(df)
            print(f"  {s:<7} flows={len(df):>7,} fields per line={dict(fc)}")
        df = pd.concat(fr, ignore_index=True)
        feats = [c for c in df.columns if c not in labs + ident + ["_point", "protocol"]]
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        print(f"flows={len(df):,} features={len(feats)} NaN={int(X.isna().sum().sum())} "
              f"inf={int(np.isinf(X.values).sum())}; physical (process) columns: none in the flow files")
        if v == "reduced":
            print("labels:", {c: df[c].value_counts().head(8).to_dict() for c in labs})
            continue

        section("(ii) LABELS (extended)")
        for c in labs:
            print(f"  {c}: {df[c].value_counts().to_dict()}")
        it_only = (df["IT_B_Label"] != df["IT_B_Label"].iloc[0]) if False else None
        itb, nstb = df["IT_B_Label"].astype(str), df["NST_B_Label"].astype(str)
        norm = itb.value_counts().index[0]
        mis = (itb != norm) & (nstb == norm)
        print(f"  flows malicious by timing (IT) but not involving the attacker (NST): "
              f"{int(mis.sum()):,} of {int((itb != norm).sum()):,} IT-malicious "
              f"({pct(int(mis.sum()), int((itb != norm).sum()))})")
        h = pd.util.hash_pandas_object(X, index=False)
        print(f"  feature duplicates: {int(h.duplicated().sum()):,} ({pct(int(h.duplicated().sum()), len(df))})")
        for c in ("IT_M_Label", "NST_M_Label"):
            g = df.groupby(h.values)[c].nunique()
            print(f"  vectors with >1 {c}: {int((g > 1).sum()):,}")

        section("(iii) TIME")
        t = pd.to_numeric(df["start"], errors="coerce")
        print(f"  capture span {t.max() - t.min():.0f} s; one continuous recording with scheduled events")
        ev = open(D + "westermo/events.txt").read().splitlines()
        starts = [e for e in ev if "-START]" in e and "[BAD" in e]
        print(f"  bad-event starts in events.txt: {len(starts)}; kinds: "
              f"{dict(Counter(e.split('[')[2].split(']')[0].replace('-START', '') for e in starts))}")
        nst = df["NST_M_Label"].astype(str)
        nnorm = nst.value_counts().index[0]
        for k in nst.unique():
            if k == nnorm:
                continue
            tk = t[nst == k]
            inside = (t >= tk.min()) & (t <= tk.max())
            print(f"  {k:<26} flows={int((nst == k).sum()):>6,}  benign share inside its span "
                  f"{pct(int((inside & (nst == nnorm)).sum()), int(inside.sum()))}")

        section("(iv) ATTRIBUTION")
        for k in nst.unique():
            s = df.loc[nst == k, "sIPs"].astype(str).value_counts().head(3)
            print(f"  {k:<26} top source IPs: " + ", ".join(f"{i} ({c})" for i, c in s.items()))

        section("(v) PREVALENCE")
        for c in ("IT_B_Label", "NST_B_Label"):
            print(f"  {c}: {({k: pct(v, len(df)) for k, v in df[c].value_counts().items()})}")

        section("(vi) SHORTCUT PROBE (NST multiclass)")
        Xc = X.replace([np.inf, -np.inf], np.nan).fillna(-1)
        keep = nst.map(nst.value_counts()) >= 10
        probe(Xc[keep].reset_index(drop=True), nst[keep].values, order=t[keep].values,
              label="flow features, identifiers excluded")
        Xi = Xc.assign(sttl_id=Xc.get("sttl", 0))
        Xid = pd.concat([Xc, df[["sAddress", "rAddress"]].astype("category").apply(lambda s: s.cat.codes)], axis=1)
        probe(Xid[keep].reset_index(drop=True), nst[keep].values,
              label="with source/destination address codes")


# ------------------------------------------------------------------------------- PowerDuck
def powerduck():
    base = D + "powerduck/dataset/"
    section("(i) STRUCTURE")
    rows, keysets, bad = [], Counter(), 0
    for p in sorted(glob.glob(base + "ipal/*/*.ipal.gz")):
        kind = p.split("/")[-2]
        scen = os.path.basename(p).replace(".ipal.gz", "")
        with gzip.open(p, "rt") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                dat = j.get("data") or {}
                keysets[len(dat)] += 1
                rows.append({"scenario": scen, "kind": kind, "id": j["id"], "t": j["timestamp"],
                             "malicious": bool(j["malicious"]), "src": j["src"], "dst": j["dest"],
                             "length": j["length"], "stNum": dat.get("stNum"), "sqNum": dat.get("sqNum"),
                             "n_values": len(dat)})
    df = pd.DataFrame(rows)
    print(f"packets={len(df):,} in {df['scenario'].nunique()} recordings "
          f"({(df.groupby('scenario')['kind'].first() == 'Normal').sum()} normal); unparsable lines={bad}")
    print(f"'data' field sizes (fields per packet): {dict(sorted(keysets.items()))}")
    mono = df.groupby("scenario")["t"].apply(lambda s: s.is_monotonic_increasing).mean()
    print(f"timestamps monotonic within recording: {pct(int(mono * df['scenario'].nunique()), df['scenario'].nunique())}")

    section("(ii) LABELS")
    mal_ids = {}
    for p in glob.glob(base + "malicious/*.json.gz"):
        with gzip.open(p, "rt") as f:
            js = json.load(f)
        mal_ids[os.path.basename(p).replace(".json.gz", "")] = {e["ipalid"] for e in js}
    agree, tot = 0, 0
    for scen, g in df[df["kind"] == "Attack"].groupby("scenario"):
        ids = mal_ids.get(scen, set())
        agree += int((g["id"].isin(ids) == g["malicious"]).sum())
        tot += len(g)
    print(f"'malicious' field agrees with malicious/*.json ids: {pct(agree, tot)}")
    print("attack packets per recording:")
    for scen, g in df[df["kind"] == "Attack"].groupby("scenario"):
        print(f"  {scen:<38} packets={len(g):>8,} malicious={int(g['malicious'].sum()):>8,} "
              f"({pct(int(g['malicious'].sum()), len(g))})")
    h = pd.util.hash_pandas_object(df[["length", "stNum", "sqNum", "src", "dst"]].astype(str), index=False)
    print(f"duplicates on (src,dst,length,stNum,sqNum): {int(h.duplicated().sum()):,}")

    section("(iii) TIME")
    for scen, g in df[df["kind"] == "Attack"].groupby("scenario"):
        m = g[g["malicious"]]
        if len(m):
            inside = g[(g["t"] >= m["t"].min()) & (g["t"] <= m["t"].max())]
            print(f"  {scen:<38} recording {g['t'].max() - g['t'].min():6.0f} s; benign share inside "
                  f"attack span {pct(int((~inside['malicious']).sum()), len(inside))}")

    section("(iv) ATTRIBUTION")
    legit = set(df.loc[df["kind"] == "Normal", "src"])
    m = df[df["malicious"]]
    print(f"publishers in normal recordings: {sorted(legit)}")
    print(f"malicious packets whose source MAC is a legitimate publisher: "
          f"{pct(int(m['src'].isin(legit).sum()), len(m))}; sources: {m['src'].value_counts().head(4).to_dict()}")

    section("(v) PREVALENCE")
    print(f"  all packets: malicious {pct(int(df['malicious'].sum()), len(df))} of {len(df):,}")
    nf = df[~df["scenario"].str.contains("flood")]
    print(f"  without the two flooding recordings: malicious {pct(int(nf['malicious'].sum()), len(nf))} of {len(nf):,}")
    print(f"  benign packets in normal recordings: {int((df['kind'] == 'Normal').sum()):,}")

    section("(vi) SHORTCUT PROBE (malicious vs benign, per packet)")
    df = df.sort_values(["scenario", "t"])
    g = df.groupby(["scenario", "src"])
    df["dt"] = g["t"].diff().fillna(-1)
    df["dSt"] = g["stNum"].diff().fillna(0)
    df["dSq"] = g["sqNum"].diff().fillna(0)
    X = df[["length", "n_values", "dt", "dSt", "dSq"]].astype(float).fillna(-1).reset_index(drop=True)
    y = np.where(df["malicious"].values, "malicious", "benign")
    probe(X, y, groups=df["scenario"].values, label="behavioural (no ids, no absolute time)")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    for name, fn in (("morris1", morris), ("hai2007", hai), ("westermo", westermo),
                     ("powerduck", powerduck)):
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                fn()
            except Exception as e:
                import traceback
                traceback.print_exc(file=buf)
        open(f"results/{name}_audit.txt", "w").write(buf.getvalue())
        print(f"\n################ {name} ################")
        print(buf.getvalue())
