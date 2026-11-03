#!/usr/bin/env python3
"""Dataset-audit protocol: the same six checks and the same shortcut probe for every release.

A loader in audit_loaders.py turns a release into a Release: one table of rows plus the
role of each column. The checks below never look at a dataset by name; only the loader
knows the release. Dataset-specific verifications (label against documentation, column
shifts, capture checksums) stay in the inspect_*.py scripts and are reported as
follow-ups, not as part of this protocol.

  (i)   STRUCTURE    fields per line in every data file, header lines repeated inside the
                     data, value kind of each column against its name
  (ii)  LABELS       exact duplicates, duplicates without identifiers, feature vectors
                     carrying more than one label (per label: share of rows in such vectors
                     and share whose vector also occurs as benign), identical files filed
                     under different labels
  (iii) TIME         timestamp present and complete, order in the file, span of each label
                     and benign share inside it (classes interleaved or in separate sessions)
  (iv)  ATTRIBUTION  source address present; top sources per label; attack rows whose source
                     also emits benign rows; source alone -> label, against the majority class
  (v)   PREVALENCE   label shares, benign volume
  (vi)  PROBE        XGBoost (100 trees, depth 6, learning rate 0.3, seed 42) on rows
                     deduplicated per label, at most 100,000 per label, labels with at least
                     10 rows; content features, then content + identifiers; random 70/30
                     stratified split, time-ordered 70/30 per label (when there is a clock),
                     and independent data (the release's own test file, or held-out groups:
                     files, captures, recordings)

  python scripts/audit_protocol.py <dataset> [<dataset> ...]     (names: --list)
Out: results/audit/<dataset>.txt and results/audit/<dataset>.json
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# learning rate 0.1: at 0.3 the multi-class booster diverges on Edge-IIoTset (macro-F1 0.02,
# against 0.63 for a single decision tree on the same rows)
PROBE = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "tree_method": "hist",
         "n_jobs": 4, "random_state": 42}
CAP = 100_000          # rows per label in the probe
MIN_ROWS = 10          # labels with fewer rows are left out of the probe
SEED = 42


@dataclass
class Release:
    """One release as the checks see it.

    df     rows; column '_y' is the label used for decisions (category level), '_t' the
           time in seconds (optional), '_src' the source address (optional), '_g' the
           group for an independent split (optional), '_part' 'train'/'test' when the
           release ships its own test file (optional)
    feats  content features (identifiers removed)
    ids    identifier columns: time, addresses, ports, sequence and frame numbers,
           checksums, session ids; numeric or categorical
    """
    name: str
    df: pd.DataFrame
    feats: list
    ids: list
    benign: str
    files: list = field(default_factory=list)          # data files for the structure check
    fmt: str = "csv"                                    # csv | tsv | arff | jsonl-gz | none
    header: bool = True                                 # data files start with a header line
    quoted: bool = True                                 # fields may be quoted (parse as CSV)
    time_unit: str = "s"                                # 's' (seconds) or 'packet' (order only)
    labelled_files: list = field(default_factory=list)  # (path, label) for the file check
    sample: str = "all rows"                            # how the rows were selected
    notes: list = field(default_factory=list)
    attribution: pd.DataFrame = None                    # separate (_y, _src) table, optional


# ----------------------------------------------------------------------------- helpers
def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def kind(s):
    """Dominant kind of the non-zero values of a column."""
    v = s[s.notna()]
    v = v[v.astype(str).str.strip().ne("0")]
    if v.empty:
        return "all-zero"
    x = pd.to_numeric(v, errors="coerce")
    if x.isna().mean() > 0.5:
        sv = v.astype(str)
        if sv.str.match(r"^\d+\.\d+\.\d+\.\d+$").mean() > 0.5:
            return "IPv4"
        if sv.str.match(r"^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$").mean() > 0.5:
            return "MAC"
        if pd.to_datetime(sv.head(200), errors="coerce", format="mixed").notna().mean() > 0.8:
            return "datetime"
        if sv.str.match(r"^0x[0-9a-fA-F]+$").mean() > 0.5:
            return "hex"
        return "text"
    x = x.dropna()
    med = x.median()
    if 1.2e9 < med < 2.0e9:
        return "epoch-s"
    if 1.2e12 < med < 2.0e12:
        return "epoch-ms"
    if x.between(2400, 2500).mean() > 0.8 or x.between(5150, 5900).mean() > 0.8:
        return "wifi-freq-MHz"
    if x.between(-100, -10).mean() > 0.8:
        return "dBm"
    return "integer" if (x == x.round()).all() else "real"


# what a column name promises, and the kinds that keep the promise
EXPECT = [
    (r"^(src|dst|scr|des|s|d|r)[ _]?ips?$|^ip\.(src|dst)(_host)?$|^(orig|resp)_h$|_ip$",
     {"IPv4", "all-zero", "text"}),
    (r"macs?$", {"MAC", "all-zero", "hex", "text"}),
    (r"port", {"integer", "all-zero"}),
    (r"time_?stamp|^ts$|epoch|^date|frame\.time$", {"epoch-s", "epoch-ms", "datetime", "real", "integer"}),
    (r"freq", {"wifi-freq-MHz", "all-zero", "integer", "real"}),
    (r"dbm", {"dBm", "all-zero"}),
]


def field_counts(p, rel):
    """Fields per data line and header lines repeated inside the data. Quoted CSV is parsed
    with the csv module; unquoted CSV, Zeek TSV and ARFF are counted with awk (fast on the
    multi-gigabyte releases). ARFF data starts after @data; Zeek logs carry '#' lines."""
    if p.startswith("zip::") or (rel.fmt == "csv" and rel.quoted):
        import csv
        csv.field_size_limit(10**9)
        fc, rep = Counter(), 0
        if p.startswith("zip::"):
            import zipfile
            _, zp, member = p.split("::")
            f = io.TextIOWrapper(zipfile.ZipFile(zp).open(member), encoding="latin-1", newline="")
        else:
            f = open(p, newline="", encoding="latin-1")
        with f:
            r = csv.reader(f)
            head = next(r) if rel.header else None
            for row in r:
                if not row:
                    continue
                fc[len(row)] += 1
                rep += row == head
        return sorted(fc.items(), key=lambda kv: -kv[1]), rep
    sep = "\t" if rel.fmt == "tsv" else ","
    if rel.fmt == "arff":
        prog = ("BEGIN{d=0} /^@[dD][aA][tT][aA]/{d=1; next} !d || /^%/ || NF==0 {next} "
                "{c[NF]++} END{for(k in c) print k\":\"c[k]}")
    else:
        hdr = "NR==1{h=$0; next} $0==h{r++} " if rel.header else ""
        prog = (("/^#/{next} " if rel.fmt == "tsv" else "") + "NF==0{next} " + hdr +
                "{c[NF]++} END{for(k in c) print k\":\"c[k]; print \"rep:\"r+0}")
    out = subprocess.run(["awk", "-F" + sep, prog, p], capture_output=True, text=True).stdout.split()
    d = dict(x.rsplit(":", 1) for x in out)
    rep = int(d.pop("rep", 0))
    return sorted(((int(k), int(v)) for k, v in d.items()), key=lambda kv: -kv[1]), rep


def structure(rel):
    section("(i) STRUCTURE")
    for p in rel.files:
        if rel.fmt == "none" or not (p.startswith("zip::") or os.path.exists(p)):
            continue
        if rel.fmt == "jsonl-gz":
            import gzip
            fc, bad = Counter(), 0
            with gzip.open(p, "rt") as f:
                for line in f:
                    try:
                        fc[len(json.loads(line))] += 1
                    except ValueError:
                        bad += 1
            print(f"  {os.path.basename(p):<44} keys per record={dict(fc)} unparsable={bad}")
            continue
        top, rep = field_counts(p, rel)
        shown = p.split("::")[-1] if p.startswith("zip::") else os.path.relpath(p, os.path.expanduser("~/datasets"))
        print(f"  {shown:<60} fields per line="
              f"{dict(top[:6])}{' (+%d more counts)' % (len(top) - 6) if len(top) > 6 else ''}"
              f"; header lines inside the data={rep}")
    df = rel.df
    cols = [c for c in rel.feats + rel.ids if c in df.columns]
    smp = df[cols].sample(min(len(df), 50_000), random_state=0)
    bad = []
    for c in cols:
        k = kind(smp[c])
        for rx, ok in EXPECT:
            if re.search(rx, c, re.I) and k not in ok:
                bad.append(f"{c} holds {k}")
                break
        else:
            if k in ("epoch-s", "epoch-ms", "IPv4", "MAC", "wifi-freq-MHz") and c in rel.feats:
                bad.append(f"{c} holds {k} (a content feature)")
    print(f"  columns checked {len(cols)}; value kind contradicting the name or role: "
          f"{len(bad)}" + (": " + "; ".join(bad[:12]) + (" ..." if len(bad) > 12 else "") if bad else ""))
    num = df[rel.feats].select_dtypes("number")
    print(f"  NaN cells in content features {int(df[rel.feats].isna().sum().sum()):,}; "
          f"infinite {int(np.isinf(num.to_numpy(np.float64)).sum()):,}")
    return {"value_kind_conflicts": bad}


def vec_hash(df, cols):
    return pd.util.hash_pandas_object(df[cols], index=False).values


def labels(rel):
    section("(ii) LABELS")
    df, y = rel.df, rel.df["_y"].astype(str).values
    n = len(df)
    hall = vec_hash(df, rel.feats + rel.ids)
    hf = vec_hash(df, rel.feats)
    dup_all = int(pd.Series(hall).duplicated().sum())
    dup_f = int(pd.Series(hf).duplicated().sum())
    print(f"rows {n:,}; duplicates on all columns {dup_all:,} ({pct(dup_all, n)}); "
          f"without identifiers {dup_f:,} ({pct(dup_f, n)})")
    pairs = pd.DataFrame({"h": hf, "y": y})
    nlab = pairs.drop_duplicates().groupby("h")["y"].size()
    multi = nlab[nlab > 1].index
    inm = np.isin(hf, multi)
    print(f"feature vectors with >1 label {len(multi):,}; rows in them {int(inm.sum()):,} "
          f"({pct(int(inm.sum()), n)})")
    ben = set(hf[y == rel.benign])
    out = {"dup_all": dup_all / n, "dup_feat": dup_f / n, "multi_rows": float(inm.mean()), "per_label": {}}
    for k in pd.Series(y).value_counts().index:
        m = y == k
        a, b = inm[m].mean(), np.isin(hf[m], list(ben)).mean() if k != rel.benign else float("nan")
        print(f"  {k:<28} rows {int(m.sum()):>11,}  in multi-label vectors {100 * a:6.2f}%"
              + ("" if k == rel.benign else f"  vector also occurs as benign {100 * b:6.2f}%"))
        out["per_label"][k] = {"rows": int(m.sum()), "multi": float(a),
                               "as_benign": None if k == rel.benign else float(b)}
    if rel.labelled_files:
        md5 = {}
        for p, lab in rel.labelled_files:
            h = hashlib.md5()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 22), b""):
                    h.update(blk)
            md5.setdefault(h.hexdigest(), []).append((os.path.basename(p), lab))
        same = [v for v in md5.values() if len({lab for _, lab in v}) > 1]
        print(f"identical files filed under different labels: {len(same)}"
              + "".join(f"\n  {v}" for v in same[:10]))
        out["identical_files_across_labels"] = len(same)
    return out


def timing(rel):
    section("(iii) TIME")
    df = rel.df
    if "_t" not in df or df["_t"].isna().all():
        print("no timestamp: time-ordered windows cannot be built")
        return {"has_time": False}
    t, y = df["_t"].to_numpy(np.float64), df["_y"].astype(str).values
    ok = ~np.isnan(t)
    grp = df["_g"].values if "_g" in df else np.zeros(len(df))
    mono = []
    for g in pd.unique(grp):
        s = t[(grp == g) & ok]
        if len(s) > 1:
            mono.append((np.diff(s) >= 0).sum() / (len(s) - 1))
    hours = rel.time_unit == "s"
    unit = (lambda v: f"{v / 3600:.2f} h") if hours else (lambda v: f"{v:,.0f} packets")
    print(f"{'timestamp' if hours else 'packet order (no timestamp)'} present in "
          f"{pct(int(ok.sum()), len(df))} of rows; non-decreasing in file order "
          f"{100 * np.mean(mono):.2f}% of steps (mean over {len(mono)} file(s)); span "
          f"{unit(np.nanmax(t) - np.nanmin(t))}")
    runs = int((pd.Series(y) != pd.Series(y).shift()).sum())
    print(f"label runs in file order {runs:,} for {len(set(y))} labels")
    bt = t[(y == rel.benign) & ok]
    out = {"has_time": True, "present": float(ok.mean()), "monotonic": float(np.mean(mono)), "per_label": {}}
    for k in pd.Series(y).value_counts().index:
        if k == rel.benign:
            continue
        s = t[(y == k) & ok]
        if not len(s):
            print(f"  {k:<28} no timestamp")
            continue
        lo, hi = s.min(), s.max()
        inside = (t >= lo) & (t <= hi) & ok
        bshare = (y[inside] == rel.benign).mean()
        bin_ = ((bt >= lo) & (bt <= hi)).sum()
        print(f"  {k:<28} span {unit(hi - lo):>16}  benign share inside {100 * bshare:6.2f}%  "
              f"benign rows inside {bin_:,}")
        out["per_label"][k] = {"benign_share_inside": float(bshare), "benign_inside": int(bin_)}
    return out


def attribution(rel):
    section("(iv) ATTRIBUTION")
    df = rel.attribution if rel.attribution is not None else rel.df
    if "_src" not in df or df["_src"].isna().all():
        print("no source address: alarms cannot be attributed to an emitter")
        return {"has_src": False}
    if rel.attribution is not None:
        print(f"from a separate table ({len(df):,} rows): {rel.notes[0] if rel.notes else ''}")
    src, y = df["_src"].astype(str).values, df["_y"].astype(str).values
    n = len(df)
    bsrc = set(src[y == rel.benign])
    for k in pd.Series(y).value_counts().index:
        m = y == k
        top = pd.Series(src[m]).value_counts().head(3)
        also = "" if k == rel.benign else f"  sources also emitting benign {pct(int(np.isin(src[m], list(bsrc)).sum()), int(m.sum()))}"
        print(f"  {k:<28} distinct sources {len(set(src[m])):>7,}  top: "
              + ", ".join(f"{i} ({pct(v, int(m.sum()))})" for i, v in top.items()) + also)
    acc = pd.DataFrame({"s": src, "y": y}).groupby(["s", "y"]).size().groupby(level=0).max().sum() / n
    maj = pd.Series(y).value_counts().iloc[0] / n
    print(f"source alone -> label (majority per source): {100 * acc:.2f}%, against {100 * maj:.2f}% "
          f"for the majority class")
    return {"has_src": True, "src_acc": float(acc), "majority": float(maj)}


def prevalence(rel):
    section("(v) PREVALENCE")
    y = rel.df["_y"].astype(str)
    vc = y.value_counts()
    for k, v in vc.items():
        print(f"  {k:<28}{v:>12,}  {pct(v, len(y))}")
    nb = int(vc.get(rel.benign, 0))
    print(f"benign {nb:,} rows ({pct(nb, len(y))}); attack {pct(len(y) - nb, len(y))}")
    return {"benign_rows": nb, "benign_share": nb / len(y)}


def probe(rel):
    section("(vi) SHORTCUT PROBE — XGBoost macro-F1 "
            f"(100 trees, depth 6; dedup per label; <= {CAP:,} rows per label)")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    df = rel.df
    y = df["_y"].astype(str).values
    keep = np.zeros(len(df), bool)
    first = ~pd.DataFrame({"h": vec_hash(df, rel.feats + rel.ids), "y": y}).duplicated().values
    rng = np.random.default_rng(SEED)
    vc = pd.Series(y[first]).value_counts()
    dropped = vc[vc < MIN_ROWS]
    for k in vc.index[vc >= MIN_ROWS]:
        ii = np.where(first & (y == k))[0]
        keep[rng.choice(ii, min(CAP, len(ii)), replace=False)] = True
    idx = np.where(keep)[0]
    names = sorted(set(y[idx]))
    if len(names) < 2:
        print(f"only one label in this table ({names}): no probe")
        return {}
    yi = np.searchsorted(names, y[idx])
    if len(dropped):
        print(f"labels left out (< {MIN_ROWS} distinct rows): {dropped.to_dict()}")
    print(f"probe rows {len(idx):,}: " + ", ".join(f"{k} {v:,}" for k, v in
                                                  pd.Series(y[idx]).value_counts().items()))
    X = df.iloc[idx][rel.feats + rel.ids].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).astype("category").cat.codes
    X = X.astype(np.float32).replace([np.inf, -np.inf], np.nan)
    splits = {"random 70/30": train_test_split(np.arange(len(idx)), test_size=0.3, stratify=yi,
                                               random_state=SEED)}
    if "_t" in df and df["_t"].notna().mean() > 0.5:
        t = df["_t"].to_numpy(np.float64)[idx]
        a, b = [], []
        for c in range(len(names)):
            ii = np.where(yi == c)[0]
            ii = ii[np.argsort(np.nan_to_num(t[ii], nan=np.inf), kind="stable")]
            cut = int(0.7 * len(ii)); a += list(ii[:cut]); b += list(ii[cut:])
        splits["time-ordered 70/30 per label"] = (np.array(a), np.array(b))
    if "_part" in df:
        part = df["_part"].values[idx]
        splits["independent: release test file"] = (np.where(part == "train")[0], np.where(part == "test")[0])
    elif "_g" in df and df["_g"].nunique() > 1:
        # leave groups out per label: each group (file, capture, recording) is filed under its
        # most frequent non-benign label; of the groups of each label, the last 30 % in name
        # order (at least one, never all) are held out, so every held-out label is also trained
        g = df["_g"].astype(str).values[idx]
        gl = pd.DataFrame({"g": g, "y": y[idx]})
        att = gl[gl["y"] != rel.benign]
        glab = att.groupby("g")["y"].agg(lambda s: s.value_counts().index[0]) if len(att) else pd.Series(dtype=str)
        glab = glab.reindex(sorted(set(g))).fillna(rel.benign)
        test_g = []
        for lab, gs in glab.groupby(glab):
            gs = sorted(gs.index)
            if len(gs) >= 2:
                test_g += gs[len(gs) - max(1, int(round(0.3 * len(gs)))):]
        te = np.isin(g, test_g)
        single = sorted(set(glab) - set(glab[glab.index.isin(test_g)]))
        splits[f"independent: held-out groups ({len(glab) - len(test_g)} train / {len(test_g)} test)"] = \
            (np.where(~te)[0], np.where(te)[0])
        if single:
            print(f"labels with a single group (trained only, not held out): {single}")
    out = {}
    for sname, cols in (("content", rel.feats), ("content + identifiers", rel.feats + rel.ids)):
        if sname.endswith("identifiers") and not rel.ids:
            continue
        for split, (a, b) in splits.items():
            if not len(a) or not len(b):
                print(f"  {sname:<24} {split:<44} (empty side, skipped)")
                continue
            present = np.unique(yi[a])
            remap = np.full(len(names), -1); remap[present] = np.arange(len(present))
            m = xgb.XGBClassifier(**PROBE)
            m.fit(X.iloc[a][cols], remap[yi[a]])
            p = present[m.predict(X.iloc[b][cols]).astype(int)]
            per = f1_score(yi[b], p, average=None, labels=range(len(names)), zero_division=0)
            seen = np.isin(range(len(names)), np.unique(yi[b]))
            mf = per[seen].mean()
            print(f"  {sname:<24} {split:<44} macro-F1={mf:.3f}  "
                  + "  ".join(f"{k}={v:.2f}" for k, v, s in zip(names, per, seen) if s))
            out[f"{sname} | {split}"] = {"macro_f1": float(mf),
                                         "per_label": {k: float(v) for k, v, s in zip(names, per, seen) if s}}
    return out


def run(name):
    import audit_loaders as L
    rels = L.LOADERS[name]()
    rels = rels if isinstance(rels, list) else [rels]
    for i, rel in enumerate(rels):
        audit(rel, name if len(rels) == 1 else f"{name}_{i + 1}")


def audit(rel, name):
    t0 = time.time()
    buf = io.StringIO()
    res = {"dataset": rel.name, "sample": rel.sample, "rows": len(rel.df),
           "feats": len(rel.feats), "ids": rel.ids}
    with redirect_stdout(buf):
        print(f"DATASET AUDIT — {rel.name}")
        print(f"rows {len(rel.df):,} ({rel.sample}); content features {len(rel.feats)}; "
              f"identifiers {rel.ids}; benign label '{rel.benign}'")
        for note in rel.notes:
            print(f"note: {note}")
        res["structure"] = structure(rel)
        res["labels"] = labels(rel)
        res["time"] = timing(rel)
        res["attribution"] = attribution(rel)
        res["prevalence"] = prevalence(rel)
        res["probe"] = probe(rel)
        print(f"\n(run time {time.time() - t0:.0f} s)")
    os.makedirs("results/audit", exist_ok=True)
    open(f"results/audit/{name}.txt", "w").write(buf.getvalue())
    json.dump(res, open(f"results/audit/{name}.json", "w"), indent=1, default=float)
    print(buf.getvalue())


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import audit_loaders
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        print("\n".join(audit_loaders.LOADERS))
        sys.exit(0)
    for nm in sys.argv[1:]:
        run(nm)
