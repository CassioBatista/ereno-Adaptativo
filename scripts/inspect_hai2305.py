#!/usr/bin/env python3
"""Run the dataset-audit protocol (paper_v2_dataset_audit.tex) on HAI 23.05 (HIL-based
Augmented ICS security dataset; GitHub icsdataset/hai, files fetched from the authors'
Kaggle and checked against the GitHub LFS SHA-256).

HAI 23.05 labels are binary (label-test*.csv). The technical manual lists every attack
(ID, scenario, target controller, start time, duration); the label segments are matched to
that list in order, which (a) cross-checks the labels and (b) yields a per-attack label set
by target control loop.

  (i)   STRUCTURE   columns, NaN, 1-s continuity, label files aligned with test files
  (ii)  LABELS      segments vs the manual (count, start, duration), control-loop classes,
                    duplicates, conflicting vectors
  (iii) TIME        attacks inside continuous operation
  (iv)  ATTRIBUTION none at network level; attacked loop as subsystem
  (v)   PREVALENCE  attack share; normal hours
  (vi)  SHORTCUT    binary and per-loop probes: random, time-ordered, cross-file,
                    held-out attack instances

  python scripts/inspect_hai2305.py ~/datasets/hai_2305
Out: results/hai2305_audit.txt
"""
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd


def section(t):
    print(f"\n=== {t} ===")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def manual_table(d):
    import pypdf
    t = "\n".join(p.extract_text() or "" for p in
                  pypdf.PdfReader(os.path.join(d, "hai_dataset_technical_details.pdf")).pages)
    s = t.index("52 attacks were conducted")         # the 23.05 attack table
    block = t[s:s + 12000]
    heads = list(re.finditer(r"\b(\d{1,2})\s+(A[12]\d{2})\s+(A[PE]\d{2})\s+(P\d-[A-Z]{2}\S*)", block))
    out, seen = [], set()
    for i, m in enumerate(heads):
        no, aid, ap, target = m.groups()
        if aid in seen:
            break                                     # the next version's table starts
        seen.add(aid)
        span = block[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(block)]
        tm = re.findall(r"(\d{1,2}:\d{2})\s+(\d+)", span)
        if not tm:
            continue
        hhmm, dur = tm[-1]
        out.append({"no": int(no), "id": aid, "scenario": ap, "target": target,
                    "loop": "-".join(target.split("-")[:2]), "hhmm": hhmm, "dur": int(dur)})
    return pd.DataFrame(out).sort_values("no")


def segments(lab):
    a = lab.values
    st = np.where((a == 1) & (np.r_[0, a[:-1]] == 0))[0]
    en = np.where((a == 1) & (np.r_[a[1:], 0] == 0))[0]
    return list(zip(st, en))


def main(d):
    section("(i) STRUCTURE")
    fr = {}
    for f in ["hai-train1", "hai-train2", "hai-train3", "hai-train4", "hai-test1", "hai-test2"]:
        df = pd.read_csv(os.path.join(d, f + ".csv"))
        df["_t"] = pd.to_datetime(df["timestamp"])
        step = df["_t"].diff().dt.total_seconds().value_counts().head(3).to_dict()
        fr[f] = df
        print(f"  {f}: rows={len(df):,} cols={df.shape[1] - 1} NaN={int(df.isna().sum().sum())} "
              f"{df['_t'].min()} .. {df['_t'].max()} step(s)={step}")
    pv = [c for c in fr["hai-train1"].columns if c not in ("timestamp", "_t")]
    for k in ("1", "2"):
        lab = pd.read_csv(os.path.join(d, f"label-test{k}.csv"))
        same = (pd.to_datetime(lab["timestamp"]).values == fr[f"hai-test{k}"]["_t"].values).all()
        fr[f"hai-test{k}"]["label"] = lab["label"].values
        print(f"  label-test{k}: rows={len(lab):,}; timestamps identical to hai-test{k}: {bool(same)}")
    print(f"process variables={len(pv)}")

    section("(ii) LABELS — segments vs the technical manual")
    man = manual_table(d)
    print(f"attacks listed in the manual for 23.05: {len(man)} "
          f"(test1 A1xx={int(man['id'].str.startswith('A1').sum())}, "
          f"test2 A2xx={int(man['id'].str.startswith('A2').sum())})")
    rows = []
    for k, prefix in (("1", "A1"), ("2", "A2")):
        df = fr[f"hai-test{k}"]
        seg = segments(df["label"])
        m = man[man["id"].str.startswith(prefix)].reset_index(drop=True)
        print(f"  test{k}: label segments={len(seg)}; manual entries={len(m)}")
        for i, (s, e) in enumerate(seg):
            r = m.iloc[i] if i < len(m) else None
            t0 = df["_t"].iloc[s]
            dur = int(e - s + 1)
            ok_t = r is not None and t0.strftime("%-H:%M") == r["hhmm"]
            ok_d = r is not None and abs(dur - r["dur"]) <= max(5, 0.1 * r["dur"])
            rows.append({"file": f"test{k}", "seg": i, "start": t0, "dur": dur,
                         "id": None if r is None else r["id"], "loop": None if r is None else r["loop"],
                         "target": None if r is None else r["target"],
                         "man_dur": None if r is None else r["dur"], "start_ok": ok_t, "dur_ok": ok_d})
            df.loc[s:e, "loop"] = None if r is None else r["loop"]
            df.loc[s:e, "attack_id"] = None if r is None else r["id"]
        df["loop"] = df["loop"].fillna("normal")
    seg = pd.DataFrame(rows)
    print(f"  segments whose start minute matches the manual: {int(seg['start_ok'].sum())}/{len(seg)}; "
          f"duration within max(5 s, 10%): {int(seg['dur_ok'].sum())}/{len(seg)}")
    bad = seg[~(seg["start_ok"] & seg["dur_ok"])]
    if len(bad):
        print("  mismatches:")
        print(bad[["file", "seg", "start", "dur", "id", "target", "man_dur"]].to_string(index=False))
    print("  attacks per target loop:", seg["loop"].value_counts().to_dict())
    test = pd.concat([fr["hai-test1"], fr["hai-test2"]], ignore_index=True)
    allx = pd.concat([fr[f] for f in fr], ignore_index=True)
    h = pd.util.hash_pandas_object(allx[pv], index=False)
    print(f"  duplicates of process vectors (time removed): {int(h.duplicated().sum()):,} "
          f"({pct(int(h.duplicated().sum()), len(allx))})")
    ht = pd.util.hash_pandas_object(test[pv], index=False)
    g = test.groupby(ht.values)["label"].nunique()
    print(f"  test vectors with >1 label: {int((g > 1).sum()):,}")

    section("(iii) TIME")
    for k in ("1", "2"):
        df = fr[f"hai-test{k}"]
        a = df["label"].values
        lo, hi = df.loc[a == 1, "_t"].min(), df.loc[a == 1, "_t"].max()
        inside = (df["_t"] >= lo) & (df["_t"] <= hi)
        print(f"  test{k}: benign share between first and last attack "
              f"{pct(int((inside & (a == 0)).sum()), int(inside.sum()))}")
    print("  training files carry no label column (normal operation by construction)")

    section("(iv) ATTRIBUTION")
    print("  process telemetry only: no network addresses; each attack targets one control loop "
          "(or two, for the combined attacks), which the per-loop labels locate")

    section("(v) PREVALENCE")
    ntrain = sum(len(fr[f]) for f in fr if "train" in f)
    print(f"  test: attack {pct(int(test['label'].sum()), len(test))} of {len(test):,} s "
          f"({len(test) / 3600:.1f} h); normal training operation {ntrain / 3600:.1f} h")
    print("  per loop (test seconds):", test["loop"].value_counts().to_dict())

    section("(vi) SHORTCUT PROBE")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inspect_cps_survey4 import probe
    X = test[pv].reset_index(drop=True)
    yb = np.where(test["label"].values == 1, "attack", "normal")
    fid = np.r_[np.zeros(len(fr["hai-test1"])), np.ones(len(fr["hai-test2"]))]
    probe(X, yb, groups=fid, order=test["_t"].values, label="binary")
    # per-loop classes; held-out attack instances within each loop
    loops = test["loop"].values
    cnt = pd.Series(loops).value_counts()
    keep = pd.Series(loops).map(cnt).values >= 200
    Xl, yl = X[keep].reset_index(drop=True), loops[keep]
    probe(Xl, yl, order=test["_t"].values[keep], label="per loop")
    import xgboost as xgb
    from sklearn.metrics import f1_score
    ids = test["attack_id"].fillna("normal").values[keep]
    att = pd.Series(ids[ids != "normal"]).unique()
    rng = np.random.RandomState(42)
    held = set(rng.choice(att, size=max(1, len(att) // 3), replace=False))
    norm_idx = np.where(ids == "normal")[0]
    te_norm = set(norm_idx[int(0.7 * len(norm_idx)):])
    te = np.array([(i in held) or (j in te_norm) for j, i in enumerate(ids)])
    names = np.unique(yl)
    yi = np.searchsorted(names, yl)
    m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist", n_jobs=4, random_state=42)
    present = np.unique(yi[~te])
    remap = {c: i for i, c in enumerate(present)}
    m.fit(Xl[~te], np.array([remap[c] for c in yi[~te]]))
    p = present[m.predict(Xl[te])]
    per = f1_score(yi[te], p, average=None, labels=range(len(names)), zero_division=0)
    print(f"  [per loop] held-out attack instances ({len(held)} of {len(att)})        macro-F1={per.mean():.3f}  "
          + "  ".join(f"{n}={v:.2f}" for n, v in zip(names, per)))


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/hai_2305")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    os.makedirs("results", exist_ok=True)
    open("results/hai2305_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
