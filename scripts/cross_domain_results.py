#!/usr/bin/env python3
"""Cross-domain results for Paper 2: the ERENO protocol applied, unchanged, to the
corrected CICIDS2017 and to CICIoT2023 under their profiles (docs/PROFILES.md).

Protocol (as for ERENO, scripts/redund_k2_fulltest.py and scripts/oos_thresholds.py):
  * N = 14 XGBoost specialists, two per category (7 categories), max_depth 4, eta 0.1,
    10 rounds, scale_pos_weight, seed 42; node i serves category i % 7 (pair i // 7);
    each specialist trains on its half of the category plus a 1/14 share of benign;
  * fusion k-of-n (k = 1, 2); membership shrinks 14 -> 3: GL keeps the retained union of
    the 14 boosters, FL re-aggregates the present nodes only;
  * window triage (corrected CICIDS2017 only, it has time): 1-s windows; thresholds
    calibrated on benign windows only, margin policy (T = ceil(1.25 max n_flags) + 1,
    f = 1.25 max fraction), alarm when n_flags >= T OR fraction > f.
Features: all columns left after the audit exclusions (no GRASP selection here).

Splits
  * CICIDS2017-C: per day and per attack category, the last 30 % of the episode's time
    span is TEST (every flow inside it, benign included); Monday (benign only) gives the
    calibration windows (first 70 %) and benign test windows (last 30 %); the rest trains.
  * CICIoT2023: files part-00000..06 train, part-00007..09 test (rows are shuffled; no
    time, so no windows).

  python scripts/cross_domain_results.py
Out: results/cross_domain_{cicids2017c,ciciot2023}.{txt,csv}
"""
import glob
import io
import math
import os
import sys
import zipfile
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import beta

SEED, N, K_SET = 42, 14, (1, 2)
XGB = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 4}
RHO_ERENO = 13.75            # benign:attack in the ERENO test file
D = os.path.expanduser("~/datasets/")


def cp_upper(x, n, conf=0.95):
    return 1.0 if x >= n else float(beta.ppf(conf, x + 1, n - x))


def train_specialists(X, cat, cats):
    rng = np.random.default_rng(SEED)
    ben = np.where(cat == "Benign")[0]
    ben_share = np.array_split(rng.permutation(ben), N)
    halves = {}
    for c in cats:
        idx = rng.permutation(np.where(cat == c)[0])
        halves[c] = np.array_split(idx, 2)
    boosters, plan = [], []
    for i in range(N):
        c = cats[i % len(cats)]
        pos = halves[c][i // len(cats)]
        neg = ben_share[i]
        rows = np.r_[pos, neg]
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        p = dict(XGB)
        if len(pos) and len(neg):
            p["scale_pos_weight"] = len(neg) / len(pos)
        boosters.append(xgb.train(p, xgb.DMatrix(X[rows], label=y), num_boost_round=10))
        plan.append((i, c, len(pos), len(neg)))
    return boosters, plan


def votes(boosters, X):
    d = xgb.DMatrix(X)
    return np.column_stack([(b.predict(d) >= 0.5).astype(np.int8) for b in boosters])


def stats(pred, is_atk):
    tp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
    nA, nB = int(is_atk.sum()), int((~is_atk).sum())
    rec = tp / nA; fpr = fp / nB if nB else 0.0
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    prec_r = rec / (rec + RHO_ERENO * fpr) if rec else 0.0          # at ERENO prevalence
    f1_r = 2 * prec_r * rec / max(prec_r + rec, 1e-12)
    return {"F1": 100 * f1, "recall": 100 * rec, "prec": 100 * prec, "FPR": 100 * fpr,
            "FP": fp, "F1@13.75": 100 * f1_r}


def shrink_table(V, is_atk, tag, rows_csv):
    print(f"\n-- membership shrink 14 -> 3 ({tag}); GL keeps the union of 14, FL the present nodes")
    print(f"{'N':>3} {'k':>2} | {'FL F1':>7} {'FL rec':>7} {'FL FPR':>7} | {'GL F1':>7} {'GL rec':>7} "
          f"{'GL FPR':>7} {'GL F1@13.75':>11}")
    for n_active in range(N, 2, -1):
        vf, vg = V[:, :n_active].sum(1), V.sum(1)
        for k in K_SET:
            f, g = stats(vf >= k, is_atk), stats(vg >= k, is_atk)
            print(f"{n_active:>3} {k:>2} | {f['F1']:7.2f} {f['recall']:7.2f} {f['FPR']:7.3f} | "
                  f"{g['F1']:7.2f} {g['recall']:7.2f} {g['FPR']:7.3f} {g['F1@13.75']:11.2f}")
            rows_csv.append({"dataset": tag, "N": n_active, "k": k,
                             **{f"FL_{a}": round(b, 4) for a, b in f.items()},
                             **{f"GL_{a}": round(b, 4) for a, b in g.items()}})


def per_category(V, cat_te, cats, k=2):
    fused = V.sum(1) >= k
    print(f"\n-- per-category recall, GL union, k>={k}")
    for c in cats:
        m = cat_te == c
        print(f"  {c:<14} n={int(m.sum()):>9,}  recall={100 * fused[m].mean():6.2f}%")
    m = cat_te == "Benign"
    print(f"  {'Benign':<14} n={int(m.sum()):>9,}  FPR={100 * fused[m].mean():6.3f}%")


# ----------------------------------------------------------------------------- CICIDS2017-C
IDENT = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]
TEXT = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "Attempted Category"]
FINGERPRINT = ["FWD Init Win Bytes", "Bwd Init Win Bytes"]
CATS_C = ["DoS", "DDoS", "PortScan", "BruteForce", "Web", "Bot", "Infiltration"]


def cat_cicids(l):
    l = l.lower()
    if "attempted" in l or l == "benign":
        return "Benign"
    if l == "heartbleed":
        return "drop"
    if l.startswith("ddos"):
        return "DDoS"
    if l.startswith("dos"):
        return "DoS"
    if "portscan" in l:
        return "PortScan"              # Portscan + Infiltration - Portscan (merged)
    if l.startswith("infiltration"):
        return "Infiltration"
    if "patator" in l:
        return "BruteForce"
    if l.startswith("web attack"):
        return "Web"
    if l.startswith("botnet"):
        return "Bot"
    return "??"


def windows(ts_s, fused, is_atk):
    """1-s windows over a time-sorted segment; returns arrays (n, nf, frac, atk)."""
    w = np.floor(ts_s).astype(np.int64)
    uw, start = np.unique(w, return_index=True)
    end = np.r_[start[1:], len(w)]
    cs = np.r_[0, np.cumsum(fused)]
    ca = np.r_[0, np.cumsum(is_atk)]
    n = end - start
    nf = cs[end] - cs[start]
    return n, nf, nf / n, (ca[end] - ca[start]) > 0


def cicids():
    z = zipfile.ZipFile(D + "cicids2017_improved/CICIDS2017_improved.zip")
    frames = []
    for d in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        df = pd.read_csv(z.open(f"{d}.csv"), low_memory=False, dtype={c: "str" for c in TEXT})
        df["_day"] = d
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["_cat"] = df["Label"].map(cat_cicids)
    df = df[df["_cat"] != "drop"].reset_index(drop=True)
    print("categories:", df["_cat"].value_counts().to_dict())
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="mixed")
    feats = [c for c in df.columns if c not in TEXT + IDENT + FINGERPRINT + ["_day", "_cat", "_ts"]]
    sel = os.environ.get("CICIDS2017C_FEATURES")       # e.g. the GRASP combined selection
    if sel:
        import json
        feats = json.load(open(sel))["names"]
        print(f"feature selection from {sel}: {len(feats)} features")
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float32").to_numpy()
    X[~np.isfinite(X)] = np.nan
    print(f"flows={len(df):,}; features={len(feats)} (identifiers and Init Win Bytes excluded)")

    test = np.zeros(len(df), bool)
    for d, g in df.groupby("_day"):
        if d == "monday":
            continue
        for c, gc in g.groupby("_cat"):
            if c == "Benign":
                continue
            t70, tmax = gc["_ts"].quantile(0.7), gc["_ts"].max()
            test |= ((df["_day"] == d) & (df["_ts"] >= t70) & (df["_ts"] <= tmax)).to_numpy()
    mon = (df["_day"] == "monday").to_numpy()
    t70m = df.loc[mon, "_ts"].quantile(0.7)
    calib = mon & (df["_ts"] < t70m).to_numpy()
    test |= mon & (df["_ts"] >= t70m).to_numpy()
    train = ~test & ~calib
    cat = df["_cat"].to_numpy()
    print("train per category:", pd.Series(cat[train]).value_counts().to_dict())
    print("test  per category:", pd.Series(cat[test]).value_counts().to_dict())
    print(f"calibration (Monday, first 70 %): {int(calib.sum()):,} benign flows")

    boosters, plan = train_specialists(X[train], cat[train], CATS_C)
    print("specialists:", [(i, c, p, n) for i, c, p, n in plan])
    Vte = votes(boosters, X[test])
    is_atk = cat[test] != "Benign"
    rows = []
    shrink_table(Vte, is_atk, "cicids2017c", rows)
    per_category(Vte, cat[test], CATS_C)

    # window triage (seconds since epoch; the datetime unit may be us or ns)
    secs = (df["_ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    Vc = votes(boosters, X[calib])
    out = []
    for k in K_SET:
        for wlen in (1, 10):
            fc = (Vc.sum(1) >= k).astype(int)
            tc = secs[calib] / wlen
            o = np.argsort(tc, kind="stable")
            n, nf, fr, atk = windows(tc[o], fc[o], np.zeros(len(o), bool))
            T = int(math.ceil(nf.max() * 1.25)) + 1
            f = fr.max() * 1.25
            ft = (Vte.sum(1) >= k).astype(int)
            tt = secs[test] / wlen
            o = np.argsort(tt, kind="stable")
            n2, nf2, fr2, atk2 = windows(tt[o], ft[o], is_atk[o])
            r1 = nf2 >= T
            r2 = r1 | (fr2 > f)
            nA, nB = int(atk2.sum()), int((~atk2).sum())
            for rn, r in (("volume", r1), ("volume+fraction", r2)):
                tp, fp = int((r & atk2).sum()), int((r & ~atk2).sum())
                out.append((k, wlen, rn, T, f, tp, nA, fp, nB))
    print("\n-- window triage (calibration: Monday benign windows, first 70 %; margin x1.25)")
    print(f"{'k':>2} {'win':>4} {'rule':<16} {'T':>5} {'f':>6} | {'attack windows':>18} | "
          f"{'benign windows':>22} {'FPR 95% up':>10}")
    for k, wlen, rn, T, f, tp, nA, fp, nB in out:
        print(f"{k:>2} {wlen:>3}s {rn:<16} {T:>5} {min(f, 9.99):6.3f} | {tp:>7,}/{nA:<7,} "
              f"{100 * tp / nA:5.1f}% | {fp:>7,}/{nB:<9,} {100 * fp / nB:5.2f}% "
              f"{100 * cp_upper(fp, nB):9.2f}%")
        rows.append({"dataset": "cicids2017c", "N": "windows", "k": k, "window_s": wlen,
                     "rule": rn, "T": T, "f": f, "TP": tp, "nA": nA, "FP": fp, "nB": nB})
    return rows


# ----------------------------------------------------------------------------- CICIoT2023
CATS_I = ["DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]


def cat_ciciot(lbl):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inspect_ciciot2023 import category
    return category(lbl)


def ciciot():
    parts = sorted(glob.glob(D + "ciciot2023/part-*.csv"))
    tr = pd.concat([pd.read_csv(p) for p in parts[:7]], ignore_index=True)
    te = pd.concat([pd.read_csv(p) for p in parts[7:]], ignore_index=True)
    lab = tr.columns[-1]
    from inspect_ciciot2023 import category
    for d in (tr, te):
        d["_cat"] = d[lab].map(category)
    feats = [c for c in tr.columns if c not in (lab, "_cat", "IAT")]
    print(f"train rows={len(tr):,} (parts 0-6), test rows={len(te):,} (parts 7-9); "
          f"features={len(feats)} (IAT excluded)")
    print("train per category:", tr["_cat"].value_counts().to_dict())
    print("test  per category:", te["_cat"].value_counts().to_dict())
    Xtr = tr[feats].to_numpy(np.float32)
    Xte = te[feats].to_numpy(np.float32)
    for X in (Xtr, Xte):
        X[~np.isfinite(X)] = np.nan
    boosters, plan = train_specialists(Xtr, tr["_cat"].to_numpy(), CATS_I)
    print("specialists:", [(i, c, p, n) for i, c, p, n in plan])
    V = votes(boosters, Xte)
    is_atk = te["_cat"].to_numpy() != "Benign"
    rows = []
    shrink_table(V, is_atk, "ciciot2023", rows)
    per_category(V, te["_cat"].to_numpy(), CATS_I)
    print("\n(no window triage: the CSV release has no time; F1@13.75 rescales precision to "
          "the ERENO benign:attack ratio, recall and FPR are prevalence-invariant)")
    return rows


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    only = sys.argv[1] if len(sys.argv) > 1 else None       # cicids2017c | ciciot2023
    suffix = os.environ.get("CROSS_SUFFIX", "")             # e.g. _grasp
    for name, fn in (("cicids2017c", cicids), ("ciciot2023", ciciot)):
        if only and name != only:
            continue
        name = name + suffix
        buf = io.StringIO()
        with redirect_stdout(buf):
            rows = fn()
        txt = buf.getvalue()
        open(f"results/cross_domain_{name}.txt", "w").write(txt)
        pd.DataFrame(rows).to_csv(f"results/cross_domain_{name}.csv", index=False)
        print(f"\n################ {name} ################\n{txt}", flush=True)
