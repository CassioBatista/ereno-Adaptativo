#!/usr/bin/env python3
"""Evaluation-v1, step 1, EXPLORATORY variant (not pre-registered): per-model decision
thresholds calibrated on held-out benign rows, against the fixed threshold of 0.5.

Same domain, features, partition, model settings and test split as scripts/ev1_step1.py
(imported from it). The only changes:
  - Calibration rows. Each node keeps CAL = 20 % of its benign rows apart (random, seeded,
    independent of the partition draws) and trains on the other 80 %; the attack rows are
    all used for training. For the attack-based partition the held-out rows are taken from
    the node's benign slice and shared by the specialists of that node; for IID, from the
    benign rows of each IID part; for Centralized, from the pooled benign rows.
  - Threshold. A model with target false-positive rate a flags a sample when its score is
    strictly above t, the (floor(a n) + 1)-th largest score of its n held-out benign rows
    (so at most floor(a n) of them are flagged). When a n < 1, t is the largest held-out
    benign score ("resolution-limited": the calibration set is too small for the target).
  - Targets. A total false-positive budget A for the fused detector, split over its models by
    the union bound: a = A / S per specialist (attack-based OR), A / N per IID model,
    A for Centralized. A in {1 %, 0.1 %}. The k = 2 row uses the thresholds of A = 1 %.
  - Reported for each partition: the models trained on 80 % of the benign rows at 0.5 (to
    separate the effect of the smaller training set from that of the threshold) and
    calibrated at each A.
Only training rows are used to set thresholds; the test split is read once, for scoring.

  python scripts/ev1_step1_calib.py --domain ereno|cicids|morris4|ciciot [--seed 42] [--nodes 14|spec]
Out: results/ev1_step1_calib_<domain>_s<seed>[_n<N>].{txt,json}
"""
import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import ev1_step1 as E  # noqa: E402

CAL = 0.20
TARGETS = (0.01, 0.001)


def split_cal(idx_benign, rng):
    """(fit, cal) parts of a set of benign row indices."""
    p = rng.permutation(idx_benign)
    k = int(round(CAL * len(p)))
    return np.sort(p[k:]), np.sort(p[:k])


def threshold(s, a):
    s = np.sort(np.asarray(s, np.float64))[::-1]
    n = len(s)
    k = int(np.floor(a * n))
    return (float(s[min(k, n - 1)]), n, k, a * n < 1)


def run(domain, seed):
    import xgboost as xgb
    dom = E.DOMAINS[domain]
    N = E.N
    tag = f"{domain}_s{seed}" + ("" if N == 14 else f"_n{N}")
    t0 = time.time()
    feats, _ = E.selection(dom)
    X, y, classes, nc = E.load(dom["train"], feats)
    units, slices, kept = E.attack_partition(y, nc, classes, dom["cap"], seed)
    S = len(units)
    types = [c for c in range(len(classes)) if c != nc and (y == c).any()]
    rng = np.random.default_rng(seed + 7)          # calibration draws, apart from the partition's

    def fit(rows, yb, rounds):
        b, p, q = E.train(X[rows], yb, rounds, seed)
        return b, p, q

    def pred(b, rows):
        return b.predict(xgb.DMatrix(X[rows]))

    # attack-based: per node, 80 % of the benign slice for training, 20 % for calibration
    node_split = [split_cal(sl, rng) for sl in slices]
    spec = []
    for j, (name, codes, node, idx) in enumerate(units):
        fit_b, cal_b = node_split[node]
        rows = np.concatenate([idx, fit_b])
        b, p, q = fit(rows, np.isin(y[rows], codes).astype(np.int8), E.TREES)
        sc = pred(b, cal_b)
        spec.append({"j": j, "name": name, "node": node + 1, "attack": p, "benign_fit": q,
                     "n_cal": len(cal_b), "booster": b, "cal_scores": sc})
        print(f"[spec {j:02d}] node {node + 1:2d} {name[:50]}: {p:,} / {q:,} fit, {len(cal_b):,} cal", flush=True)

    iid_rounds = int(round(E.TREES * S / N))
    iid = []
    for i, idx in enumerate(E.iid_partition(y, nc, dom["cap"], seed)):
        fit_b, cal_b = split_cal(idx[y[idx] == nc], rng)
        rows = np.concatenate([idx[y[idx] != nc], fit_b])
        b, p, q = fit(rows, (y[rows] != nc).astype(np.int8), iid_rounds)
        iid.append({"node": i + 1, "attack": p, "benign_fit": q, "n_cal": len(cal_b),
                    "booster": b, "cal_scores": pred(b, cal_b)})
    print(f"[iid] {N} models of {iid_rounds} rounds", flush=True)

    fit_b, cal_b = split_cal(kept[y[kept] == nc], rng)
    rows = np.concatenate([kept[y[kept] != nc], fit_b])
    b, p, q = fit(rows, (y[rows] != nc).astype(np.int8), E.TREES * S)
    cen = {"attack": p, "benign_fit": q, "n_cal": len(cal_b), "booster": b, "cal_scores": pred(b, cal_b)}
    print(f"[central] {E.TREES * S} rounds", flush=True)
    del X, y, kept, slices, units

    Xt, yt, classes_t, nct = E.load(dom["test"], feats)
    assert classes_t == classes and nct == nc
    d = xgb.DMatrix(Xt)
    del Xt

    def tscore(m):
        m["booster"].set_param({"nthread": E.NTHREAD})
        return m["booster"].predict(d).astype(np.float32)
    Ps = np.column_stack([tscore(s) for s in spec])
    Pi = np.column_stack([tscore(m) for m in iid])
    Pc = tscore(cen)
    print(f"[score] {len(yt):,} test rows", flush=True)

    thr = {}
    for A in TARGETS:
        thr[A] = {"spec": [threshold(s["cal_scores"], A / S) for s in spec],
                  "iid": [threshold(m["cal_scores"], A / N) for m in iid],
                  "central": threshold(cen["cal_scores"], A)}
    det = {"attack-based OR @0.5 (80% benign)": (Ps >= 0.5).any(1),
           "IID OR @0.5 (80% benign)": (Pi >= 0.5).any(1),
           "Centralized @0.5 (80% benign)": Pc >= 0.5}
    for A in TARGETS:
        ts = np.array([t[0] for t in thr[A]["spec"]])
        ti = np.array([t[0] for t in thr[A]["iid"]])
        lab = f"A={100 * A:g}%"
        det[f"attack-based OR calib {lab}"] = (Ps > ts).any(1)
        if A == TARGETS[0]:
            det[f"attack-based k=2 calib {lab}"] = (Ps > ts).sum(1) >= 2
        det[f"IID OR calib {lab}"] = (Pi > ti).any(1)
        det[f"Centralized calib {lab}"] = Pc > thr[A]["central"][0]
    res = {k: E.metrics(v, yt, nc, types) for k, v in det.items()}

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"evaluation-v1 step 1, CALIBRATED THRESHOLDS (exploratory, not pre-registered) -- "
              f"{dom['name']} -- seed {seed} -- {N} nodes")
        print(f"features ({len(feats)}): {feats}")
        print(f"specialists {S}, IID {N} x {iid_rounds} rounds, Centralized {E.TREES * S} rounds; "
              f"{100 * CAL:g} % of each node's benign rows held out for calibration; test {len(yt):,} rows "
              f"(prevalence {100 * (yt != nc).mean():.2f} %)")
        print(f"\n{'detector':<38}{'F1':>8}{'recall':>9}{'prec.':>9}{'FPR %':>10}{'FP':>9}{'F1@6.8%':>9}{'P@1%':>9}{'P@0.1%':>9}")
        for k in det:
            e = res[k]
            b = ">=" if e["fp_zero_bound"] else "  "
            print(f"{k:<38}{100 * e['f1']:8.2f}{100 * e['recall']:9.2f}{100 * e['precision']:9.2f}"
                  f"{100 * e['fpr']:10.4f}{e['fp']:9,}{100 * e['f1_at_6.8']:9.2f}"
                  f"{b}{100 * e['precision_at']['0.01']:7.2f}{b}{100 * e['precision_at']['0.001']:7.2f}")
        print("\nper-type recall % (* fewer than 100 test samples)")
        keys = ["attack-based OR @0.5 (80% benign)", "attack-based OR calib A=1%", "attack-based OR calib A=0.1%",
                "IID OR calib A=1%", "Centralized calib A=1%"]
        print(f"{'type':<34}{'n':>9}" + "".join(f"{s:>12}" for s in ("AB@0.5", "AB A=1%", "AB A=.1%", "IID A=1%", "Cen A=1%")))
        for c in types:
            n = res[keys[0]]["per_type"][c]["n"]
            print(f"{'*' if n < 100 else ' '}{classes[c][:33]:<33}{n:9,}"
                  + "".join(f"{100 * res[k]['per_type'][c]['recall']:12.2f}" for k in keys))
        for A in TARGETS:
            print(f"\nthresholds, A = {100 * A:g} % (per specialist a = {100 * A / S:.4f} %): "
                  "node, name, n_cal, t, test FPR of the specialist, resolution-limited")
            for s, (t, n, k, lim) in zip(spec, thr[A]["spec"]):
                fpr = float((Ps[yt == nc, s['j']] > t).mean())
                print(f"  {s['node']:2d} {s['name'][:40]:<40} {n:7,} {t:.4f} {100 * fpr:7.3f}{'  limited' if lim else ''}")
            ti = thr[A]["iid"]
            print(f"  IID: t {min(t[0] for t in ti):.4f}-{max(t[0] for t in ti):.4f}, n_cal {ti[0][1]:,}, "
                  f"limited {sum(t[3] for t in ti)}/{len(ti)}; Centralized t {thr[A]['central'][0]:.4f}, "
                  f"n_cal {thr[A]['central'][1]:,}{', limited' if thr[A]['central'][3] else ''}")
        print(f"\nelapsed {time.time() - t0:.0f} s")
    print(buf.getvalue(), end="")
    out = f"results/ev1_step1_calib_{tag}"
    open(f"{out}.txt", "w").write(buf.getvalue())

    def js(m):
        return {k: (v if k != "per_type" else {classes[c]: x for c, x in v.items()}) for k, v in m.items()}
    json.dump({"domain": dom["name"], "seed": seed, "nodes": N, "cal_fraction": CAL, "targets": TARGETS,
               "results": {k: js(v) for k, v in res.items()},
               "thresholds": {str(A): {"spec": [list(t) for t in thr[A]["spec"]],
                                       "iid": [list(t) for t in thr[A]["iid"]],
                                       "central": list(thr[A]["central"])} for A in TARGETS},
               "specialists": [{k: v for k, v in s.items() if k not in ("booster", "cal_scores")} for s in spec]},
              open(f"{out}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=E.DOMAINS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nodes", default="14")
    a = ap.parse_args()
    os.chdir(ROOT)
    b = E.busy()
    if b:
        sys.exit("another job is running; not starting:\n" + "\n".join(b))
    E.N = E.specialist_count(E.DOMAINS[a.domain]["train"]) if a.nodes == "spec" else int(a.nodes)
    run(a.domain, a.seed)
