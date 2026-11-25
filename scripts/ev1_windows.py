#!/usr/bin/env python3
"""Evaluation-v1, section 6: window alarms (ERENO, CICIDS2017, Morris-4), on the per-sample
scores of step 1 (results/.cache/ev1_step1_<domain>_s<seed>.npz; scripts/ev1_step1.py), so the
specialists, features and test rows are those of Section 5.1. No model is trained here.

Per window (docs/EVALUATION_PROTOCOL.md, section 6):
  - a sample is flagged when the OR of the specialists fires (score >= 0.5 for any of them);
  - network-wide flagged count n_f and flagged fraction n_f / n; per specialist S_s = the
    flagged samples on which specialist s fired;
  - alarm: n_f >= T  OR  n_f / n > f  OR  any S_s >= T_s, with T = ceil(1.25 x max benign n_f)
    + 1, f = 1.25 x max benign fraction, T_s = max(ceil(1.25 x max benign S_s) + 1, 2)
    (the rule of the earlier runs, scripts/_cicids_profile_ablation.py);
  - blocked-5 out-of-sample calibration: the occupied windows, in time order, are cut into 5
    contiguous blocks; each block is scored with thresholds calibrated on the attack-free
    windows of the other four; results pooled over the five blocks.
A window is an attack window when it holds any attack sample; per type, the attack windows
that hold samples of that type. Reported: attack windows alarmed (exact 95 % interval) and
benign windows alarmed (false alarms; 95 % upper bound 3/n when none), for the full rule and
for the network-wide part alone (n_f, fraction), per type, and the thresholds of each fold.

Windows (section 6): ERENO 1 s; CICIDS2017 10 s (sensitivity 1 s and 60 s); Morris-4 100 s
(sensitivity 10 s and 1,000 s). Profile ablation, CICIDS2017: count windows of M consecutive
test flows in time order, M = mean test flows per occupied 10-s window, attribution none.
Time: ERENO column F1 of the test file (`Time`, an identifier, not a feature); CICIDS2017
and Morris-4 all_in_one_ev1_<ds>_test_order.npy (scripts/build_ev1_datasets.py). Rows are
sorted by time (stable) before windowing. CICIoT2023 has no window (rows shuffled).

  python scripts/ev1_windows.py [--seed 42] [--domains ereno,cicids,morris4]
Out: results/ev1_windows_<domain>_s<seed>.{txt,json}
"""
import argparse
import json
import math
import os

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
MARGIN, FLOOR, BLOCKS = 1.25, 2, 5
DOMS = {
    "ereno":   dict(name="ERENO", time=("csv", "all_in_one_ereno_test.csv", 0),
                    windows=[("time", 1.0, "primary")]),
    "cicids":  dict(name="CICIDS2017", time=("npy", "all_in_one_ev1_cicids_test_order.npy"),
                    windows=[("time", 10.0, "primary"), ("time", 1.0, "sensitivity"),
                             ("time", 60.0, "sensitivity"), ("count", 10.0, "profile ablation")]),
    "morris4": dict(name="Morris-4", time=("npy", "all_in_one_ev1_morris4_test_order.npy"),
                    windows=[("time", 100.0, "primary"), ("time", 10.0, "sensitivity"),
                             ("time", 1000.0, "sensitivity")]),
}


def cp(k, n):
    from scipy.stats import beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def times(dom):
    kind, path = dom["time"][0], os.path.join(ROOT, dom["time"][1])
    if kind == "npy":
        return np.load(path).astype(np.float64)
    import pandas as pd
    skip = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            skip += 1
            if line.strip().lower().startswith("@data"):
                break
    return pd.read_csv(path, skiprows=skip, header=None, usecols=[dom["time"][2]])[dom["time"][2]].to_numpy(np.float64)


def window_table(w, fired, y, nc, types):
    """Per occupied window (w sorted, non-decreasing): n, n_f, S (windows x specialists),
    attack flag and, per type, whether the window holds that type."""
    starts = np.flatnonzero(np.r_[True, w[1:] != w[:-1]])
    flag = fired.any(1)
    n = np.diff(np.r_[starts, len(w)])
    nf = np.add.reduceat(flag.astype(np.int64), starts)
    S = np.add.reduceat(fired.astype(np.int32), starts, axis=0)
    atk = np.add.reduceat((y != nc).astype(np.int64), starts) > 0
    has = {c: np.add.reduceat((y == c).astype(np.int64), starts) > 0 for c in types}
    return {"n": n, "nf": nf, "frac": nf / n, "S": S, "atk": atk, "has": has}


def calibrate(W, idx):
    b = idx[~W["atk"][idx]]
    if len(b) == 0:                     # no attack-free window to calibrate on: the block is not scored
        return None
    T = int(math.ceil(MARGIN * W["nf"][b].max())) + 1
    f = MARGIN * float(W["frac"][b].max())
    Ts = np.maximum(np.ceil(MARGIN * W["S"][b].max(0)) + 1, FLOOR).astype(np.int64)
    return {"T": T, "f": f, "Ts": Ts, "n_benign": int(len(b))}


def evaluate(W):
    m = len(W["n"])
    folds = np.array_split(np.arange(m), BLOCKS)
    net = np.zeros(m, bool)
    full = np.zeros(m, bool)
    scored = np.ones(m, bool)
    log = []
    for k, held in enumerate(folds):
        cal = np.setdiff1d(np.arange(m), held)
        th = calibrate(W, cal)
        if th is None:
            scored[held] = False
            log.append({"fold": k, "windows": int(len(held)), "attack": int(W["atk"][held].sum()),
                        "benign_calibration": 0, "T": None, "f": None, "Ts_min": None, "Ts_max": None})
            continue
        nw = (W["nf"][held] >= th["T"]) | (W["frac"][held] > th["f"])
        sp = (W["S"][held] >= th["Ts"]).any(1)
        net[held], full[held] = nw, nw | sp
        log.append({"fold": k, "windows": int(len(held)), "attack": int(W["atk"][held].sum()),
                    "benign_calibration": th["n_benign"], "T": th["T"], "f": round(th["f"], 6),
                    "Ts_min": int(th["Ts"].min()), "Ts_max": int(th["Ts"].max())})
    return net, full, log, scored


def rates(al, W, types, classes, scored):
    A, B = W["atk"] & scored, ~W["atk"] & scored
    out = {}
    for name, a in al.items():
        ka, na, kb, nb = int(a[A].sum()), int(A.sum()), int(a[B].sum()), int(B.sum())
        per = {}
        for c in types:
            h = W["has"][c] & scored
            if h.any():
                per[str(classes[c])] = {"windows": int(h.sum()), "alarmed": int(a[h].sum()),
                                        "share": float(a[h].mean()), "ci95": cp(int(a[h].sum()), int(h.sum()))}
        out[name] = {"attack_windows": na, "attack_alarmed": ka, "tpr": ka / na if na else float("nan"),
                     "tpr_ci95": cp(ka, na), "benign_windows": nb, "false_alarms": kb,
                     "fpr": kb / nb if nb else float("nan"),
                     "fpr_hi95": cp(kb, nb)[1] if kb else (3 / nb if nb else float("nan")), "per_type": per}
    return out


def run(dkey, seed):
    dom = DOMS[dkey]
    z = np.load(os.path.join(ROOT, f"results/.cache/ev1_step1_{dkey}_s{seed}.npz"), allow_pickle=True)
    Ps, y, nc, classes = z["Ps"], z["y"].astype(np.int64), int(z["nc"]), [str(c) for c in z["classes"]]
    t = times(dom)
    assert len(t) == len(y), (len(t), len(y))
    o = np.argsort(t, kind="stable")
    t, fired, y = t[o], Ps[o] >= 0.5, y[o]
    del Ps
    types = [c for c in range(len(classes)) if c != nc and (y == c).any()]
    lines = [f"evaluation-v1 section 6, window alarms -- {dom['name']} -- seed {seed}; "
             f"{len(y):,} test rows ({int((y != nc).sum()):,} attack), {fired.shape[1]} specialists, OR at 0.5; "
             f"time span {(t[-1] - t[0]) / 3600:.2f} h",
             f"alarm: n_f >= ceil({MARGIN} max benign) + 1 OR fraction > {MARGIN} max benign OR any S_s >= "
             f"max(ceil({MARGIN} max benign S_s) + 1, {FLOOR}); blocked-{BLOCKS} out-of-sample calibration", ""]
    res = {"domain": dom["name"], "seed": seed, "rows": int(len(y)), "configs": []}
    per10 = None
    for mode, size, role in dom["windows"]:
        if mode == "time":
            w = np.floor((t - t[0]) / size).astype(np.int64)
            label = f"{size:g}-s windows"
            if size == 10.0:
                per10 = len(y) / len(np.unique(w))
        else:
            M = int(round(per10 if per10 else len(y) / len(np.unique(np.floor((t - t[0]) / size)))))
            w = np.arange(len(y)) // M
            label = f"count windows of M = {M} flows (mean per occupied 10-s window), attribution none"
        W = window_table(w, fired, y, nc, types)
        net, full, log, scored = evaluate(W)
        r = rates({"network-wide": net, "full rule": full}, W, types, classes, scored)
        med = float(np.median(W["n"]))
        lines.append(f"=== {label} ({role}): {int(W['atk'].sum()):,} attack + {int((~W['atk']).sum()):,} benign "
                     f"windows; samples per window median {med:.0f}, mean {W['n'].mean():.1f}"
                     + ("" if scored.all() else f"; NOT SCORED (no attack-free window in the other blocks): "
                        f"{int((~scored).sum()):,} windows, {int((W['atk'] & ~scored).sum()):,} attack"))
        if not (scored & ~W["atk"]).any() or not (scored & W["atk"]).any():
            lines.append("  no block could be calibrated and scored" if not scored.any() else
                         "  scored windows lack attack or benign windows")
        for name, q in r.items():
            fa = (f"{q['false_alarms']:,} of {q['benign_windows']:,} ({100 * q['fpr']:.3f} %)" if q["false_alarms"]
                  else f"0 of {q['benign_windows']:,} (95 % upper bound {100 * q['fpr_hi95']:.3f} %)")
            lines.append(f"  {name:<13} attack windows alarmed {q['attack_alarmed']:,}/{q['attack_windows']:,} = "
                         f"{100 * q['tpr']:.2f} % [{100 * q['tpr_ci95'][0]:.2f}, {100 * q['tpr_ci95'][1]:.2f}]; "
                         f"false alarms {fa}")
        lines.append("  per type, full rule (windows holding the type: alarmed / windows):")
        for c, p in sorted(r["full rule"]["per_type"].items(), key=lambda kv: kv[1]["share"]):
            lines.append(f"    {c:<34} {p['alarmed']:>6,}/{p['windows']:<6,} {100 * p['share']:6.2f} %"
                         f"  [{100 * p['ci95'][0]:.1f}, {100 * p['ci95'][1]:.1f}]")
        lines.append("  folds: " + "; ".join(
            f"{g['fold']}: {g['windows']} win ({g['attack']} attack), " +
            ("not calibrated (no attack-free window in the other blocks)" if g["T"] is None else
             f"{g['benign_calibration']} benign for calibration, T={g['T']}, f={g['f']:.4f}, "
             f"T_s {g['Ts_min']}-{g['Ts_max']}") for g in log))
        lines.append("")
        res["configs"].append({"window": label, "mode": mode, "size": size, "role": role,
                               "samples_median": med, "results": r, "folds": log})
    out = os.path.join(ROOT, f"results/ev1_windows_{dkey}_s{seed}")
    open(out + ".txt", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(out + ".json", "w"), indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domains", default="ereno,cicids,morris4")
    a = ap.parse_args()
    for d in a.domains.split(","):
        run(d, a.seed)
