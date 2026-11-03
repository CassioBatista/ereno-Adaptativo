#!/usr/bin/env python3
"""Fusion rule selection on ERENO, as pre-registered in docs/FUSION_RULE.md.

Candidates R0 (OR @0.5), R1 (k >= 2 @0.5), R2 (max score, one calibrated threshold),
R3 (per-specialist calibrated thresholds, OR), R4 (pair agreement, one calibrated
threshold). Thresholds are calibrated on held-out benign samples (the 20 % of each
specialist's partition never used for training) to a fused false-positive rate of 0.1 %;
the ERENO test file is then used once. Scenarios: Full (14 specialists), Absence (one
specialist per attack, recalibrated), Silent fault (second specialist of each pair outputs
0, thresholds of Full). Hard constraints H1-H3 and the decision rule are those of the
document.

Two processes (memory):
  python scripts/fusion_rule_selection.py calib   # held-out rows -> calibration scores cache
  python scripts/fusion_rule_selection.py         # selection
Needs the specialists and test scores of scripts/k_fusion_audit.py (train, score).
Out: results/fusion_rule_selection.txt; calibration scores in results/.cache/ (gitignored)
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import k_fusion_audit as K

TARGET = 0.001          # fused false-positive rate on the calibration benign
H_KEEP = 0.90           # H1, H2: share of the Full recall each attack must keep
H_FPR = 2 * TARGET      # H3: test false-positive rate bound
TIE = 0.5               # percentage points
SIMPLICITY = ["R2", "R3", "R4", "R0", "R1"]
CALIB = "results/.cache/ereno_probs_n14_calib.npz"


def phase_calib():
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    import python.util as util
    from fd.dataset import load_and_partition
    util.load_arff = K.lean_load_arff
    parts, _, _ = load_and_partition(f"{K.DATASET}.csv", K.FEATURES, K.N, seed=K.SEED,
                                     partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    Xs, ys = [], []
    for X_c, y_c in parts:
        _, Xcal, _, ycal = train_test_split(X_c, y_c, test_size=0.2, random_state=K.SEED)
        Xs.append(Xcal)
        ys.append(ycal)
    X, y = np.concatenate(Xs), np.concatenate(ys).astype(np.int16)
    d = xgb.DMatrix(X)
    P = []
    for i in range(K.N):
        b = xgb.Booster()
        b.load_model(f"{K.MODELS}/spec{i:02d}.json")
        P.append(b.predict(d).astype(np.float32))
    np.savez_compressed(CALIB, P=np.column_stack(P), y=y, nc=nc)
    print(f"[calib] {len(y):,} held-out rows, {int((y == nc).sum()):,} benign -> {CALIB}", flush=True)


def qthr(score, rate):
    """Threshold t such that a share `rate` of the scores is > t."""
    return float(np.quantile(score, 1 - rate, method="higher"))


def calibrate(rule, Pc, spec, present):
    idx = np.where(present)[0]
    if rule in ("R0", "R1"):
        return None
    if rule == "R2":
        return qthr(Pc[:, idx].max(1), TARGET)
    if rule == "R3":
        return {i: qthr(Pc[:, i], TARGET / len(idx)) for i in idx}
    if rule == "R4":
        return qthr(pair_score(Pc, spec, present), TARGET)


def pair_score(P, spec, present):
    """Per attack, the minimum over its present specialists; then the maximum over attacks."""
    cols = []
    for a in np.unique(spec):
        idx = [i for i in np.where(spec == a)[0] if present[i]]
        if idx:
            cols.append(P[:, idx].min(1))
    return np.max(np.column_stack(cols), axis=1)


def alarm(rule, P, spec, present, thr):
    idx = np.where(present)[0]
    if rule == "R0":
        return (P[:, idx] > 0.5).any(1)
    if rule == "R1":
        return (P[:, idx] > 0.5).sum(1) >= 2
    if rule == "R2":
        return P[:, idx].max(1) > thr
    if rule == "R3":
        return np.column_stack([P[:, i] > thr[i] for i in idx]).any(1)
    if rule == "R4":
        return pair_score(P, spec, present) > thr


def evaluate(pred, y, nc, attacks):
    atk = y != nc
    tp, fp = int((pred & atk).sum()), int((pred & ~atk).sum())
    rec, fpr = tp / atk.sum(), fp / (~atk).sum()
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    per = {a: float(pred[y == a].mean()) for a in attacks}
    return {"recall": rec, "fpr": fpr, "precision": prec, "f1": f1, "fp": fp, "per": per}


def main():
    P, y, nc, spec, cv = K.load()
    z = np.load(CALIB)
    Pc_all, yc = z["P"], z["y"]
    Pc = Pc_all[yc == nc]                     # calibration uses benign samples only
    spec = np.asarray(spec)
    attacks = [int(a) for a in np.unique(spec)]
    name = {a: cv[a] for a in attacks}
    first = np.zeros(K.N, bool)
    for a in attacks:
        first[np.where(spec == a)[0].min()] = True
    full = np.ones(K.N, bool)
    print(f"test {len(y):,} samples ({int((y != nc).sum()):,} attack); calibration benign "
          f"{len(Pc):,} held-out samples; target fused FPR {100 * TARGET:.2f}%")
    res = {}
    for rule in ["R0", "R1", "R2", "R3", "R4"]:
        thr_full = calibrate(rule, Pc, spec, full)
        thr_abs = calibrate(rule, Pc, spec, first)
        Ps = P.copy()
        Ps[:, ~first] = 0.0                     # silent second specialist of each pair
        r = {
            "Full": evaluate(alarm(rule, P, spec, full, thr_full), y, nc, attacks),
            "Absence": evaluate(alarm(rule, P, spec, first, thr_abs), y, nc, attacks),
            "Silent": evaluate(alarm(rule, Ps, spec, full, thr_full), y, nc, attacks),
        }
        del Ps
        cal_fpr = alarm(rule, Pc, spec, full, thr_full).mean()
        h1 = all(r["Absence"]["per"][a] >= H_KEEP * r["Full"]["per"][a] for a in attacks)
        h2 = all(r["Silent"]["per"][a] >= H_KEEP * r["Full"]["per"][a] for a in attacks)
        h3 = r["Full"]["fpr"] <= H_FPR
        res[rule] = {"scen": r, "H1": h1, "H2": h2, "H3": h3, "cal_fpr": float(cal_fpr),
                     "thr": thr_full if not isinstance(thr_full, dict) else
                     {int(k): v for k, v in thr_full.items()}}
        print(f"\n=== {rule} === calibration-benign FPR {100 * cal_fpr:.3f}%   "
              f"H1 {'pass' if h1 else 'FAIL'}  H2 {'pass' if h2 else 'FAIL'}  H3 {'pass' if h3 else 'FAIL'}")
        if thr_full is not None and not isinstance(thr_full, dict):
            print(f"  threshold (Full) {thr_full:.4f}")
        for s, e in r.items():
            print(f"  {s:<8} F1 {100 * e['f1']:6.2f}  recall {100 * e['recall']:6.2f}  precision "
                  f"{100 * e['precision']:6.2f}  FPR {100 * e['fpr']:.4f}% (FP {e['fp']:,})")
            print("           per attack: " + "  ".join(f"{name[a][:14]} {100 * v:5.1f}" for a, v in e["per"].items()))

    print("\n=== DECISION (docs/FUSION_RULE.md, section 5) ===")
    ok = [k for k, v in res.items() if v["H1"] and v["H2"] and v["H3"]]
    if ok:
        best = max(100 * res[k]["scen"]["Full"]["recall"] for k in ok)
        tied = [k for k in ok if best - 100 * res[k]["scen"]["Full"]["recall"] <= TIE]
        chosen = min(tied, key=SIMPLICITY.index)
        print(f"rules meeting H1-H3: {ok}; best Full recall {best:.2f}%; tied within {TIE} pp: {tied}")
    else:
        nmet = {k: v["H1"] + v["H2"] + v["H3"] for k, v in res.items()}
        top = max(nmet.values())
        cand = [k for k in res if nmet[k] == top]
        best = max(res[k]["scen"]["Full"]["recall"] for k in cand)
        tied = [k for k in cand if 100 * (best - res[k]["scen"]["Full"]["recall"]) <= TIE]
        chosen = min(tied, key=SIMPLICITY.index)
        print(f"no rule meets H1-H3; most constraints met ({top}): {cand}")
    print(f"CHOSEN RULE: {chosen}")
    json.dump({"chosen": chosen, "rules": {k: {kk: vv for kk, vv in v.items() if kk != "scen"} |
                                           {"scen": {s: {m: e[m] for m in ("recall", "fpr", "f1", "fp")}
                                                     for s, e in v["scen"].items()}}
                                           for k, v in res.items()}},
              open("results/fusion_rule_selection.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    if len(sys.argv) > 1 and sys.argv[1] == "calib":
        phase_calib()
        sys.exit(0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/fusion_rule_selection.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
