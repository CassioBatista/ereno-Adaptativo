#!/usr/bin/env python3
"""Out-of-sample selection of the window-level triage thresholds.

The triage rules reported so far were chosen on the very windows they were scored on:
T=275 is the worst benign n_flags (274) + 1, and k_votes>=5 sits one vote above the
benign maximum. Their "zero false windows" is therefore in-sample, by construction. This
script applies the SAME calibration policy out of sample.

Calibration policy (uses BENIGN windows only -- attack windows never enter it):
    T = max benign n_flags + 1
    f = max benign flagged fraction (n_flags / window_samples), rule: fraction > f
    K = max benign k_votes + 1
plus a "margin" variant with headroom: T and f inflated by MARGIN, K one vote higher.

Rules evaluated:
    R1  n_flags >= T
    R2  R1 or fraction > f
    R3  R2 or k_votes >= K

Validation schemes:
  * blocked-5 (PRIMARY): 5 contiguous blocks of windows in time order; calibrate on 4,
    evaluate on the held-out one, rotate. Calibrate on the past, apply to the future,
    and every window is evaluated exactly once;
  * halves: the same with 2 blocks (first half -> second half, and back);
  * random-2 (REFERENCE, optimistic): 200 stratified random halves. It ignores time,
    so it borrows information across the trace's bursts.

What to expect: with a "max + 1" policy, an out-of-sample false window occurs exactly
when the held-out block contains a benign window more extreme than every calibration
one -- a record. Under exchangeability that has probability ~m/(n+m) per fold.

  python scripts/oos_thresholds.py
Out: results/oos_thresholds_ereno.{txt,csv}
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import beta

from scored_cache import get_scored

K_FUSION = 2
CAPS = [("all", 10 ** 9), ("500k", 500_000)]
MARGIN = 0.25
RATIOS = [10, 30, 100, 1000]
REPS = 200
OUT = "results/oos_thresholds_ereno"


def cp_upper(x, n, conf=0.95):
    return 1.0 if x >= n else float(beta.ppf(conf, x + 1, n - x))


def cp_lower(x, n, conf=0.95):
    return 0.0 if x <= 0 else float(beta.ppf(1 - conf, x, n - x + 1))


def windows(ts, votes, y, nc, cv):
    flag = votes >= K_FUSION
    t0 = float(ts[0])
    win = np.floor(ts - t0).astype(np.int64)
    bnd = np.searchsorted(win, np.arange(int(win[-1]) + 2))
    cs = np.r_[0, np.cumsum(flag)]
    W = []
    for w in range(int(win[-1]) + 1):
        lo, hi = int(bnd[w]), int(bnd[w + 1])
        if hi == lo:
            continue
        nf = int(cs[hi] - cs[lo])
        yy = y[lo:hi]
        atk = bool((yy != nc).any())
        W.append({"w": w, "n": hi - lo, "nf": nf, "frac": nf / (hi - lo),
                  "kv": int(votes[lo:hi][flag[lo:hi]].max()) if nf else 0,
                  "atk": atk,
                  "cls": cv[int(np.bincount(yy[yy != nc]).argmax())] if atk else "benign"})
    return W


def calibrate(benign, margin=0.0):
    mx_nf = max(r["nf"] for r in benign)
    mx_fr = max(r["frac"] for r in benign)
    mx_kv = max(r["kv"] for r in benign)
    if margin:
        return {"T": int(math.ceil(mx_nf * (1 + margin))) + 1,
                "f": mx_fr * (1 + margin), "K": mx_kv + 2}
    return {"T": mx_nf + 1, "f": mx_fr, "K": mx_kv + 1}


def rules(th):
    r1 = lambda r: r["nf"] >= th["T"]
    r2 = lambda r: r1(r) or r["frac"] > th["f"]
    r3 = lambda r: r2(r) or r["kv"] >= th["K"]
    return {"R1 volume": r1, "R2 +fraction": r2, "R3 +k_votes": r3}


def run_folds(W, folds, margin):
    """folds: list of index arrays (held-out sets). Returns pooled counts per rule."""
    pooled = {}
    fold_log = []
    for fi, held in enumerate(folds):
        held = set(held.tolist())
        cal_b = [W[i] for i in range(len(W)) if i not in held and not W[i]["atk"]]
        th = calibrate(cal_b, margin)
        ev = [W[i] for i in held]
        fold_log.append((fi, th, sum(not r["atk"] for r in ev), sum(r["atk"] for r in ev)))
        for name, fn in rules(th).items():
            p = pooled.setdefault(name, {"tp": 0, "fp": 0, "nA": 0, "nB": 0, "fp_windows": []})
            for r in ev:
                hit = fn(r)
                if r["atk"]:
                    p["nA"] += 1; p["tp"] += hit
                else:
                    p["nB"] += 1; p["fp"] += hit
                    if hit:
                        p["fp_windows"].append((fi, r["w"], r["nf"], round(r["frac"], 4), r["kv"]))
    return pooled, fold_log


def blocked(n, k):
    return np.array_split(np.arange(n), k)


def main():
    lines, csv_rows = [], []

    def out(s=""):
        print(s, flush=True); lines.append(s)

    out("Out-of-sample selection of window-level triage thresholds — ERENO, N=14, fusion k>=2")
    out("calibration uses benign windows only: T = max n_flags + 1, f = max fraction, "
        "K = max k_votes + 1;  margin variant: T,f x1.25 and K+1")
    for cname, cap in CAPS:
        ts, fired, y, nc, cv, spec_attack, _ = get_scored(benign_cap=cap)
        W = windows(ts, fired.sum(axis=0), y, nc, cv)
        nB = sum(not r["atk"] for r in W); nA = sum(r["atk"] for r in W)
        out()
        out(f"=================== benign_cap = {cname}  ({nB} benign + {nA} attack windows) "
            f"===================")
        ins = calibrate([r for r in W if not r["atk"]])
        out(f"in-sample thresholds (all windows): T={ins['T']}  f={ins['f']:.4f}  K={ins['K']}")

        schemes = [("blocked-5", [blocked(len(W), 5)]),
                   ("halves", [blocked(len(W), 2)])]
        rng = np.random.default_rng(11)
        idxB = np.array([i for i, r in enumerate(W) if not r["atk"]])
        idxA = np.array([i for i, r in enumerate(W) if r["atk"]])
        rand = []
        for _ in range(REPS):
            pb, pa = rng.permutation(idxB), rng.permutation(idxA)
            h1 = np.r_[pb[: len(pb) // 2], pa[: len(pa) // 2]]
            h2 = np.r_[pb[len(pb) // 2:], pa[len(pa) // 2:]]
            rand.append([h1, h2])
        schemes.append(("random-2 x200", rand))

        for margin in (0.0, MARGIN):
            mtag = "max+1" if margin == 0 else f"margin x{1 + margin:.2f}"
            out()
            out(f"--- policy: {mtag} ---")
            out(f"{'scheme':14s} {'rule':14s} {'TP':>9s} {'TPR':>6s} {'FP':>8s} {'FPR':>7s} "
                f"{'FPR 95% up':>10s} " + " ".join(f"{'cert@' + str(r) + ':1':>10s}" for r in RATIOS))
            for sname, fold_sets in schemes:
                agg = {}
                logs = []
                for folds in fold_sets:
                    pooled, flog = run_folds(W, folds, margin)
                    logs.append(flog)
                    for name, p in pooled.items():
                        a = agg.setdefault(name, {"tp": 0, "fp": 0, "nA": 0, "nB": 0, "fpw": []})
                        a["tp"] += p["tp"]; a["fp"] += p["fp"]; a["nA"] += p["nA"]; a["nB"] += p["nB"]
                        a["fpw"] += p["fp_windows"]
                for name, a in agg.items():
                    tpr = a["tp"] / a["nA"]; fpr = a["fp"] / a["nB"]
                    # certification uses one evaluation pass worth of windows (per repetition)
                    reps = len(fold_sets)
                    fp1, nB1 = a["fp"] / reps, a["nB"] / reps
                    tp1, nA1 = a["tp"] / reps, a["nA"] / reps
                    hi = cp_upper(int(round(fp1)), int(nB1)); lo = cp_lower(int(round(tp1)), int(nA1))
                    cert = [lo / (lo + r * hi) for r in RATIOS]
                    out(f"{sname:14s} {name:14s} {a['tp']:>4d}/{a['nA']:<4d} {tpr:6.3f} "
                        f"{a['fp']:>3d}/{a['nB']:<4d} {100 * fpr:6.2f}% {100 * hi:9.2f}% "
                        + " ".join(f"{c:10.3f}" for c in cert))
                    csv_rows.append({"cap": cname, "policy": mtag, "scheme": sname, "rule": name,
                                     "tp": a["tp"], "nA": a["nA"], "fp": a["fp"], "nB": a["nB"],
                                     "tpr": tpr, "fpr": fpr, "fpr_hi95": hi,
                                     **{f"cert_{r}": c for r, c in zip(RATIOS, cert)}})
                if sname != "random-2 x200":
                    for fi, th, nb, na in logs[0]:
                        out(f"{'':14s}   fold {fi}: {nb:3d} benign / {na:3d} attack held out -> "
                            f"T={th['T']}, f={th['f']:.4f}, K={th['K']}")
                    fpw = agg["R3 +k_votes"]["fpw"]
                    if fpw:
                        out(f"{'':14s}   out-of-sample false windows (fold, window, n_flags, "
                            f"fraction, k_votes): {fpw}")

    os.makedirs("results", exist_ok=True)
    with open(f"{OUT}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    with open(f"{OUT}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n-> {OUT}.{{txt,csv}}")


if __name__ == "__main__":
    main()
