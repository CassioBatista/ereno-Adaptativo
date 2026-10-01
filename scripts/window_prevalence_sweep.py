#!/usr/bin/env python3
"""Prevalence sweep at WINDOW level, the level at which ReSIDS actually alarms.

The sample-level sweep (scripts/prevalence_sweep.py) showed precision collapsing as
attacks become rare: at k>=2, false alarms outnumber true ones beyond ~150:1. But ReSIDS
does not alarm per sample. It aggregates per 1-s window and the event carries n_flags,
the alarm's volume, precisely so the monitor can triage. The question this answers: does
window-level triage hold precision up when attacks are rare?

Per window of the ERENO test trace (full pool of 14 specialists, fusion k>=2):
  n_flags = samples in the window passing k-of-n
A window is an ATTACK window if it holds any attack sample. For each triage threshold T,
the monitor raises an alarm iff n_flags >= T, giving a window-level TPR(T) and FPR(T).

Prevalence r = benign windows per attack window. As at sample level, TPR and FPR are
conditional rates and precision(r) = TPR / (TPR + r * FPR).

THE LIMIT, stated up front: the test trace has only ~724 benign windows (~12 min of
benign traffic; the trace is bursty). When a threshold produces ZERO false windows, FPR
is not zero -- it is merely below a one-sided 95% Clopper-Pearson bound (~3/724, the
rule of three). At extreme r that bound, not the point estimate, governs precision. So
for every T the script reports the precision the data can CERTIFY, and how many hours of
benign traffic would be needed to certify more.

Benign windows from the train split are deliberately NOT used to enlarge the pool: the
ERENO train/test are block-split and block leakage is documented in this project (random
CV F1 99% vs 41% on the real test), so a train-block FPR would be optimistic.

  python scripts/window_prevalence_sweep.py
Out: results/window_prevalence_sweep_ereno.{csv,txt,png,pdf}
"""
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import beta

from scored_cache import get_scored

K = 2
WINDOW_S = 1.0
THRESHOLDS = [1, 2, 5, 10, 20, 56, 100, 150, 200, 275, 400, 600, 973]
RATIOS = [1, 3, 10, 30, 100, 300, 1000]
SHOW_T = [1, 10, 56, 275]
CONF = 0.95
OUT = "results/window_prevalence_sweep_ereno"


def cp_upper(x, n, conf=CONF):
    """One-sided Clopper-Pearson upper bound."""
    return 1.0 if x >= n else float(beta.ppf(conf, x + 1, n - x))


def cp_lower(x, n, conf=CONF):
    return 0.0 if x <= 0 else float(beta.ppf(1 - conf, x, n - x + 1))


def main():
    ts, fired, y, nc, cv, spec_attack, cache = get_scored()
    votes = fired.sum(axis=0)
    flag = votes >= K
    t0 = float(ts[0])
    win = np.floor((ts - t0) / WINDOW_S).astype(np.int64)
    bnd = np.searchsorted(win, np.arange(int(win[-1]) + 2))
    cs = np.r_[0, np.cumsum(flag)]

    rows = []
    for w in range(int(win[-1]) + 1):
        lo, hi = int(bnd[w]), int(bnd[w + 1])
        if hi == lo:
            continue
        yy = y[lo:hi]
        atk = yy != nc
        nf = int(cs[hi] - cs[lo])
        kv = int(votes[lo:hi][flag[lo:hi]].max()) if nf else 0
        cls = (Counter(yy[atk].tolist()).most_common(1)[0][0] if atk.any() else nc)
        rows.append({"window": w, "n_samples": hi - lo, "attack": int(atk.any()),
                     "attack_samples": int(atk.sum()), "class": cv[cls] if cls < len(cv) else str(cls),
                     "n_flags": nf, "k_votes": kv})
    A = [r for r in rows if r["attack"]]
    B = [r for r in rows if not r["attack"]]
    nA, nB = len(A), len(B)
    nat = nB / nA
    nfA = np.array([r["n_flags"] for r in A]); nfB = np.array([r["n_flags"] for r in B])

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"ERENO window-level prevalence sweep — 1-s windows, N=14, fusion k>={K}, full pool")
    out(f"non-empty windows: {len(rows):,} = {nB} benign + {nA} attack "
        f"(natural window ratio {nat:.2f}:1)")
    out(f"benign windows with any alarm (n_flags>=1): {int((nfB >= 1).sum())} "
        f"({100 * (nfB >= 1).mean():.1f}%); worst benign n_flags = {int(nfB.max())}")
    out(f"benign n_flags median (alarming windows only) = "
        f"{int(np.median(nfB[nfB >= 1])) if (nfB >= 1).any() else 0}; "
        f"attack n_flags median = {int(np.median(nfA))}")
    out()

    # ---- per-threshold rates with certification bounds -----------------------
    table = []
    out(f"{'T':>5s} {'TPR':>7s} {'TP':>5s} {'FP':>4s} {'FPR':>8s} {'FPR 95% up':>11s} "
        f"{'r_cert(p>=.9)':>14s} {'hrs to cert p>=.9 @1000:1':>27s}")
    for T in THRESHOLDS:
        tp = int((nfA >= T).sum()); fp = int((nfB >= T).sum())
        tpr = tp / nA; fpr = fp / nB
        fpr_hi = cp_upper(fp, nB); tpr_lo = cp_lower(tp, nA)
        # largest r at which precision >= 0.9 is certified by THIS data
        r_cert = tpr_lo * (1 - 0.9) / (0.9 * fpr_hi) if fpr_hi > 0 else float("inf")
        # benign windows needed, with zero false windows, to certify p>=0.9 at r=1000
        need_fpr = tpr * (1 - 0.9) / (0.9 * 1000) if tpr > 0 else float("nan")
        need_win = -np.log(1 - CONF) / need_fpr if need_fpr and need_fpr > 0 else float("nan")
        hrs = need_win * WINDOW_S / 3600
        table.append({"T": T, "tp": tp, "fp": fp, "tpr": tpr, "tpr_lo": tpr_lo, "fpr": fpr,
                      "fpr_hi": fpr_hi, "r_cert_p90": r_cert,
                      "benign_windows_to_cert_p90_at_1000": need_win, "hours": hrs})
        out(f"{T:>5d} {tpr:7.3f} {tp:5d} {fp:4d} {100 * fpr:7.3f}% {100 * fpr_hi:10.3f}% "
            f"{r_cert:14.1f} {hrs:27.1f}")
    out()

    # ---- precision vs r, estimate and certified lower bound --------------------
    csv_rows = []
    out("precision by prevalence (estimate | certified lower bound, 95%)")
    out(f"{'T':>5s} " + " ".join(f"{str(r) + ':1':>17s}" for r in RATIOS))
    for t in table:
        cells = []
        for r in RATIOS:
            est = t["tpr"] / (t["tpr"] + r * t["fpr"]) if t["tpr"] + r * t["fpr"] else float("nan")
            low = t["tpr_lo"] / (t["tpr_lo"] + r * t["fpr_hi"]) if t["tpr_lo"] + r * t["fpr_hi"] else 0.0
            cells.append(f"{est:7.3f} | {low:6.3f}")
            csv_rows.append({"T": t["T"], "ratio": r, "tpr": t["tpr"], "fpr": t["fpr"],
                             "fpr_hi95": t["fpr_hi"], "precision": est,
                             "precision_certified_lo95": low,
                             "false_alarms_per_true": (r * t["fpr"] / t["tpr"]) if t["tpr"] else float("inf")})
        out(f"{t['T']:>5d} " + " ".join(f"{c:>17s}" for c in cells))
    out()

    # ---- which attack windows does triage drop? --------------------------------
    T0 = 275
    miss = [r for r in A if r["n_flags"] < T0]
    out(f"attack windows below T={T0}: {len(miss)} of {nA} ({100 * len(miss) / nA:.1f}%)")
    by = Counter(r["class"] for r in miss)
    tot = Counter(r["class"] for r in A)
    for c in sorted(tot, key=lambda c: -by.get(c, 0)):
        out(f"  {c:26s} missed {by.get(c, 0):3d} / {tot[c]:3d}")
    sm = np.array([r["n_samples"] for r in miss]) if miss else np.array([0])
    sk = np.array([r["n_samples"] for r in A if r["n_flags"] >= T0])
    out(f"  window size of the missed: median {int(np.median(sm))} samples "
        f"(caught: median {int(np.median(sk))}) — sparse windows, not stealth, if these differ")

    # ---- volume is not the only signal the event carries ------------------------
    # The windows missed by absolute volume are almost all SPARSE (1-2 samples): n_flags
    # cannot reach T when the window holds fewer than T samples. The event carries
    # window_samples as the denominator, and k_votes as corroboration strength, so the
    # monitor can also triage on the flagged FRACTION and on k_votes.
    kb = max((r["k_votes"] for r in B if r["n_flags"] > 0), default=0)
    fb = max((r["n_flags"] / r["n_samples"] for r in B if r["n_flags"] > 0), default=0.0)
    out()
    out(f"benign alarming windows: max flagged fraction {fb:.4f}, max k_votes {kb}")
    rules = [
        ("volume  n_flags>=275", lambda r: r["n_flags"] >= 275),
        ("+ fraction>=0.05", lambda r: r["n_flags"] >= 275 or r["n_flags"] / r["n_samples"] >= 0.05),
        ("+ k_votes>=5", lambda r: (r["n_flags"] >= 275 or r["n_flags"] / r["n_samples"] >= 0.05
                                    or r["k_votes"] >= 5)),
    ]
    for f in (0.05, 0.1, 0.25, 0.5):
        tp = sum(1 for r in A if r["n_flags"] >= 275 or r["n_flags"] / r["n_samples"] >= f)
        fp = sum(1 for r in B if r["n_flags"] >= 275 or r["n_flags"] / r["n_samples"] >= f)
        out(f"  fraction threshold {f:4.2f}: TP {tp}/{nA}  FP {fp}/{nB}   (stability of the rule)")
    out(f"{'rule':24s} {'TP':>8s} {'FP':>6s} " + " ".join(f"{'cert@' + str(r) + ':1':>11s}" for r in RATIOS))
    for name, fn in rules:
        tp = sum(1 for r in A if fn(r)); fp = sum(1 for r in B if fn(r))
        lo = cp_lower(tp, nA); hi = cp_upper(fp, nB)
        cert = [lo / (lo + r * hi) for r in RATIOS]
        out(f"{name:24s} {tp:>4d}/{nA} {fp:>3d}/{nB} " + " ".join(f"{c:11.3f}" for c in cert))
    left = [r for r in A if not rules[-1][1](r)]
    out(f"still missed by all three: {len(left)} — attack diluted inside full benign windows:")
    for r in left:
        out(f"  {r['class']:22s} {r['attack_samples']:4d} attack samples in {r['n_samples']:5d}  "
            f"n_flags {r['n_flags']:4d}  k_votes {r['k_votes']}")
    out("CAVEAT: T=275 is the worst benign n_flags + 1, chosen on these same 724 windows, so its "
        "0 FP is in-sample. The fraction rule is not knife-edge (benign max "
        f"{fb:.3f} vs 1.0 on sparse attacks, stable from 0.05 to 0.5); k_votes>=5 is (benign max {kb}).")

    # sample-level reference at the same k, for the figure
    is_atk = y != nc
    s_tpr = float((flag & is_atk).sum() / is_atk.sum())
    s_fpr = float((flag & ~is_atk).sum() / (~is_atk).sum())
    out()
    out(f"sample-level reference (k>={K}): TPR {s_tpr:.4f}, FPR {100 * s_fpr:.3f}%")

    os.makedirs("results", exist_ok=True)
    with open(f"{OUT}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader(); w.writerows(csv_rows)
    with open(f"{OUT}_windows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(f"{OUT}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- figure ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    Ts = np.array(THRESHOLDS, float)
    ax[0].plot(Ts, [t["tpr"] for t in table], "o-", color="#1f77b4", label="TPR (attack windows)")
    ax[0].plot(Ts, [t["fpr"] for t in table], "s-", color="#d62728", label="FPR (benign windows)")
    ax[0].plot(Ts, [t["fpr_hi"] for t in table], "s:", color="#d62728", alpha=0.6,
               label=f"FPR, one-sided 95% upper bound (n={nB})")
    ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=1e-3)
    ax[0].set_ylim(0, 1.5)          # rates are non-negative; symlog otherwise wastes half the axis
    ax[0].set_xlabel("triage threshold T (alarm iff n_flags ≥ T)")
    ax[0].set_title("Window-level rates vs triage threshold")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, loc="center left")
    xs = np.array(RATIOS, float)
    ax[1].plot(xs, s_tpr / (s_tpr + xs * s_fpr), "k--", lw=1, label=f"sample level, k≥{K}")
    for T, col in zip(SHOW_T, ["#7f7f7f", "#ff7f0e", "#9467bd", "#2ca02c"]):
        t = next(t for t in table if t["T"] == T)
        est = t["tpr"] / (t["tpr"] + xs * t["fpr"])
        low = t["tpr_lo"] / (t["tpr_lo"] + xs * t["fpr_hi"])
        ax[1].plot(xs, est, "o-", color=col, label=f"window, T={T} (FP {t['fp']}/{nB})")
        ax[1].fill_between(xs, low, est, color=col, alpha=0.15)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.3)
    ax[1].set_xlabel("benign : attack ratio"); ax[1].set_ylabel("precision")
    ax[1].set_title("Precision: estimate (line) and certified lower bound (band)")
    ax[1].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=200); fig.savefig(f"{OUT}.pdf")
    print(f"\n-> {OUT}.{{csv,txt,png,pdf}} and {OUT}_windows.csv")


if __name__ == "__main__":
    main()
