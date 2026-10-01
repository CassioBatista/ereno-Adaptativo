#!/usr/bin/env python3
"""Does the training-time benign cap cause the false positives that k>=2 fails to filter?

At N=14 (two specialists per attack), k>=2 barely moves the FPR (0.686% -> 0.666%): the
two specialists of an attack fire on the SAME benign samples. Three explanations compete:

  (a) the cap: with benign_cap = 500k split over 14 specialists, each sees ~36k benign
      samples of ~2.77M available, too little of normal variability, so they all
      misfire on the same under-represented benign regions;
  (b) training imbalance per specialist (positives vs benign), absorbed by
      scale_pos_weight but possibly not fully;
  (c) class overlap: some benign samples genuinely look like a given attack, so ANY
      specialist of that attack fires on them -- systematic, not decorrelatable
      (cf. the masquerade split: FP Jaccard 0.99 on disjoint training data).

The sweep trains the same operating point (N=14, combined-24, seed 42) with the cap at
500k (current), 1M and ALL benign, and measures:
  * per-specialist FPR and own-class recall, against its training imbalance;
  * fusion FPR/TPR for k>=1, 2, 3 on the full test (all 2.75M benign);
  * FP-set Jaccard between specialists: same-attack pairs vs cross-attack pairs. If the
    same-attack Jaccard stays near 1 with all benign, (c) is the cause, not (a);
  * window level: benign windows that alarm, worst benign n_flags, and the triage rules.

The test set is unaffected by the cap: it always uses every benign sample.

  python scripts/benign_cap_sweep.py
Out: results/benign_cap_sweep_ereno.{csv,txt,png,pdf}
"""
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from scored_cache import get_scored

CAPS = [("500k", 500_000), ("1M", 1_000_000), ("all", 10 ** 9)]
KS = [1, 2, 3]
OUT = "results/benign_cap_sweep_ereno"


def window_metrics(ts, flag, votes, y, nc):
    t0 = float(ts[0])
    win = np.floor(ts - t0).astype(np.int64)
    bnd = np.searchsorted(win, np.arange(int(win[-1]) + 2))
    cs = np.r_[0, np.cumsum(flag)]
    B, A = [], []
    for w in range(int(win[-1]) + 1):
        lo, hi = int(bnd[w]), int(bnd[w + 1])
        if hi == lo:
            continue
        nf = int(cs[hi] - cs[lo])
        kv = int(votes[lo:hi][flag[lo:hi]].max()) if nf else 0
        rec = (nf, hi - lo, kv)
        (A if (y[lo:hi] != nc).any() else B).append(rec)
    worst = max(nf for nf, _, _ in B)
    rule = lambda nf, n, kv, T: nf >= T or nf / n >= 0.05
    return {
        "benign_windows": len(B), "attack_windows": len(A),
        "benign_alarming": sum(1 for nf, _, _ in B if nf > 0),
        "worst_benign_nflags": worst,
        "benign_kvotes_max": max(kv for _, _, kv in B),
        "tp_T275": sum(1 for nf, _, _ in A if nf >= 275),
        "fp_T275": sum(1 for nf, _, _ in B if nf >= 275),
        "tp_rule275": sum(1 for r in A if rule(*r, 275)),
        "fp_rule275": sum(1 for r in B if rule(*r, 275)),
    }


def main():
    lines, rows, spec_rows = [], [], []

    def out(s=""):
        print(s, flush=True); lines.append(s)

    res = {}
    for name, cap in CAPS:
        ts, fired, y, nc, cv, spec_attack, path = get_scored(benign_cap=cap)
        is_b = y == nc
        n_b = int(is_b.sum())
        votes = fired.sum(axis=0)
        N = fired.shape[0]

        # per-specialist
        Fb = fired[:, is_b]
        fp_counts = Fb.sum(axis=1)
        for i in range(N):
            at = spec_attack[i]
            own = y == at
            spec_rows.append({"cap": name, "specialist": i, "attack": cv[at],
                              "fpr": fp_counts[i] / n_b,
                              "own_recall": float(fired[i, own].mean()),
                              "fp": int(fp_counts[i])})

        # FP Jaccard
        Ff = Fb.astype(np.float32)
        inter = Ff @ Ff.T
        union = fp_counts[:, None] + fp_counts[None, :] - inter
        jac = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        same, cross = [], []
        for i in range(N):
            for j in range(i + 1, N):
                (same if spec_attack[i] == spec_attack[j] else cross).append(jac[i, j])
        del Ff, Fb

        fus = {}
        for k in KS:
            pred = votes >= k
            fus[k] = {"tpr": float(pred[~is_b].mean()), "fpr": float(pred[is_b].mean()),
                      "fp": int(pred[is_b].sum())}
        wm = window_metrics(ts, votes >= 2, votes, y, nc)
        res[name] = {"fus": fus, "same": same, "cross": cross, "wm": wm,
                     "per_spec_benign": None}
        rows.append({"cap": name, **{f"k{k}_{m}": fus[k][m] for k in KS for m in ("tpr", "fpr", "fp")},
                     "jac_same_mean": float(np.nanmean(same)), "jac_same_min": float(np.nanmin(same)),
                     "jac_cross_mean": float(np.nanmean(cross)), "jac_cross_max": float(np.nanmax(cross)),
                     **wm})
        del fired

    # ---------------- report ----------------
    out("ERENO benign_cap sweep — N=14 (2 specialists/attack), combined-24, seed 42")
    out("test set identical across caps: all benign samples of the test split")
    out()
    out(f"{'cap':>5s} {'k>=1 FPR':>9s} {'k>=2 FPR':>9s} {'k>=3 FPR':>9s} "
        f"{'k>=2 TPR':>9s} {'k>=3 TPR':>9s} {'k2/k1 FP':>9s} {'Jac same':>9s} {'Jac cross':>10s}")
    for r in rows:
        out(f"{r['cap']:>5s} {100 * r['k1_fpr']:8.3f}% {100 * r['k2_fpr']:8.3f}% "
            f"{100 * r['k3_fpr']:8.4f}% {r['k2_tpr']:9.4f} {r['k3_tpr']:9.4f} "
            f"{r['k2_fp'] / max(r['k1_fp'], 1):9.3f} {r['jac_same_mean']:9.3f} {r['jac_cross_mean']:10.4f}")
    out("  k2/k1 FP = fraction of OR false positives that survive k>=2 (1.0 = k>=2 filters nothing)")
    out()
    out("window level (k>=2, 1-s windows)")
    out(f"{'cap':>5s} {'benign alarming':>16s} {'worst benign n_flags':>21s} "
        f"{'max benign k_votes':>19s} {'T=275 TP/FP':>12s} {'T=275|frac TP/FP':>17s}")
    for r in rows:
        out(f"{r['cap']:>5s} {r['benign_alarming']:>7d}/{r['benign_windows']:<8d} "
            f"{r['worst_benign_nflags']:>21d} {r['benign_kvotes_max']:>19d} "
            f"{r['tp_T275']:>5d}/{r['fp_T275']:<6d} {r['tp_rule275']:>8d}/{r['fp_rule275']:<8d}")
    out()
    out("per specialist FPR (%) by cap, pairs of the same attack side by side")
    caps = [c for c, _ in CAPS]
    out(f"{'attack':24s} " + " ".join(f"{c + ' A':>9s} {c + ' B':>9s}" for c in caps))
    attacks = []
    for s in spec_rows:
        if s["attack"] not in attacks:
            attacks.append(s["attack"])
    for at in attacks:
        cells = []
        for c in caps:
            pair = [s for s in spec_rows if s["cap"] == c and s["attack"] == at]
            cells += [f"{100 * p['fpr']:9.3f}" for p in pair]
        out(f"{at:24s} " + " ".join(cells))
    out()
    out("own-class recall by cap (mean of the pair)")
    for at in attacks:
        cells = []
        for c in caps:
            pair = [s for s in spec_rows if s["cap"] == c and s["attack"] == at]
            cells.append(f"{np.mean([p['own_recall'] for p in pair]):.4f}")
        out(f"  {at:24s} " + "  ".join(f"{c}={v}" for c, v in zip(caps, cells)))

    os.makedirs("results", exist_ok=True)
    with open(f"{OUT}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(f"{OUT}_specialists.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(spec_rows[0])); w.writeheader(); w.writerows(spec_rows)
    with open(f"{OUT}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(caps))
    for k, col in zip(KS, ["#1f77b4", "#d62728", "#2ca02c"]):
        ax[0].plot(x, [100 * r[f"k{k}_fpr"] for r in rows], "o-", color=col, label=f"k≥{k}")
    ax[0].set_yscale("log"); ax[0].set_ylabel("sample FPR (%)"); ax[0].set_title("Fusion FPR vs benign cap")
    ax[1].plot(x, [r["jac_same_mean"] for r in rows], "o-", color="#d62728", label="same-attack pairs (mean)")
    ax[1].plot(x, [r["jac_cross_mean"] for r in rows], "o-", color="#1f77b4", label="cross-attack pairs (mean)")
    ax[1].set_ylim(0, 1.02); ax[1].set_ylabel("Jaccard of FP sets"); ax[1].set_title("Are the errors shared?")
    for at in attacks:
        ax[2].plot(x, [100 * np.mean([s["fpr"] for s in spec_rows if s["cap"] == c and s["attack"] == at])
                       for c in caps], "o-", label=at)
    ax[2].set_yscale("log"); ax[2].set_ylabel("specialist FPR (%), pair mean"); ax[2].set_title("Per attack")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([f"cap {c}" for c in caps]); a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=200); fig.savefig(f"{OUT}.pdf")
    print(f"\n-> {OUT}.{{csv,txt,png,pdf}} and {OUT}_specialists.csv")


if __name__ == "__main__":
    main()
