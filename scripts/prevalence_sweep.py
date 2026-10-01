#!/usr/bin/env python3
"""Prevalence sweep on ERENO: how detection quality depends on how RARE attacks are.

Real traffic is overwhelmingly benign, and a balanced test set hides exactly the
difficulty that matters operationally (base-rate fallacy, Axelsson 2000). This script
evaluates ONE trained model at controlled benign:attack ratios r, from 1:1 to 1000:1.

What can and cannot move:
  * recall (TPR) and FPR are rates CONDITIONAL on the true class, so they do not depend on
    prevalence. They stay flat across r -- that is a check, not a finding;
  * precision does: precision(r) = TPR / (TPR + r * FPR). So do F1 and the operational
    number an analyst actually feels, false alarms per true alarm = r * FPR / TPR.
The sampled precision must agree with that closed form; the script reports both.

Model: the operating point used throughout (conf/scenarios/*.yaml): N=14 specialists,
2 per attack, combined-24 features, seed 42. Fusion rules k>=1 (OR), k>=2, k>=3, plus
average precision with the vote count (0..14) as the score -- the k-of-n ladder IS the
precision-recall curve.

Sampling: at each r, keep the larger class whole and subsample the other to reach r;
attacks are subsampled with proportional allocation per attack class, so the attack mix
is preserved. 20 draws per r give the spread.

  python scripts/prevalence_sweep.py
Out: results/prevalence_sweep_ereno.{csv,txt,png,pdf}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from scored_cache import get_scored

RATIOS = [1, 3, 10, 30, 100, 300, 1000]
KS = [1, 2, 3]
DRAWS = 20
SEED = 7
OUT = "results/prevalence_sweep_ereno"


def rates(votes, is_atk, k):
    pred = votes >= k
    tp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
    fn = int((~pred & is_atk).sum()); tn = int((~pred & ~is_atk).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "fpr": fpr, "f1": f1, "mcc": mcc,
            "fp_per_tp": fp / tp if tp else float("inf")}


def avg_precision(score, is_atk):
    """AP with ties handled by threshold (score is an integer vote count)."""
    order = np.argsort(-score, kind="stable")
    s, y = score[order], is_atk[order]
    tp_c = np.cumsum(y); fp_c = np.cumsum(~y)
    last = np.r_[np.diff(s) != 0, True]          # one point per distinct threshold
    tp_c, fp_c = tp_c[last], fp_c[last]
    P = y.sum()
    prec = tp_c / (tp_c + fp_c)
    rec = tp_c / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def sample(rng, b_idx, a_by_class, r):
    n_b, n_a = len(b_idx), sum(len(v) for v in a_by_class.values())
    if n_b / r <= n_a:                    # benign is the binding class: keep it whole
        nb, na = n_b, int(round(n_b / r))
    else:                                 # attacks are binding: keep them whole
        na, nb = n_a, int(round(n_a * r))
    bs = b_idx if nb == n_b else rng.choice(b_idx, nb, replace=False)
    parts = []
    for c, idx in a_by_class.items():     # proportional allocation keeps the attack mix
        m = int(round(na * len(idx) / n_a))
        if m:
            parts.append(idx if m >= len(idx) else rng.choice(idx, m, replace=False))
    return np.concatenate([bs] + parts)


def main():
    ts, fired, y, nc, cv, spec_attack, _ = get_scored()
    votes = fired.sum(axis=0).astype(np.int16)
    is_atk = y != nc
    b_idx = np.where(~is_atk)[0]
    a_by_class = {int(c): np.where(y == c)[0] for c in np.unique(y[is_atk])}
    n_b, n_a = len(b_idx), int(is_atk.sum())
    natural = n_b / n_a
    print(f"[sweep] test: {n_b:,} benign, {n_a:,} attack -> natural ratio {natural:.1f}:1")

    full = {k: rates(votes, is_atk, k) for k in KS}
    for k in KS:
        f = full[k]
        print(f"[sweep] full test k>={k}: TPR={f['recall']:.4f} FPR={f['fpr']:.5f} "
              f"precision={f['precision']:.4f}")

    rng = np.random.default_rng(SEED)
    rows = []
    points = sorted(set(RATIOS) | {round(natural, 1)})
    for r in points:
        is_nat = abs(r - natural) < 0.05
        draws = 1 if is_nat else DRAWS
        for d in range(draws):
            idx = np.arange(len(y)) if is_nat else sample(rng, b_idx, a_by_class, r)
            v, a = votes[idx], is_atk[idx]
            ap = avg_precision(v.astype(float), a)
            for k in KS:
                m = rates(v, a, k)
                tpr, fpr = full[k]["recall"], full[k]["fpr"]
                rows.append({"ratio": r, "natural": int(is_nat), "draw": d, "k": k,
                             "n_benign": int((~a).sum()), "n_attack": int(a.sum()),
                             **m, "ap": ap,
                             "precision_analytic": tpr / (tpr + r * fpr),
                             "fp_per_tp_analytic": r * fpr / tpr})

    os.makedirs("results", exist_ok=True)
    with open(f"{OUT}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- summary -------------------------------------------------------------
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"ERENO prevalence sweep — N=14 specialists, combined-24, seed 42; "
        f"{DRAWS} draws per ratio")
    out(f"test set: {n_b:,} benign / {n_a:,} attack (natural {natural:.1f}:1)")
    out("recall and FPR are prevalence-invariant (conditional rates); precision, F1 and "
        "false alarms per true alarm are not.")
    for k in KS:
        out()
        out(f"--- fusion k>={k}  (full test: recall {full[k]['recall']:.4f}, "
            f"FPR {100*full[k]['fpr']:.3f}%) ---")
        out(f"{'benign:attack':>14s} {'n_attack':>9s} {'precision':>18s} {'analytic':>9s} "
            f"{'F1':>16s} {'FP per TP':>16s} {'AP':>7s}")
        for r in points:
            sel = [x for x in rows if x["ratio"] == r and x["k"] == k]
            def ms(key):
                v = np.array([x[key] for x in sel], float)
                if len(v) == 1:
                    return f"{v[0]:.4f}"
                return f"{v.mean():.4f}±{v.std():.4f}"
            tag = " (natural)" if sel[0]["natural"] else ""
            out(f"{str(r) + ':1' + tag:>14s} {sel[0]['n_attack']:>9,} {ms('precision'):>18s} "
                f"{sel[0]['precision_analytic']:>9.4f} {ms('f1'):>16s} "
                f"{ms('fp_per_tp'):>16s} {np.mean([x['ap'] for x in sel]):>7.4f}")
    with open(f"{OUT}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.array(points, float)
    for k, col in zip(KS, ["#1f77b4", "#d62728", "#2ca02c"]):
        mean = [np.mean([x["precision"] for x in rows if x["ratio"] == r and x["k"] == k])
                for r in points]
        lo = [np.percentile([x["precision"] for x in rows if x["ratio"] == r and x["k"] == k], 2.5)
              for r in points]
        hi = [np.percentile([x["precision"] for x in rows if x["ratio"] == r and x["k"] == k], 97.5)
              for r in points]
        ana = [full[k]["recall"] / (full[k]["recall"] + r * full[k]["fpr"]) for r in xs]
        ax[0].plot(xs, mean, "o-", color=col, label=f"k≥{k} (recall {full[k]['recall']:.3f}, "
                                                    f"FPR {100*full[k]['fpr']:.2f}%)")
        ax[0].fill_between(xs, lo, hi, color=col, alpha=0.15)
        ax[0].plot(xs, ana, "--", color=col, lw=0.8)
        fpt = [np.mean([x["fp_per_tp"] for x in rows if x["ratio"] == r and x["k"] == k])
               for r in points]
        ax[1].plot(xs, fpt, "o-", color=col, label=f"k≥{k}")
    for a in ax:
        a.set_xscale("log"); a.axvline(natural, color="grey", ls=":", lw=1)
        a.set_xlabel("benign : attack ratio"); a.grid(alpha=0.3)
    ax[0].set_ylabel("precision"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title("Precision vs. attack rarity (dashed = TPR/(TPR+r·FPR))")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[1].set_yscale("log"); ax[1].set_ylabel("false alarms per true alarm")
    ax[1].axhline(1, color="black", lw=0.6)
    ax[1].set_title("Analyst workload"); ax[1].legend(fontsize=8)
    ax[0].text(natural * 1.08, 0.55, f"ERENO test\n{natural:.1f}:1", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=200); fig.savefig(f"{OUT}.pdf")
    print(f"\n-> {OUT}.{{csv,txt,png,pdf}}")


if __name__ == "__main__":
    main()
