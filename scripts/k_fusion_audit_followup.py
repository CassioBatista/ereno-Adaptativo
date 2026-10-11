#!/usr/bin/env python3
"""Follow-up of k_fusion_audit.py on its cached probabilities (no retraining).

(a) Recall is ~99.96% for every option, so options are compared by the false positives
    each needs to reach a given recall (all thresholds tuned alike, in sample).
(b) FL shrinking: true cost of ABSENCE = recall lost with k = 1 (other specialists may still
    catch an attack whose own specialists left); VETO = recall k = 1 keeps and k >= 2 loses.
    Per-attack recall shows whether whole categories are lost.
(c) Where the false positives of the reference come from (pairs firing together).

  python scripts/k_fusion_audit_followup.py
Out: results/k_fusion_audit_followup.txt
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np

CACHE = "results/.cache/ereno_probs_n14_k2audit.npz"


def main():
    z = np.load(CACHE)
    P, y, nc, spec, cv = z["P"], z["y"], int(z["nc"]), z["spec"], list(z["cv"])
    atk = y != nc
    classes = sorted(set(spec.tolist()))
    pairs = {c: np.where(spec == c)[0] for c in classes}
    nm = lambda c: cv[c][:16]

    # ---------------- (a) false positives needed for a target recall
    def fp_at_recall(score, target):
        a = np.sort(score[atk])                      # attack scores ascending
        k = int(np.floor((1 - target) * len(a)))     # may miss at most k attacks
        thr = a[k] if k < len(a) else np.inf         # flag score >= thr keeps >= target
        return int((score[~atk] >= thr).sum()), thr

    opts = {
        "one per attack (half 0), k=1": P[:, [pairs[c][0] for c in classes]].max(1),
        "one per attack (half 1), k=1": P[:, [pairs[c][1] for c in classes]].max(1),
        "mean of the pair, k=1": np.column_stack([P[:, pairs[c]].mean(1) for c in classes]).max(1),
        "min of the pair (= k>=2 within the pair)": np.column_stack([P[:, pairs[c]].min(1) for c in classes]).max(1),
        "all 14, k=1 (max)": P.max(1),
        "all 14, k>=2 (2nd highest score)": np.sort(P, axis=1)[:, -2],
    }
    print("=== (a) false positives needed to reach a recall (thresholds tuned alike) ===")
    print(f"  benign samples: {int((~atk).sum()):,}; attack samples: {int(atk.sum()):,}")
    targets = (0.999, 0.9995, 0.9996)
    print("  " + f"{'option':<44}" + "".join(f"  FP@rec {100 * t:.2f}%" for t in targets))
    for k, s in opts.items():
        print("  " + f"{k:<44}" + "".join(f"  {fp_at_recall(s, t)[0]:>14,}" for t in targets))
    print("  (k>=2 over the 14 = flag when the 2nd-highest score passes the threshold;"
          " 'min of the pair' = both specialists of one attack agree)")

    # ---------------- (b) FL shrinking: absence vs veto, per attack
    V = P >= 0.5
    rec_gl = (V.sum(1) >= 2)[atk].mean() * 100
    print("\n=== (b) FL keeps the first n nodes (k at 0.5): absence vs veto ===")
    print(f"  GL union k>=2 recall {rec_gl:.2f}% at every n (retained union)")
    print("   n | rec k>=2 | rec k=1 | absence = GL - k=1 | veto = k=1 - k>=2 | per-attack recall k=1 / k>=2")
    for n in range(14, 2, -1):
        S = np.arange(n)
        v = V[:, S].sum(1)
        r2, r1 = (v >= 2)[atk].mean() * 100, (v >= 1)[atk].mean() * 100
        per = "  ".join(f"{nm(c)}[{int(np.isin(pairs[c], S).sum())}] {100 * (v >= 1)[y == c].mean():5.1f}/"
                        f"{100 * (v >= 2)[y == c].mean():5.1f}" for c in classes)
        print(f"  {n:>2} | {r2:7.2f} | {r1:7.2f} | {rec_gl - r1:6.2f} | {r1 - r2:6.2f} | {per}")

    # ---------------- (c) where the reference false positives come from
    ref_fp = (V.sum(1) >= 2) & ~atk
    print(f"\n=== (c) the {int(ref_fp.sum()):,} false positives of k>=2 @0.5 (GL union, 14) ===")
    both = {c: int((V[:, pairs[c]].all(1) & ref_fp).sum()) for c in classes}
    any_ = {c: int((V[:, pairs[c]].any(1) & ref_fp).sum()) for c in classes}
    only_pair = 0
    for i in np.where(ref_fp)[0][:0]:
        pass
    fired_cls = np.column_stack([V[:, pairs[c]].any(1) for c in classes])
    n_cls = fired_cls[ref_fp].sum(1)
    print("  false positives on which BOTH specialists of an attack fired (the pair alone closes k>=2):")
    for c in classes:
        print(f"    {nm(c):<18} both {both[c]:>7,}   at least one {any_[c]:>7,}")
    print(f"  false positives whose votes come from a single attack's pair: {int((n_cls == 1).sum()):,} "
          f"({100 * (n_cls == 1).mean():.1f}%); from two or more attacks: {int((n_cls >= 2).sum()):,}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/k_fusion_audit_followup.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
