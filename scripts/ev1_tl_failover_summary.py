#!/usr/bin/env python3
"""EXPLORATORY (protocol change log 2026-11-26): the baseline of aggregator re-election against
ReSIDS, fixed FL and fixed GL on the faithful replay (seed 42). Per domain, trace and late block:
the share of node-rounds after the arrival without protection (rule of
scripts/ev1_tl_faithful_summary.py; the reference of each domain is the largest recall in the
faithful runs, so that every configuration is scored against the same reference) for ReSIDS,
fixed FL, fixed GL and the three failover variants; then, per variant, the cells where it is
better or worse than ReSIDS by more than one point, and the smallest and largest difference in the
share of nodes protected against the attacks present, over the rounds.

  python scripts/ev1_tl_failover_summary.py
Out: results/ev1_tl_failover_summary.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev1_tl_faithful_summary as S  # noqa: E402

FO = os.path.join(S.ROOT, "results/ev1_tl/faithful_failover")
VARIANTS = [("failover_majority1", "Failover (majority, 1 round)"),
            ("failover_majority3", "Failover (majority, 3 rounds)"),
            ("failover_none1", "Failover (no quorum, 1 round)")]


def load(dom, tr, key):
    orig = S.rdir
    if key.startswith("failover"):
        S.rdir = lambda d, t, p: os.path.join(FO, f"{d}_{t}_{p}")
    try:
        return S.node_eval(dom, tr, key) if S.have(dom, tr, key) else None
    finally:
        S.rdir = orig


def main():
    lines = ["EXPLORATORY: aggregator re-election (failover) against ReSIDS, fixed FL and fixed GL "
             "(faithful replay, seed 42)", "",
             "unprotected share of node-rounds after the arrival (%):"]
    cmp = {k: [] for k, _ in VARIANTS}
    gain = {k: [] for k, _ in VARIANTS}
    for dom, dn in S.DOMS:
        main_runs = {(tr, p): S.node_eval(dom, tr, p) for tr, _ in S.TRACES for p, _ in S.POLS}
        ref = {a: max(np.nanmax([r[a] for r in rows]) for rows in main_runs.values()) for a in S.BLOCKS}
        refp = S.ref_present(main_runs)
        for tr, _ in S.TRACES:
            runs = {lab: main_runs[(tr, p)] for p, lab in S.POLS}
            for k, _ in VARIANTS:
                rows = load(dom, tr, k)
                if rows is not None:
                    runs[k] = rows
            for a in S.BLOCKS:
                e = {lab: 100 * float(np.mean([not (r[a] >= S.REL * ref[a]) for r in rows if r["t"] >= a]))
                     for lab, rows in runs.items()}
                lines.append(f"  {dn:<11}{tr} block {a // 60} min: " + "  ".join(
                    f"{lab.replace('failover_', 'FO-')} {v:5.1f}" for lab, v in e.items()))
                for k, _ in VARIANTS:
                    if k in e:
                        cmp[k].append(e[k] - e["ReSIDS"])
            ps = {lab: dict(S.protected_share(rows, ref, refp)) for lab, rows in runs.items()}
            for k, _ in VARIANTS:
                if k in ps:
                    gain[k] += [100 * (ps["ReSIDS"][t] - ps[k][t]) for t in ps["ReSIDS"] if t in ps[k]]
    lines += ["", "per variant, against ReSIDS (unprotected share; positive = failover worse):"]
    for k, lab in VARIANTS:
        d = np.array(cmp[k])
        if not len(d):
            lines.append(f"  {lab}: no runs")
            continue
        g = np.array(gain[k])
        lines.append(f"  {lab}: cells {len(d)}; failover better by > 1 point in {int((d < -1).sum())}, worse in "
                     f"{int((d > 1).sum())}; difference mean {d.mean():+.1f}, min {d.min():+.1f}, max {d.max():+.1f}; "
                     f"gain of ReSIDS in nodes protected per round min {g.min():+.1f}, max {g.max():+.1f}")
    out = os.path.join(S.ROOT, "results/ev1_tl_failover_summary.txt")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
