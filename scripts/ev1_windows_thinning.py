#!/usr/bin/env python3
"""EXPLORATORY (not registered; protocol change log 2026-11-25): window alarms when attacks are
rare. The benign stream of the test split is kept whole and each attack sample is kept with
probability r (r = 1, 0.1, 0.01, 0.001), so that the same attacks reach a much longer benign
stream at a much lower prevalence; everything else is the rule of section 6 on the primary
windows (scripts/ev1_windows.py: OR at 0.5, T, f, T_s with margin 1.25, blocked-5). Thinning
draws: seeds 1-5; reported as mean (min-max).

Per r: prevalence (attack samples / samples), attack windows, attack windows alarmed, the
share of attack samples that fall in an alarmed window (what reaches the operator), the
per-sample recall of the same samples (for comparison), and false alarms.

  python scripts/ev1_windows_thinning.py [--seed 42]
Out: results/ev1_windows_thinning_s<seed>.{txt,json}
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev1_windows as E  # noqa: E402

RATES = [1.0, 0.1, 0.01, 0.001]
DRAWS = [1, 2, 3, 4, 5]
PRIMARY = {"ereno": 1.0, "cicids": 10.0, "morris4": 100.0}


def one(t, fired, y, nc, size):
    w = np.floor((t - t[0]) / size).astype(np.int64)
    W = E.window_table(w, fired, y, nc, [])
    starts = np.flatnonzero(np.r_[True, w[1:] != w[:-1]])
    na = np.add.reduceat((y != nc).astype(np.int64), starts)
    _, full, _, scored = E.evaluate(W)
    A, B = W["atk"] & scored, ~W["atk"] & scored
    atk = y != nc
    return {"prevalence": float(atk.mean()), "attack_samples": int(atk.sum()),
            "attack_windows": int(A.sum()), "alarmed": int(full[A].sum()),
            "tpr": float(full[A].mean()) if A.any() else float("nan"),
            "sample_coverage": float(na[full & scored].sum() / max(1, na[scored].sum())),
            "per_sample_recall": float(fired[atk].any(1).mean()) if atk.any() else float("nan"),
            "benign_windows": int(B.sum()), "false_alarms": int(full[B].sum())}


def main(seed):
    lines = ["EXPLORATORY: window alarms with thinned attacks (benign stream whole; each attack sample "
             "kept with probability r; seeds 1-5, mean (min-max)); rule and blocked-5 of section 6, "
             "primary windows", ""]
    res = {}
    for d, size in PRIMARY.items():
        dom = E.DOMS[d]
        z = np.load(os.path.join(E.ROOT, f"results/.cache/ev1_step1_{d}_s{seed}.npz"), allow_pickle=True)
        y0, nc = z["y"].astype(np.int64), int(z["nc"])
        t0 = E.times(dom)
        o = np.argsort(t0, kind="stable")
        t0, f0, y0 = t0[o], z["Ps"][o] >= 0.5, y0[o]
        lines.append(f"=== {dom['name']}, {size:g}-s windows")
        lines.append(f"  {'r':>6} {'prevalence':>12} {'attack win':>14} {'alarmed %':>20} "
                     f"{'samples reaching operator %':>30} {'per-sample recall %':>20} {'false alarms':>18}")
        res[d] = {}
        for r in RATES:
            runs = []
            for s in (DRAWS if r < 1 else [0]):
                keep = (y0 == nc) | (np.random.default_rng(s).random(len(y0)) < r) if r < 1 else np.ones(len(y0), bool)
                runs.append(one(t0[keep], f0[keep], y0[keep], nc, size))
            res[d][str(r)] = runs
            g = lambda k: np.array([q[k] for q in runs], float)
            fmt = lambda a, p=1, sc=100: (f"{sc * a.mean():.{p}f} ({sc * a.min():.{p}f}-{sc * a.max():.{p}f})"
                                           if len(a) > 1 else f"{sc * a.mean():.{p}f}")
            lines.append(f"  {r:>6g} {fmt(g('prevalence'), 3):>12} {fmt(g('attack_windows'), 0, 1):>14} "
                         f"{fmt(g('tpr')):>20} {fmt(g('sample_coverage')):>30} {fmt(g('per_sample_recall')):>20} "
                         f"{fmt(g('false_alarms'), 0, 1):>12} / {int(g('benign_windows').mean()):,}")
        lines.append("")
    out = os.path.join(E.ROOT, f"results/ev1_windows_thinning_s{seed}")
    open(out + ".txt", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(out + ".json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
