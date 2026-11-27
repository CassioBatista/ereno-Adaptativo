#!/usr/bin/env python3
"""EXPLORATORY (no training): the outage sweep of scripts/ev1_tl_sweep.py over seeds 42-46, with
the boosters of the faithful T2 switching runs of each seed (results/ev1_tl/faithful and
faithful_seeds/s<seed>). First checks that the offline replay reproduces the logged holdings of
every run of each seed; then, per domain and family, the quantities quoted in Section 4.3:
largest gain of ReSIDS over the better fixed mode (and the outage at which it occurs), crossover
outage (first D at which fixed FL leaves at least as many node-rounds unprotected as fixed GL),
largest lag of fixed FL and of fixed GL behind ReSIDS, and the smallest gain of ReSIDS over either
fixed mode in any cell (never negative under C1 and C2).

  python scripts/ev1_tl_sweep_seeds.py
Out: results/ev1_tl_sweep_seeds.txt (per seed, then mean and range over the five seeds)
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import ev1_tl_sweep as S  # noqa: E402
from ev1_tl_offline import Run, propagate  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]


def validate(seed):
    base = "results/ev1_tl/faithful" if seed == 42 else f"results/ev1_tl/faithful_seeds/s{seed}"
    bad = total = 0
    for d in sorted(glob.glob(os.path.join(ROOT, base, "*_T[12]_*"))):
        if not os.path.exists(os.path.join(d, "rounds.jsonl")):
            continue
        run = Run(d)
        sim = propagate(run, run.schedule())
        total += 1
        bad += any(sim[k][i] != set(r["holdings"][i]) for k, r in enumerate(run.rounds) for i in range(run.n))
    return total, bad


def table(dom, family):
    run = Run(S.run_dir(dom))
    mA, mB = S.labels_of(run, [S.BLOCK, S.BLOCK_B])
    pall = np.zeros(mA.size, bool)
    for h in set().union(*run.own):
        pall |= run.hits[h]
    refA, refB = pall[mA].mean(), pall[mB].mean()
    memo = {}

    def prot(hs, withB):
        k = (frozenset(hs), withB)
        if k not in memo:
            p = np.zeros(mA.size, bool)
            for h in hs:
                p |= run.hits[h]
            ok = p[mA].mean() >= S.REL * refA
            if withB:
                ok = ok and p[mB].mean() >= S.REL * refB
            memo[k] = ok
        return memo[k]

    rows = []
    for D in S.DS:
        arrival = {i: (float(S.W + 1) if i in S.BLOCK
                       else float(S.W + 1 + D) if (family == 2 and i in S.BLOCK_B) else 0.0)
                   for i in range(run.n)}
        un = {}
        for p in ("ReSIDS", "FL", "GL"):
            hold = propagate(run, S.schedule(p, D), arrival)
            vals = []
            for k in range(S.W, S.W + S.H):
                withB = family == 2 and (k + 1) >= S.W + 1 + D
                vals += [not prot(h, withB) for h in hold[k]]
            un[p] = 100 * np.mean(vals)
        rows.append((D, un["ReSIDS"], un["FL"], un["GL"]))
    return rows


def summary(rows):
    gfl = [fl - r for _, r, fl, gl in rows]
    ggl = [gl - r for _, r, fl, gl in rows]
    best = [min(a, b) for a, b in zip(gfl, ggl)]
    k = int(np.argmax(best))
    cross = next((D for D, r, fl, gl in rows if fl >= gl and D > 0), None)
    return {"gain_best": best[k], "at_D": rows[k][0], "cross": cross,
            "fl_lag": max(gfl), "gl_lag": max(ggl), "min_gain": min(min(gfl), min(ggl))}


def main():
    out, res = [], {}
    out.append("EXPLORATORY (no training): outage sweep over seeds 42-46 (scripts/ev1_tl_sweep.py "
               "scenarios; boosters of the faithful T2 switching run of each seed).")
    for seed in SEEDS:
        S.SEED = seed
        total, bad = validate(seed)
        out.append(f"\nseed {seed}: offline replay reproduces the logged holdings in {total - bad} of {total} runs")
        for fam in (1, 2):
            for dom, name in S.DOMS:
                s = summary(table(dom, fam))
                res.setdefault((fam, name), []).append(s)
                out.append(f"  family {fam} {name:<11} gain over better fixed {s['gain_best']:5.1f} (D={s['at_D']}), "
                           f"crossover D={s['cross']}, FL lag max {s['fl_lag']:5.1f}, GL lag max {s['gl_lag']:5.1f}, "
                           f"smallest gain {s['min_gain']:4.1f}")
    out.append("\nover the five seeds, mean (min-max):")
    for (fam, name), ss in res.items():
        f = lambda k: f"{np.mean([s[k] for s in ss]):5.1f} ({min(s[k] for s in ss):.1f}-{max(s[k] for s in ss):.1f})"
        cr = [s["cross"] for s in ss if s["cross"] is not None]
        out.append(f"  family {fam} {name:<11} gain over better fixed {f('gain_best')}, at D {sorted({s['at_D'] for s in ss})}, "
                   f"crossover D {sorted(set(cr)) if cr else 'none'}, FL lag {f('fl_lag')}, GL lag {f('gl_lag')}, "
                   f"smallest gain {min(s['min_gain'] for s in ss):.1f}")
    p = os.path.join(ROOT, "results/ev1_tl_sweep_seeds.txt")
    open(p, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
