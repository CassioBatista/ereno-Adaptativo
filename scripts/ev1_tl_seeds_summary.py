#!/usr/bin/env python3
"""EXPLORATORY (protocol change log 2026-11-26): dispersion of the faithful replay over seeds.
Seed 42 is results/ev1_tl/faithful (the reported runs); seeds 43-46 are
results/ev1_tl/faithful_seeds/s<seed> (scripts/ev1_tl_runners/run_tl_faithful_seeds.sh). For each
seed, the quantities of Section 5.3 with the rule of scripts/ev1_tl_faithful_summary.py (protected:
recall on a late block >= 95 % of the reference of the domain in that seed): informative cells
(> 1 point between the three configurations), cells where ReSIDS is the best (within 1 point),
mean / max regret of each configuration, and the smallest gain of ReSIDS over each fixed mode in
the share of nodes protected against the attacks present; then, per cell, the unprotected share of
each configuration over the seeds (mean, min-max). Seeds missing or incomplete are reported and
skipped.

  python scripts/ev1_tl_seeds_summary.py
Out: results/ev1_tl_seeds_summary.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev1_tl_faithful_summary as S  # noqa: E402

BASE = os.path.join(S.ROOT, "results/ev1_tl")
SEEDS = [42, 43, 44, 45, 46]


def seed_dir(seed):
    return os.path.join(BASE, "faithful") if seed == 42 else os.path.join(BASE, "faithful_seeds", f"s{seed}")


def one_seed(seed):
    root = seed_dir(seed)
    S.rdir = lambda d, tr, p: os.path.join(root, f"{d}_{tr}_{p}")
    cells, gains = {}, {"FL": [], "GL": []}
    for dom, dn in S.DOMS:
        if not all(S.have(dom, tr, p) for tr, _ in S.TRACES for p, _ in S.POLS):
            return None
        runs = {(tr, p): S.node_eval(dom, tr, p) for tr, _ in S.TRACES for p, _ in S.POLS}
        ref = {a: max(np.nanmax([r[a] for r in rows]) for rows in runs.values()) for a in S.BLOCKS}
        refp = S.ref_present(runs)
        for tr, _ in S.TRACES:
            for p, lab in S.POLS:
                rows = runs[(tr, p)]
                for a in S.BLOCKS:
                    after = [r for r in rows if r["t"] >= a]
                    cells.setdefault((dn, tr, a), {})[lab] = 100 * float(np.mean(
                        [not (r[a] >= S.REL * ref[a]) for r in after]))
            ps = {p: dict(S.protected_share(runs[(tr, p)], ref, refp)) for p, _ in S.POLS}
            for k, p in (("FL", "fixed_federated"), ("GL", "fixed_gossip")):
                gains[k] += [100 * (ps["switching"][t] - ps[p][t]) for t in ps["switching"]]
    inf = {k: v for k, v in cells.items() if max(v.values()) - min(v.values()) > 1.0}
    reg = {lab: [v[lab] - min(v.values()) for v in inf.values()] for _, lab in S.POLS}
    return {"cells": cells, "n_inf": len(inf),
            "best": sum(1 for v in inf.values() if v["ReSIDS"] - min(v.values()) <= 1.0),
            "worst": {lab: sum(1 for v in inf.values() if v[lab] == max(v.values())) for _, lab in S.POLS},
            "reg_mean": {lab: float(np.mean(x)) if x else 0.0 for lab, x in reg.items()},
            "reg_max": {lab: float(np.max(x)) if x else 0.0 for lab, x in reg.items()},
            "gain_min": {k: float(min(v)) for k, v in gains.items()}}


def main():
    res, lines = {}, ["EXPLORATORY: faithful replay over seeds (42 reported; 43-46 added 2026-11-26)", ""]
    for s in SEEDS:
        r = one_seed(s)
        if r is None:
            lines.append(f"seed {s}: missing or incomplete ({seed_dir(s)}), skipped")
            continue
        res[s] = r
        lines.append(f"seed {s}: informative {r['n_inf']}/16; ReSIDS best in {r['best']}; worst ReSIDS/FL/GL "
                     f"{r['worst']['ReSIDS']}/{r['worst']['FL']}/{r['worst']['GL']}; regret mean/max "
                     + ", ".join(f"{lab} {r['reg_mean'][lab]:.1f}/{r['reg_max'][lab]:.1f}" for _, lab in S.POLS)
                     + f"; smallest gain vs FL {r['gain_min']['FL']:.1f}, vs GL {r['gain_min']['GL']:.1f}")
    if len(res) > 1:
        lines += ["", f"over {len(res)} seeds (mean, min-max):"]
        agg = lambda f: np.array([f(r) for r in res.values()], float)
        for name, f in (("informative cells", lambda r: r["n_inf"]), ("ReSIDS best in", lambda r: r["best"]),
                        *((f"regret mean {lab}", (lambda lab: lambda r: r["reg_mean"][lab])(lab)) for _, lab in S.POLS),
                        *((f"regret max {lab}", (lambda lab: lambda r: r["reg_max"][lab])(lab)) for _, lab in S.POLS),
                        ("smallest gain vs FL", lambda r: r["gain_min"]["FL"]),
                        ("smallest gain vs GL", lambda r: r["gain_min"]["GL"])):
            a = agg(f)
            lines.append(f"  {name:<22} {a.mean():6.1f}  ({a.min():.1f}-{a.max():.1f})")
        lines += ["", "unprotected share of node-rounds per cell (%), mean (min-max) over the seeds:"]
        for key in sorted(next(iter(res.values()))["cells"]):
            parts = []
            for _, lab in S.POLS:
                a = np.array([r["cells"][key][lab] for r in res.values()])
                parts.append(f"{lab} {a.mean():5.1f} ({a.min():.1f}-{a.max():.1f})")
            lines.append(f"  {key[0]:<11}{key[1]} block {key[2] // 60} min: " + "  ".join(parts))
    out = os.path.join(S.ROOT, "results/ev1_tl_seeds_summary.txt")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
