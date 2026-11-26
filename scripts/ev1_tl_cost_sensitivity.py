#!/usr/bin/env python3
"""EXPLORATORY (protocol change log 2026-11-26), on the faithful replay (results/ev1_tl/faithful,
fixed GL with gl_peers both), no new run:

1. Communication per policy, from the holdings logs (rounds.jsonl) and the booster files:
   - federated round: every active node uploads the set it holds and downloads the union
     (2 transfers per node); gossip round (GLow): the head of the round, active[(r-1) % n],
     receives the set of each of its ring neighbours that is also active (1 transfer each);
     local learning: nothing.
   - transfers, and full-state volume (sum of the sizes of the sets sent, booster JSON files),
     totals over the run; the trust-level probing of Disaster-FD is the same for every policy
     and is not counted;
   - new boosters delivered: the volume of the boosters each node receives that it did not hold
     (its own new specialist excluded), the lower bound of any exchange that sends only what the
     receiver lacks.
2. Sensitivity of the protection criterion: recall on a late block at least REL of the domain
   reference, REL = 0.90, 0.95 (reported) and 0.99; per REL: informative cells (> 1 point between
   the three policies), cells where ReSIDS is the best (within 1 point), mean / max regret of the
   fixed modes, and the smallest gain of ReSIDS over each fixed mode in the share of nodes protected
   against the attacks present (negative = ReSIDS behind in some round).

  python scripts/ev1_tl_cost_sensitivity.py
Out: results/ev1_tl_cost_sensitivity.txt
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev1_tl_faithful_summary as S  # noqa: E402

N = 14


def booster_kb(run_dir):
    d = os.path.join(run_dir, "boosters")
    return {f[:-5]: os.path.getsize(os.path.join(d, f)) / 1024 for f in os.listdir(d) if f.endswith(".json")}


def cost(run_dir):
    kb = booster_kb(run_dir)
    size = lambda hs: sum(kb.get(h, 0.0) for h in hs)
    rounds = [json.loads(l) for l in open(os.path.join(run_dir, "rounds.jsonl"))]
    prev = [[] for _ in range(N)]
    out = {"federated": [0, 0, 0.0], "gossip": [0, 0, 0.0], "local": [0, 0, 0.0]}   # rounds, transfers, KB
    seen, delivered = set(), 0.0
    for r in rounds:
        # delivered: copies of boosters a node did not hold before the round, minus the copy of
        # a booster that exists nowhere before the round (its own node trained it)
        newly = {}
        for i in range(N):
            for h in set(r["holdings"][i]) - set(prev[i]):
                newly[h] = newly.get(h, 0) + 1
        for h, k in newly.items():
            delivered += kb.get(h, 0.0) * (k - (0 if h in seen else 1))
        for hs in r["holdings"]:
            seen.update(hs)
        mode, act = r["mode"], sorted(r["active"] or [])
        c = out.setdefault(mode, [0, 0, 0.0])
        c[0] += 1
        if mode == "federated" and act:
            union = set().union(*[set(r["holdings"][i]) for i in act])
            c[1] += 2 * len(act)
            c[2] += sum(size(prev[i]) for i in act) + len(act) * size(union)
        elif mode == "gossip" and act:
            head = act[(r["round"] - 1) % len(act)]
            nb = [j for j in ((head - 1) % N, (head + 1) % N) if j in act]
            c[1] += len(nb)
            c[2] += sum(size(prev[j]) for j in nb)
        prev = r["holdings"]
    out["delivered_kb"] = delivered
    return out


def main():
    lines = ["EXPLORATORY: communication per policy and sensitivity of the protection criterion "
             "(faithful replay, results/ev1_tl/faithful)", ""]
    lines.append("1. Communication over the run: rounds in FL / GL / local, booster-set transfers, "
                 "full-state volume (MB)")
    tot = {}
    for dom, dn in S.DOMS:
        for tr, _ in S.TRACES:
            for pol, lab in S.POLS:
                d = S.rdir(dom, tr, pol)
                if not os.path.isdir(os.path.join(d, "boosters")):
                    continue
                c = cost(d)
                tr_n = c["federated"][1] + c["gossip"][1]
                mb = (c["federated"][2] + c["gossip"][2]) / 1024
                dl = c["delivered_kb"] / 1024
                t = tot.setdefault((tr, lab), [0, 0.0, 0.0, 0])
                t[0] += tr_n; t[1] += mb; t[2] += dl; t[3] += 1
                lines.append(f"  {dn:<11}{tr} {lab:<7} rounds {c['federated'][0]:3d}/{c['gossip'][0]:3d}/"
                             f"{c.get('local', [0])[0]:3d}  transfers {tr_n:5d}  full-state {mb:7.2f} MB  "
                             f"new boosters delivered {dl:6.2f} MB")
    lines.append("  mean over the four domains:")
    for (tr, lab), (n, mb, dl, k) in sorted(tot.items()):
        lines.append(f"    {tr} {lab:<7} transfers {n / k:7.1f}  full-state {mb / k:7.2f} MB  "
                     f"new boosters delivered {dl / k:6.2f} MB")
    lines += ["", "2. Sensitivity of the protection criterion"]
    for rel in (0.90, 0.95, 0.99):
        S.REL = rel
        cells, gains = {}, {"FL": [], "GL": []}
        for dom, dn in S.DOMS:
            runs = {(tr, p): S.node_eval(dom, tr, p) for tr, _ in S.TRACES for p, _ in S.POLS if S.have(dom, tr, p)}
            ref = {a: max(np.nanmax([r[a] for r in rows]) for rows in runs.values()) for a in S.BLOCKS}
            refp = S.ref_present(runs)
            for tr, _ in S.TRACES:
                for p, lab in S.POLS:
                    rows = runs[(tr, p)]
                    for a in S.BLOCKS:
                        after = [r for r in rows if r["t"] >= a]
                        cells.setdefault((dn, tr, a), {})[lab] = 100 * float(np.mean([not (r[a] >= rel * ref[a]) for r in after]))
                ps = {p: dict(S.protected_share(runs[(tr, p)], ref, refp)) for p, _ in S.POLS}
                for k, p in (("FL", "fixed_federated"), ("GL", "fixed_gossip")):
                    gains[k] += [100 * (ps["switching"][t] - ps[p][t]) for t in ps["switching"]]
        inf = {k: v for k, v in cells.items() if max(v.values()) - min(v.values()) > 1.0}
        best = sum(1 for v in inf.values() if v["ReSIDS"] - min(v.values()) <= 1.0)
        reg = {p: [v[p] - min(v.values()) for v in inf.values()] for _, p in S.POLS}
        lines.append(f"  REL {rel:.2f}: informative cells {len(inf)} of {len(cells)}; ReSIDS best (within 1) in {best}; "
                     f"regret mean/max ReSIDS {np.mean(reg['ReSIDS']):.1f}/{np.max(reg['ReSIDS']):.1f}, "
                     f"FL {np.mean(reg['FL']):.1f}/{np.max(reg['FL']):.1f}, GL {np.mean(reg['GL']):.1f}/{np.max(reg['GL']):.1f}; "
                     f"smallest gain vs FL {min(gains['FL']):.1f}, vs GL {min(gains['GL']):.1f}")
    S.REL = 0.95
    out = os.path.join(S.ROOT, "results/ev1_tl_cost_sensitivity.txt")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
