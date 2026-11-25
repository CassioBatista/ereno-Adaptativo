#!/usr/bin/env python3
"""Coverage rule for the specialists (exploratory, after the step-1 results; paper 2, 5.2).

One rule for every domain, decided on training data only, as the nodes could compute it after
an exchange: each node applies the specialists it received to its own training rows.
  - coverage C[t, s]: share of the training rows of attack type t that specialist s flags
    (score >= 0.5), computed by the node(s) holding t;
  - precision: the false-positive rate of s on the benign training rows of the other nodes
    (rows s never saw); s is precise when it is at most --precise (default 1 %);
  - cover: C[t, s] >= --cover (default 90 %).
Variants (the test split is used only to score them):
  all        every specialist (step 1, attack-based OR);
  cover-1    greedy minimal set of precise specialists such that every type covered by some
             precise specialist keeps one; a type that no precise specialist covers keeps its
             own specialist(s);
  cover-2    as cover-1, with two covering specialists on distinct nodes where available;
  precise    cover-1 without the imprecise specialists: a type that no precise specialist
             covers is left to whatever kept specialist flags it;
  precise-2  cover-2 without the imprecise specialists (added after the seeds 43-46 check,
             where cover-1 lost DoS Hulk in CICIDS2017).
Greedy order: the specialist that covers the most types still in need, then the lower
false-positive rate, then the lower index.

  python scripts/ev1_coverage_rule.py [--cover 0.9] [--precise 0.01] [--seed 42]
Out: results/ev1_coverage_rule[_s<seed>].{txt,json} (step-1 models and scores of that seed)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017"), ("morris4", "Morris-4"), ("ciciot", "CICIoT2023")]
PREV = 0.068


def train_side(dom, seed=42):
    """Coverage matrix and out-of-node benign FPR of every specialist, on the training split."""
    import ev1_step1 as E
    import xgboost as xgb
    d = E.DOMAINS[dom]
    feats, _ = E.selection(d)
    X, y, classes, nc = E.load(d["train"], feats)
    units, slices, kept = E.attack_partition(y, nc, classes, d["cap"], seed)
    spec = json.load(open(os.path.join(ROOT, f"results/ev1_step1_{dom}_s{seed}.json")))["specialists"]
    assert [(u[0], u[2] + 1) for u in units] == [(s["name"], s["node"]) for s in spec]
    types = sorted({c for u in units for c in u[1]})
    rows_t = {c: np.concatenate([u[3] for u in units if c in u[1]]) for c in types}
    ben_node = {n: slices[n] for n in range(len(slices))}
    mdir = os.path.join(ROOT, f"results/.cache/ev1_step1_models_{dom}_s{seed}")
    allrows = np.unique(np.concatenate([*rows_t.values(), *ben_node.values()]))
    dm = xgb.DMatrix(X[allrows])
    pos = {r: i for i, r in enumerate(allrows)}
    take = lambda idx: np.fromiter((pos[r] for r in idx), dtype=np.int64, count=len(idx))
    idx_t = {c: take(rows_t[c]) for c in types}
    idx_b = {n: take(ben_node[n]) for n in ben_node}
    del X
    C = np.zeros((len(types), len(spec)))
    fpr = np.zeros(len(spec))
    for s in spec:
        b = xgb.Booster()
        b.load_model(os.path.join(mdir, f"spec{s['j']:02d}.json"))
        h = b.predict(dm) >= 0.5
        for i, c in enumerate(types):
            C[i, s["j"]] = h[idx_t[c]].mean()
        other = np.concatenate([idx_b[n] for n in idx_b if n != s["node"] - 1])
        fpr[s["j"]] = h[other].mean()
    return spec, types, classes, C, fpr


def greedy(spec, types, C, fpr, cover, precise, k):
    node = np.array([s["node"] for s in spec])
    ok = (C >= cover) & (fpr <= precise)[None, :]
    need = {i: min(k, len(set(node[ok[i]]))) for i in range(len(types))}
    got = {i: set() for i in range(len(types))}           # nodes already covering type i
    chosen = []
    while any(need[i] > 0 for i in need):
        best, key = None, None
        for j in range(len(spec)):
            if j in chosen:
                continue
            gain = sum(1 for i in need if need[i] > 0 and ok[i, j] and node[j] not in got[i])
            if gain == 0:
                continue
            kk = (-gain, fpr[j], j)
            if key is None or kk < key:
                best, key = j, kk
        if best is None:
            break
        chosen.append(best)
        for i in need:
            if need[i] > 0 and ok[i, best] and node[best] not in got[i]:
                got[i].add(node[best])
                need[i] -= 1
    unprotected = [i for i in range(len(types)) if not ok[i].any()]
    return chosen, unprotected


def evaluate(dom, keep, seed=42):
    z = np.load(os.path.join(ROOT, f"results/.cache/ev1_step1_{dom}_s{seed}.npz"))
    hit, y, nc = z["Ps"] >= 0.5, z["y"], int(z["nc"])
    classes = [str(c) for c in z["classes"]]
    pred = hit[:, sorted(keep)].any(1) if keep else np.zeros(len(y), bool)
    att, ben = y != nc, y == nc
    R, F = pred[att].mean(), pred[ben].mean()
    Fb = F if F > 0 else 3 / ben.sum()
    P = R * PREV / (R * PREV + Fb * (1 - PREV)) if R > 0 else 0.0
    per = {classes[c]: float(pred[y == c].mean()) for c in sorted(set(y[att].tolist()))}
    return {"recall": float(R), "fpr": float(F), "f1_at_6.8": float(2 * P * R / (P + R)) if R > 0 else 0.0,
            "per_type": per}


def main(cover, precise, seed):
    out, lines = {}, [f"coverage rule (exploratory), step-1 seed {seed}: cover >= {100 * cover:g} % of the "
                      f"type's training rows; precise = FPR <= {100 * precise:g} % on the benign training rows "
                      f"of the other nodes", ""]
    for dom, name in DOMS:
        spec, types, classes, C, fpr = train_side(dom, seed)
        S = len(spec)
        tn = [classes[c] for c in types]
        own = {i: [s["j"] for s in spec if types[i] in s["codes"]] for i in range(len(types))}
        c1, unp = greedy(spec, types, C, fpr, cover, precise, 1)
        c2, _ = greedy(spec, types, C, fpr, cover, precise, 2)
        add_own = sorted({j for i in unp for j in own[i]})
        variants = {"all": list(range(S)), "cover-1": sorted(set(c1) | set(add_own)),
                    "cover-2": sorted(set(c2) | set(add_own)), "precise": sorted(set(c1)),
                    "precise-2": sorted(set(c2))}
        res = {v: {"kept": ks, **evaluate(dom, ks, seed)} for v, ks in variants.items()}
        base = res["all"]["per_type"]
        lines += [f"== {name}: {S} specialists, {int((fpr <= precise).sum())} precise; "
                  f"types no precise specialist covers: {len(unp)}"
                  + (f" ({', '.join(tn[i] for i in unp)})" if unp else ""),
                  f"   {'variant':<9}{'kept':>6}{'recall':>9}{'FPR %':>9}{'F1@6.8':>9}   types losing >= 5 points of recall"]
        for v, r in res.items():
            lost = [f"{t} {100 * base[t]:.0f}->{100 * r['per_type'][t]:.0f}"
                    for t in base if base[t] - r["per_type"][t] >= 0.05]
            lines.append(f"   {v:<9}{len(r['kept']):>3}/{S:<2}{100 * r['recall']:>9.2f}{100 * r['fpr']:>9.3f}"
                         f"{100 * r['f1_at_6.8']:>9.2f}   {'; '.join(lost) or '-'}")
        lines.append("   cover-1 keeps: " + ", ".join(f"{spec[j]['name'][:24]} (node {spec[j]['node']})"
                                                     for j in res["cover-1"]["kept"]))
        lines.append("   imprecise specialists (FPR on other nodes' benign): " + ", ".join(
            f"{spec[j]['name'][:20]} {100 * fpr[j]:.1f} %" for j in range(S) if fpr[j] > precise) or "   none")
        lines.append("")
        out[dom] = {"types": tn, "C_train": C.tolist(), "fpr_train_other_nodes": fpr.tolist(),
                    "unprotected": [tn[i] for i in unp],
                    "variants": {v: {k2: v2 for k2, v2 in r.items()} for v, r in res.items()}}
        print("\n".join(lines[-(len(res) + 5):]), flush=True)
    tag = ("" if (cover, precise) == (0.9, 0.01) else f"_c{cover:g}_p{precise:g}") + \
          ("" if seed == 42 else f"_s{seed}")
    json.dump(out, open(os.path.join(ROOT, f"results/ev1_coverage_rule{tag}.json"), "w"), indent=1)
    open(os.path.join(ROOT, f"results/ev1_coverage_rule{tag}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", type=float, default=0.9)
    ap.add_argument("--precise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.chdir(ROOT)
    main(a.cover, a.precise, a.seed)
