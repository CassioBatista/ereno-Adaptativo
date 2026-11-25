#!/usr/bin/env python3
"""Summary of the coverage rule over step-1 seeds 42-46 (results/ev1_coverage_rule[_s<seed>].json):
kept specialists, recall, FPR and F1 at 6.8 % per domain and variant, as min-max over the seeds,
and the change from 'all' within each seed. Out: results/ev1_coverage_rule_seeds.txt"""
import json
import os

ROOT = os.path.expanduser("~/ereno-Adaptativo")
SEEDS = [42, 43, 44, 45, 46]
DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017"), ("morris4", "Morris-4"), ("ciciot", "CICIoT2023")]
VARS = ["all", "cover-1", "cover-2", "precise", "precise-2"]
R = {s: json.load(open(os.path.join(ROOT, f"results/ev1_coverage_rule{'' if s == 42 else f'_s{s}'}.json")))
     for s in SEEDS}
lines = ["coverage rule (cover >= 90 %, precise <= 1 %), step-1 seeds 42-46: min-max over the seeds; "
         "d = change from 'all' in the same seed (min..max)", ""]
for dom, name in DOMS:
    lines.append(f"== {name}")
    lines.append(f"   {'variant':<9}{'kept':>9}{'recall %':>16}{'d recall':>16}{'FPR %':>16}{'d FPR':>16}{'F1@6.8':>14}")
    for v in VARS:
        k = [len(R[s][dom]["variants"][v]["kept"]) for s in SEEDS]
        rec = [100 * R[s][dom]["variants"][v]["recall"] for s in SEEDS]
        fpr = [100 * R[s][dom]["variants"][v]["fpr"] for s in SEEDS]
        f1 = [100 * R[s][dom]["variants"][v]["f1_at_6.8"] for s in SEEDS]
        dr = [rec[i] - 100 * R[s][dom]["variants"]["all"]["recall"] for i, s in enumerate(SEEDS)]
        dfp = [fpr[i] - 100 * R[s][dom]["variants"]["all"]["fpr"] for i, s in enumerate(SEEDS)]
        mm = lambda a, f="{:.2f}": f"{f.format(min(a))}-{f.format(max(a))}"
        lines.append(f"   {v:<9}{f'{min(k)}-{max(k)}':>9}{mm(rec):>16}{mm(dr, '{:+.2f}'):>16}"
                     f"{mm(fpr, '{:.3f}'):>16}{mm(dfp, '{:+.3f}'):>16}{mm(f1):>14}")
    unp = [len(R[s][dom]["unprotected"]) for s in SEEDS]
    lines.append(f"   types no precise specialist covers: {min(unp)}-{max(unp)}")
    # per-type losses of the precise variant, union over seeds
    lost = {}
    for s in SEEDS:
        base = R[s][dom]["variants"]["all"]["per_type"]
        pv = R[s][dom]["variants"]["precise"]["per_type"]
        for t in base:
            if base[t] - pv[t] >= 0.05:
                lost.setdefault(t, []).append(s)
    lines.append(f"   precise: types losing >= 5 points in some seed: "
                 + ("; ".join(f"{t} ({len(v)}/5 seeds)" for t, v in sorted(lost.items())) or "-"))
    for v in ("cover-1", "cover-2", "precise-2"):
        lost1 = set()
        for s in SEEDS:
            base = R[s][dom]["variants"]["all"]["per_type"]
            c1 = R[s][dom]["variants"][v]["per_type"]
            lost1 |= {f"{t} s{s} {100 * base[t]:.0f}->{100 * c1[t]:.0f}" for t in base if base[t] - c1[t] >= 0.05}
        lines.append(f"   {v}: types losing >= 5 points in some seed: {'; '.join(sorted(lost1)) or '-'}")
    lines.append("")
txt = "\n".join(lines)
open(os.path.join(ROOT, "results/ev1_coverage_rule_seeds.txt"), "w").write(txt + "\n")
print(txt)
