#!/usr/bin/env python3
"""Sensitivity of the trust-level replay to the round period (evaluation-v1 section 9;
paper 2, 5.3), on the faithful replay (links: scenario, rounds offset by half a period):
Δ = 5, 15, 60, 180, 300, 600 s; 60 s from results/ev1_tl/faithful, the others from
results/ev1_tl/delta_sens. ReSIDS, fixed GL and fixed FL (retention), registered arrivals.

Same definitions as scripts/ev1_tl_grid_summary.py and ev1_tl_arguments.py: protected = recall on
the block's types >= 95 % of the domain reference (largest recall on them in the 60-s grid);
exposure = share of the node-rounds after arrival without protection, over all nodes and over
the nodes that reach the aggregator or a neighbour; median latency to protection in minutes;
regret = exposure minus the best of the three policies in the same cell. Runs not yet done are
reported as missing, so the summary can be read while the sensitivity is running.

  python scripts/ev1_tl_delta_summary.py
Out: results/ev1_tl_delta_summary.{txt,csv}
"""
import csv
import glob
import os

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
GRID = os.path.join(ROOT, "results/ev1_tl/faithful")
SENS = os.path.join(ROOT, "results/ev1_tl/delta_sens")
DELTAS = [5, 15, 60, 180, 300, 600]
DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017"), ("morris4", "Morris-4"), ("ciciot", "CICIoT2023")]
TRACES = ["T1", "T2"]
POLS = [("switching", "ReSIDS"), ("fixed_gossip", "Fixed GL"), ("fixed_federated", "Fixed FL")]
BLOCKS = (1800, 3600)
REL = 0.95


def run_dir(dom, tr, pol, d):
    return (os.path.join(GRID, f"{dom}_{tr}_{pol}") if d == 60
            else os.path.join(SENS, f"{dom}_{tr}_{pol}_d{d}"))


def load(path):
    f = lambda v: float(v) if v not in ("", "nan") else np.nan
    return [{"t": float(r["t_s"]), "node": int(r["node"]), "reach": r["reachable"] == "1",
             **{a: f(r[f"recall_late_{a}"]) for a in BLOCKS}}
            for r in csv.DictReader(open(os.path.join(path, "node_eval.csv")))]


def reference(dom):
    ref = {a: 0.0 for a in BLOCKS}
    for p in glob.glob(os.path.join(GRID, f"{dom}_T*")):
        if not os.path.exists(os.path.join(p, "node_eval.csv")):
            continue
        for r in load(p):
            for a in BLOCKS:
                if not np.isnan(r[a]):
                    ref[a] = max(ref[a], r[a])
    return ref


def main():
    out, lines = [], ["sensitivity to the round period (registered arrivals 30/60 min); per block: protected at the "
                      "end / median latency / exposure all nodes / exposure reachable nodes", ""]
    cells = {}
    for dom, dn in DOMS:
        if not any(os.path.exists(os.path.join(run_dir(dom, "T2", "switching", d), "node_eval.csv"))
                   for d in DELTAS if d != 60):
            continue
        ref = reference(dom)
        lines.append(f"==== {dn}")
        for tr in TRACES:
            lines.append(f"--- {tr}")
            for d in DELTAS:
                for pol, lab in POLS:
                    p = run_dir(dom, tr, pol, d)
                    if not os.path.exists(os.path.join(p, "node_eval.csv")):
                        lines.append(f"  {d:>4} s {lab:<9} (missing)")
                        continue
                    rows = load(p)
                    ts = sorted({r["t"] for r in rows})
                    last = [r for r in rows if r["t"] == ts[-1]]
                    rec = {"domain": dn, "trace": tr, "delta_s": d, "config": lab}
                    txt = ""
                    for k, a in enumerate(BLOCKS, 1):
                        thr = REL * ref[a]
                        first = {}
                        for r in sorted(rows, key=lambda r: r["t"]):
                            if r["t"] >= a and r[a] >= thr and r["node"] not in first:
                                first[r["node"]] = r["t"]
                        lat = [first[n] - a for n in first]
                        after = [r for r in rows if r["t"] >= a]
                        reach = [r for r in after if r["reach"]]
                        e_all = float(np.mean([not (r[a] >= thr) for r in after])) if after else np.nan
                        e_rch = float(np.mean([not (r[a] >= thr) for r in reach])) if reach else np.nan
                        prot = sum(1 for r in last if r[a] >= thr)
                        mlat = np.median(lat) / 60 if lat else np.nan
                        rec.update({f"b{k}_protected_end": prot, f"b{k}_median_latency_min": mlat,
                                    f"b{k}_exposure": e_all, f"b{k}_exposure_reachable": e_rch})
                        cells.setdefault((dn, tr, d, k), {})[lab] = 100 * e_all
                        txt += (f"  | block {k}: {prot:2d}/14 {('-' if np.isnan(mlat) else f'{mlat:.0f} min'):>7} "
                                f"{100 * e_all:5.1f}% {100 * e_rch:5.1f}%")
                    out.append(rec)
                    lines.append(f"  {d:>4} s {lab:<9}{txt}")
        lines.append("")
    # regret per period, over the cells where the three differ by more than one point
    lines.append("regret (exposure minus the best of the three, points) per period, over informative cells")
    for d in DELTAS:
        inf = [v for (dn, tr, dd, k), v in cells.items() if dd == d and len(v) == 3
               and max(v.values()) - min(v.values()) > 1.0]
        if not inf:
            continue
        parts = []
        for _, lab in POLS:
            reg = [v[lab] - min(v.values()) for v in inf]
            worst = sum(1 for v in inf if v[lab] == max(v.values()))
            parts.append(f"{lab} mean {np.mean(reg):.1f} max {np.max(reg):.1f} worst {worst}")
        lines.append(f"  {d:>4} s ({len(inf)} cells): " + "; ".join(parts))
    txt = "\n".join(lines)
    open(os.path.join(ROOT, "results/ev1_tl_delta_summary.txt"), "w").write(txt + "\n")
    if out:
        with open(os.path.join(ROOT, "results/ev1_tl_delta_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            for r in out:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(txt)


if __name__ == "__main__":
    main()
