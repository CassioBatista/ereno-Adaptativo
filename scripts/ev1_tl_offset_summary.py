#!/usr/bin/env python3
"""Arrival-offset check of the trust-level grid (paper 2, 5.3; exploratory, not registered).

The registered late blocks join at 30 and 60 min, on phase boundaries of both Disaster-FD
traces. This summary sets the registered runs (offset 0, results/ev1_tl/grid_d60) beside the
runs with both blocks shifted by +5 and +20 min (results/ev1_tl/arrival_offsets), for ReSIDS,
fixed GL and fixed FL (all with retention), ERENO and CICIDS2017, T1 and T2.

Same definitions as scripts/ev1_tl_grid_summary.py: a node is protected against a block when
its recall on the block's types is at least 95 % of the domain reference (the largest recall
on those types in any run of the registered grid); median latency from arrival to protection
over the nodes protected by the end; exposure = share of the node-rounds after arrival without
protection. Block 1 = nodes 3, 4, 11, 12; block 2 = nodes 9, 10.

  python scripts/ev1_tl_offset_summary.py
Out: results/ev1_tl_offset_summary.{txt,csv}
"""
import csv
import glob
import os

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
GRID = os.path.join(ROOT, "results/ev1_tl/grid_d60")
OFF = os.path.join(ROOT, "results/ev1_tl/arrival_offsets")
DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017")]
TRACES = ["T1", "T2"]
POLS = [("switching", "ReSIDS"), ("fixed_gossip", "Fixed GL"), ("fixed_federated", "Fixed FL")]
OFFSETS = [0, 5, 20]
REL = 0.95


def run_dir(dom, tr, pol, off):
    return (os.path.join(GRID, f"{dom}_{tr}_{pol}_retain_d60") if off == 0
            else os.path.join(OFF, f"{dom}_{tr}_{pol}_retain_off{off}"))


def load(path):
    rows = list(csv.DictReader(open(os.path.join(path, "node_eval.csv"))))
    late = sorted(int(float(c.split("_")[-1])) for c in rows[0] if c.startswith("recall_late_"))
    f = lambda v: float(v) if v not in ("", "nan") else np.nan
    return late, [{"t": float(r["t_s"]), "node": int(r["node"]), "reach": r["reachable"] == "1",
                   **{a: f(r[f"recall_late_{a}"]) for a in late}} for r in rows]


def reference(dom):
    """Largest recall on the types of each block over every run of the registered grid."""
    ref = {1: 0.0, 2: 0.0}
    for p in glob.glob(os.path.join(GRID, f"{dom}_*_d60")):
        late, rows = load(p)
        for k, a in zip((1, 2), late):
            ref[k] = max(ref[k], np.nanmax([r[a] for r in rows]))
    return ref


def main():
    out, lines = [], ["arrival-offset check (exploratory): blocks at 30+o and 60+o min; protected = recall on the "
                      f"block's types >= {100 * REL:.0f} % of the domain reference", ""]
    for dom, dn in DOMS:
        ref = reference(dom)
        lines.append(f"==== {dn} (reference: block 1 {100 * ref[1]:.2f} %, block 2 {100 * ref[2]:.2f} %)")
        for tr in TRACES:
            lines.append(f"--- {tr}")
            lines.append(f"{'offset':>7} {'config':<10}" + "".join(
                f"{f'| block {k}: arrival':>20}{'prot. end':>10}{'median lat.':>12}{'exposure':>10}" for k in (1, 2)))
            for off in OFFSETS:
                for pol, lab in POLS:
                    p = run_dir(dom, tr, pol, off)
                    if not os.path.exists(os.path.join(p, "node_eval.csv")):
                        lines.append(f"{off:>5} m {lab:<10} (missing)")
                        continue
                    late, rows = load(p)
                    ts = sorted({r["t"] for r in rows})
                    last = [r for r in rows if r["t"] == ts[-1]]
                    rec = {"domain": dn, "trace": tr, "offset_min": off, "config": lab}
                    cells = ""
                    for k, a in zip((1, 2), late):
                        thr = REL * ref[k]
                        first = {}
                        for r in sorted(rows, key=lambda r: r["t"]):
                            if r["t"] >= a and r[a] >= thr and r["node"] not in first:
                                first[r["node"]] = r["t"]
                        lat = [first[n] - a for n in first]
                        after = [r for r in rows if r["t"] >= a]
                        expo = sum(1 for r in after if not (r[a] >= thr)) / max(len(after), 1)
                        prot = sum(1 for r in last if r[a] >= thr)
                        rec.update({f"b{k}_arrival_min": a / 60, f"b{k}_protected_end": prot,
                                    f"b{k}_median_latency_min": np.median(lat) / 60 if lat else np.nan,
                                    f"b{k}_exposure": expo})
                        cells += (f"{f'| {a / 60:.0f} min':>20}{f'{prot}/14':>10}"
                                  f"{(f'{np.median(lat) / 60:.0f} min' if lat else '-'):>12}{100 * expo:>9.1f}%")
                    out.append(rec)
                    lines.append(f"{off:>5} m {lab:<10}{cells}")
        lines.append("")
    txt = "\n".join(lines)
    open(os.path.join(ROOT, "results/ev1_tl_offset_summary.txt"), "w").write(txt + "\n")
    with open(os.path.join(ROOT, "results/ev1_tl_offset_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(txt)


if __name__ == "__main__":
    main()
