#!/usr/bin/env python3
"""Federation with retention at the server: per-round results of the four runs and their
comparison with the published runs of results/conv_adapt_shrink1.csv.

Runs (logs under results/, produced by `ereno.py distributed ... --conf <config> --clients 10`):
  flret_k1, flret_k2   conf/experiments/ereno_adapt_fl_retain_shrink1_k{1,2}.yaml
                       federated throughout, one node lost every 10 rounds (10 -> 3)
  flglfl_ret_k1        conf/experiments/ereno_adapt_fl_gl_fl_retain_early_return_k1.yaml
                       FL -> GL -> FL, the server returns empty with 3 nodes left
  glfl_ret_lost_k1     conf/experiments/ereno_adapt_gl_fl_retain_lost_before_diffusion_k1.yaml
                       7 of 10 nodes lost after 3 gossip rounds from scratch
Reference: flgl_k* (pool retained after the FL->GL switch) and glfl_k* (federated aggregate
re-formed each round) in results/conv_adapt_shrink1.csv.

  python scripts/fl_retention_check.py
Out: results/conv_adapt_fl_retain.csv, results/fl_retention_check.txt
"""
import csv
import io
import os
import re
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = {"flret_k1": "adapt_flret_k1", "flret_k2": "adapt_flret_k2",
        "flglfl_ret_k1": "adapt_flglfl_ret_k1", "glfl_ret_lost_k1": "adapt_glfl_ret_lost_k1"}


def parse(name):
    log = open(os.path.join(HERE, "results", name + ".log"), errors="replace").read()
    rounds = re.findall(r"^ROUND;(\d+);(\w+);f1=([\d.]+);recall=([\d.]+);fpr=([\d.]+)", log, re.M)
    return [(int(r), m, float(f), float(rc), float(fp)) for r, m, f, rc, fp in rounds], \
        re.findall(r"^\[Hybrid\] (.*)$", log, re.M)


def spans(rows):
    """Compress consecutive rounds with the same mode and F1 into 'r1-r2 mode F1'."""
    out, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i][1:3] != rows[start][1:3]:
            out.append(f"r{rows[start][0]}-{rows[i - 1][0]} {rows[start][1]} F1 {rows[start][2]:.2f}")
            start = i
    return "; ".join(out)


def main():
    ref = {}
    for row in csv.DictReader(open(os.path.join(HERE, "results", "conv_adapt_shrink1.csv"))):
        ref[(row["run"], int(row["round"]))] = float(row["f1"])
    allrows = []
    print("Federation with retention at the server (N = 10, ERENO, 500k test subsample per round)\n")
    for run, name in LOGS.items():
        rows, hybrid = parse(name)
        allrows += [[run, *r] for r in rows]
        print(f"{run}: {len(rows)} rounds | {spans(rows)}")
        for h in hybrid:
            print(f"    [Hybrid] {h}")
        if run.startswith("flret"):
            k = run[-1]
            d_ret = max(abs(r[2] - ref[(f"flgl_k{k}", r[0])]) for r in rows if r[0] >= 11)
            print(f"    max |F1 - pool retained (flgl_k{k})| over rounds 11-90: {d_ret:.4f}")
            print(f"    re-formed aggregate (glfl_k{k}) at round 90: {ref[(f'glfl_k{k}', 90)]:.2f}")
        print()
    with open(os.path.join(HERE, "results", "conv_adapt_fl_retain.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "round", "mode", "f1", "recall", "fpr"])
        w.writerows(allrows)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open(os.path.join(HERE, "results", "fl_retention_check.txt"), "w").write(buf.getvalue())
    print(buf.getvalue())
