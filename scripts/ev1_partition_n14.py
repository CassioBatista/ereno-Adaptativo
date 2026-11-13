#!/usr/bin/env python3
"""Specialists of evaluation-v1 and their placement on the 14 nodes
(docs/EVALUATION_PROTOCOL.md, section 2.3, amendment of 2026-11-13).

Rule, fixed before any evaluation run:
  1. Specialists. One per attack type; the types with fewer than SMALL = 100 training rows
     share one specialist (their rows together). If there are fewer specialists than nodes
     (ERENO, 7 types), the extra nodes go to the largest specialists, one at a time, and a
     specialist placed on k nodes has its rows split into k disjoint random parts (seed 42),
     one per node: the rule of the pipeline's `attack` partitioner (fd/dataset.py).
  2. Placement. Specialists in decreasing order of training rows (ties: order of the ARFF
     class declaration, the class codes of python/util.load_arff), in contiguous blocks:
     with S specialists, q = S div 14 and r = S mod 14, nodes 1..14-r take q consecutive
     specialists and the last r nodes take q + 1. Large specialists share a node only with
     large ones, small ones only with small ones.
  3. Each node holds the i-th of 14 disjoint random slices of the training benign rows
     (seed 42; ERENO capped at 500,000 first); every specialist on node i is trained on its
     own attack rows against that slice. The placement does not depend on the benign rows.
  4. Budget: S specialists of 10 trees each; the centralized reference has 10 x S trees.

Only the label column of each training file is read.
  python scripts/ev1_partition_n14.py
Out: results/ev1_partition_n14.txt
"""
import io
import os
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
N, SMALL, TREES = 14, 100, 10
DATASETS = [("ERENO", "all_in_one_ereno_train.csv"),
            ("CICIDS2017, improved", "all_in_one_ev1_cicids_train.csv"),
            ("Morris-4", "all_in_one_ev1_morris4_train.csv"),
            ("CICIoT2023", "all_in_one_ev1_ciciot_train.csv")]


def label_counts(path):
    """Class values in declaration order, the benign class (first data row) and row counts."""
    classes, counts, data, first = None, None, False, None
    with open(path) as f:
        for line in f:
            if not data:
                s = line.strip()
                if s.lower().startswith("@attribute") and "{" in s:
                    classes = [c.strip().strip("'\"") for c in s[s.index("{") + 1:s.rindex("}")].split(",")]
                elif s.lower() == "@data":
                    data, idx, counts = True, {c: i for i, c in enumerate(classes)}, [0] * len(classes)
                continue
            s = line.rstrip("\n")
            if s:
                c = idx[s[s.rindex(",") + 1:].strip().strip("'\"")]
                counts[c] += 1
                first = c if first is None else first
    return classes, first, counts


def specialists(classes, benign, counts):
    """[(name, [type codes], rows)] in placement order, and the nodes of each."""
    types = [c for c in range(len(classes)) if c != benign and counts[c] > 0]
    big = [c for c in types if counts[c] >= SMALL]
    small = [c for c in types if counts[c] < SMALL]
    sp = [(classes[c], [c], counts[c]) for c in big]
    if small:
        sp.append(("shared (" + ", ".join(classes[c] for c in small) + ")", small,
                   sum(counts[c] for c in small)))
    order = sorted(range(len(sp)), key=lambda k: (-sp[k][2], min(sp[k][1])))
    sp = [sp[k] for k in order]
    if len(sp) < N:                                   # pipeline rule: extra nodes to the largest
        copies = [1] * len(sp)
        for i in range(N - len(sp)):
            copies[i % len(sp)] += 1
        units, node = [], 0
        for (name, cs, rows), k in zip(sp, copies):
            for part in np.array_split(np.arange(rows), k):
                units.append((name, cs, len(part), node)); node += 1
    else:
        # contiguous blocks: q or q + 1 consecutive specialists per node, the larger blocks
        # on the last nodes (the smallest specialists)
        q, r = divmod(len(sp), N)
        node_of = [i for i, k in enumerate([q] * (N - r) + [q + 1] * r) for _ in range(k)]
        units = [(name, cs, rows, node_of[j]) for j, (name, cs, rows) in enumerate(sp)]
    return units


def main():
    print(f"Rule: one specialist per type; types under {SMALL} training rows share one; "
          f"placement by decreasing rows, contiguous blocks over {N} nodes; {TREES} trees each.\n")
    for name, fn in DATASETS:
        classes, benign, counts = label_counts(os.path.join(ROOT, fn))
        units = specialists(classes, benign, counts)
        n_types = sum(1 for c in range(len(classes)) if c != benign and counts[c] > 0)
        print(f"== {name} ({fn}): {n_types} attack types, {len(units)} specialists, "
              f"centralized reference {TREES * len(units)} trees; benign {counts[benign]:,}")
        for node in range(N):
            mine = [u for u in units if u[3] == node]
            print(f"  node {node + 1:2d}: " + "; ".join(f"{u[0]} ({u[2]:,})" for u in mine))
        print()


if __name__ == "__main__":
    os.chdir(ROOT)
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    print(buf.getvalue(), end="")
    with open("results/ev1_partition_n14.txt", "w") as f:
        f.write(buf.getvalue())
