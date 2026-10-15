#!/usr/bin/env python3
"""Runtime switching scenario (14 nodes): per-node detection, every round, for OR and k >= 2.

The run conf/experiments/ereno_switching_scenario_n14.yaml dumps, after every round, the
pool each node holds (results/switching_scenario_n14/pools.jsonl, boosters by content hash
in boosters/). The fusion rule affects only the decision, so the same run serves k = 1 and
k >= 2. This script scores every distinct booster once on the full ERENO test file (24
GRASP columns, lean reader), then, for every round and node, the k-of-n decision of that
node's pool:

  python scripts/switching_scenario.py score   # booster predictions -> results/.cache/
  python scripts/switching_scenario.py         # per-node table and summary

Out: results/switching_scenario_n14.csv (round, mode, node, reachable, boosters, k, F1,
recall, FPR, recall of masquerade as fault) and results/switching_scenario_n14.txt.
"""
import csv
import io
import json
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
import numpy as np  # noqa: E402

from k_fusion_audit import FEATURES, TESTFILE, lean_load_arff, stats  # noqa: E402

DUMP = os.path.join(HERE, "results", "switching_scenario_n14")
CACHE = os.path.join(HERE, "results", ".cache", "switching_scenario_preds.npz")
OUT = os.path.join(HERE, "results", "switching_scenario_n14")
NEW_ATTACK = "masquerade_fake_fault"    # specialists on nodes 12 and 13, deployed at round 11


def phase_score():
    import xgboost as xgb
    os.chdir(HERE)
    Xfull, y, classes = lean_load_arff(f"{TESTFILE}.csv")
    X = Xfull[:, [f - 1 for f in sorted(FEATURES)]]
    del Xfull
    d = xgb.DMatrix(X)
    del X
    keys = sorted(f[:-4] for f in os.listdir(os.path.join(DUMP, "boosters")))
    preds = []
    for h in keys:
        b = xgb.Booster()
        with open(os.path.join(DUMP, "boosters", h + ".bin"), "rb") as fh:
            b.load_model(bytearray(fh.read()))
        preds.append(b.predict(d) >= 0.5)
        print(f"[score] booster {h}: {int(preds[-1].sum()):,} alarms", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, F=np.column_stack(preds), keys=np.array(keys),
                        y=y.astype(np.int16), cv=np.array(classes))


def main():
    z = np.load(CACHE, allow_pickle=False)
    F, keys, y, cv = z["F"], list(z["keys"]), z["y"], list(z["cv"])
    col = {h: i for i, h in enumerate(keys)}
    nc = int(y[0])
    atk = y != nc
    new = y == cv.index(NEW_ATTACK)
    rounds = [json.loads(line) for line in open(os.path.join(DUMP, "pools.jsonl"))]
    print(f"test: {len(y):,} samples; {len(keys)} distinct boosters; {len(rounds)} rounds")
    rows, cache = [], {}
    for r in rounds:
        active = r["active"]
        for node, hs in r["pools"].items():
            node = int(node)
            if not hs:
                continue
            for k in (1, 2):
                key = (frozenset(hs), k)
                if key not in cache:
                    votes = F[:, [col[h] for h in hs]].sum(axis=1)
                    pred = votes >= k
                    f1, rec, _, fpr, _ = stats(pred, atk)
                    cache[key] = (f1, rec, fpr, 100 * float(pred[new].mean()))
                f1, rec, fpr, rnew = cache[key]
                rows.append([r["round"], r["mode"], node,
                             int(active is None or node in active), len(hs), k,
                             round(f1, 4), round(rec, 4), round(fpr, 4), round(rnew, 4)])
    with open(OUT + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "mode", "node", "reachable", "boosters", "k",
                    "f1", "recall", "fpr", "recall_new_attack"])
        w.writerows(rows)
    for k in (1, 2):
        print(f"\nk {'>=' if k > 1 else '='} {k}: per round, mean over the nodes deployed "
              f"(reachable | unreachable) -- F1 / recall / recall of {NEW_ATTACK}")
        for rr in sorted({row[0] for row in rows}):
            sel = [row for row in rows if row[0] == rr and row[5] == k]
            def m(s, i):
                return f"{np.mean([x[i] for x in s]):6.2f}" if s else "   -  "
            up = [x for x in sel if x[3]]
            dn = [x for x in sel if not x[3]]
            print(f"  r{rr:>2} {sel[0][1][:3]} n={len(sel):>2} ({len(up):>2}|{len(dn):>2}) "
                  f"F1 {m(sel, 6)} ({m(up, 6)}|{m(dn, 6)})  rec {m(sel, 7)}  "
                  f"new {m(sel, 9)}  boosters {min(x[4] for x in sel)}-{max(x[4] for x in sel)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["score"]:
        phase_score()
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        open(OUT + ".txt", "w").write(buf.getvalue())
        print(buf.getvalue())
