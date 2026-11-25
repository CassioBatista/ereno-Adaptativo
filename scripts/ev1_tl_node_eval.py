#!/usr/bin/env python3
"""Per-node detection over time from the holdings log of a pipeline run (node_holdings;
fd/holdings_log.py), evaluation-v1 section 9, per-sample metrics.

For every logged round and every node, the node's detector is the OR at 0.5 of the boosters
it holds at that round. On the full test split of the run's dataset:
  - recall on the attack types already present (a type is present once the block of a
    node holding its specialist has arrived: scenario.arrival_s, default 0);
  - recall on the types whose blocks arrive late, one figure per arrival time;
  - false-positive rate on the benign rows.
Every node is scored on the same rows (the declared simplification of section 9). Window
alarms (section 6) need the recomposed stream and are not computed here.

  python scripts/ev1_tl_node_eval.py <holdings_log_dir> [--phase 900]
Out: <dir>/node_eval.csv (round, t_s, node, node_mode, n_boosters, recall_present,
     recall_late_<t>, fpr), <dir>/node_eval.txt (summary per phase), <dir>/scores.npz
     (cache: one hit vector per booster).
"""
import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main(d, phase):
    import xgboost as xgb
    import ev1_step1 as E
    hdr = json.load(open(os.path.join(d, "header.json")))
    rounds = [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]
    X, y, classes, nc = E.load(f"{hdr['test_file']}.csv", hdr["features"])
    n_nodes = int(hdr["num_nodes"])
    arrival = {int(k): float(v) for k, v in (hdr.get("arrival_s") or {}).items()}
    code = {c: i for i, c in enumerate(classes)}
    node_types = [[code[s] for s in str(hdr["node_specialist"][i]).split("+") if s in code]
                  for i in range(n_nodes)]
    arr_of_type = {}
    for i, ts in enumerate(node_types):
        for c in ts:
            arr_of_type[c] = min(arr_of_type.get(c, np.inf), arrival.get(i, 0.0))
    late_times = sorted({a for a in arr_of_type.values() if a > 0})

    # one hit vector per booster, cached
    cache = os.path.join(d, "scores.npz")
    hits = dict(np.load(cache)) if os.path.exists(cache) else {}
    need = sorted({h for r in rounds for hs in r["holdings"] for h in hs} - set(hits))
    if need:
        dm = xgb.DMatrix(X)
        for h in need:
            b = xgb.Booster()
            b.load_model(os.path.join(d, "boosters", f"{h}.json"))
            hits[h] = b.predict(dm) >= 0.5
        np.savez_compressed(cache, **hits)
    del X
    ben = y == nc
    memo = {}

    def metrics(hset, t):
        key = (frozenset(hset), t)
        if key not in memo:
            pred = np.zeros(len(y), bool)
            for h in hset:
                pred |= hits[h]
            present = [c for c, a in arr_of_type.items() if a <= t]
            m = np.isin(y, present)
            out = {"recall_present": float(pred[m].mean()) if m.any() else np.nan,
                   "fpr": float(pred[ben].mean())}
            for a in late_times:
                ml = np.isin(y, [c for c, x in arr_of_type.items() if x == a])
                out[f"recall_late_{a:g}"] = float(pred[ml].mean()) if a <= t else np.nan
            memo[key] = out
        return memo[key]

    rows = []
    for r in rounds:
        t = r["t_s"] if r["t_s"] is not None else float(r["round"])
        nm = r["node_modes"] or [r["mode"]] * n_nodes
        rc = r.get("reachable") or [None] * n_nodes
        for i in range(n_nodes):
            rows.append({"round": r["round"], "t_s": t, "node": i + 1, "node_mode": nm[i],
                         "reachable": "" if rc[i] is None else int(rc[i]),
                         "n_boosters": len(r["holdings"][i]),
                         **metrics(r["holdings"][i], t)})
    cols = list(rows[0])
    with open(os.path.join(d, "node_eval.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c])
                             for c in cols) + "\n")

    print(f"run {hdr.get('run_name')}: {len(rounds)} rounds, {n_nodes} nodes, "
          f"holdings {hdr['node_holdings']}, arrival {arrival or 'none'}")
    print(f"types by arrival: " + "; ".join(
        f"{a:g} s: {', '.join(classes[c] for c, x in sorted(arr_of_type.items()) if x == a)}"
        for a in sorted(set(arr_of_type.values()))))
    late_cols = [f"recall_late_{a:g}" for a in late_times]
    by_reach = all(r["reachable"] != "" for r in rows)
    if by_reach:     # environment (same groups for every policy): R reachable / U not
        groups = (("U", lambda r: r["reachable"] == 0), ("R", lambda r: r["reachable"] == 1))
        legend = "U unreachable / R reachable: reaches the aggregator or a ring neighbour"
    else:
        groups = (("L", lambda r: r["node_mode"] == "local"),
                  ("X", lambda r: r["node_mode"] != "local"))
        legend = "L local / X in exchange"
    print(f"\nper phase of {phase:g} s: mean over the node-rounds, split by group ({legend})")
    print(f"{'phase':<12}{'group':<6}{'n':>5}{'boosters':>9}{'recall':>9}"
          + "".join(f"{'late ' + c.split('_')[-1]:>12}" for c in late_cols) + f"{'FPR %':>9}")
    t_max = max(r["t_s"] for r in rows)
    for k in range(int(t_max // phase) + 1):
        seg = [r for r in rows if k * phase <= r["t_s"] < (k + 1) * phase]
        for g, sel in groups:
            s = [r for r in seg if sel(r)]
            if not s:
                continue
            mean = lambda c: np.nanmean([r[c] for r in s]) if any(
                not np.isnan(r[c]) for r in s) else np.nan
            print(f"{f'{k * phase:g}-{(k + 1) * phase:g}':<12}{g:<6}{len(s):>5}"
                  f"{np.mean([r['n_boosters'] for r in s]):>9.1f}{100 * mean('recall_present'):>9.2f}"
                  + "".join(f"{100 * mean(c):>12.2f}" for c in late_cols)
                  + f"{100 * mean('fpr'):>9.3f}")
    print("\nper node at the last round of each phase: boosters / recall present / "
          + " / ".join(late_cols) + " / FPR % (mode)")
    for k in range(int(t_max // phase) + 1):
        seg = [r for r in rows if k * phase <= r["t_s"] < (k + 1) * phase]
        if not seg:
            continue
        last = max(r["round"] for r in seg)
        print(f"  t={max(r['t_s'] for r in seg):>6.0f} s (round {last}):")
        for r in [r for r in seg if r["round"] == last]:
            reach = "" if r["reachable"] == "" else ("R" if r["reachable"] else "U")
            print(f"    node {r['node']:2d} {r['node_mode'][0].upper()}{reach} {r['n_boosters']:3d}  "
                  f"{100 * r['recall_present']:6.2f}  "
                  + "  ".join("   —  " if np.isnan(r[c]) else f"{100 * r[c]:6.2f}" for c in late_cols)
                  + f"  {100 * r['fpr']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--phase", type=float, default=900.0)
    a = ap.parse_args()
    d = os.path.abspath(os.path.expanduser(a.log_dir))
    os.chdir(ROOT)
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d, a.phase)
    open(os.path.join(d, "node_eval.txt"), "w").write(buf.getvalue())
    print(buf.getvalue())
