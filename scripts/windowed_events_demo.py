#!/usr/bin/env python3
"""Demo: time-windowed intrusion_detected events with window_start/window_end.

Standalone experiment - does NOT touch main_dist.py or any config that produces the
paper's current results. It shows what the monitor's event stream looks like once a
round consumes a TIME SLICE of the traffic instead of re-evaluating the whole test set.

Setup: N=14 (2 specialists per attack, 7 ERENO attacks), k>=2 corroboration.
  * the absolute-time column F1 is kept as METADATA only - it never enters the
    feature vector (the 24 GRASP features are unchanged);
  * the test split is sorted by F1 and cut into fixed windows of DELTA seconds;
  * a window with at least one corroborated flag emits one intrusion_detected event;
  * simulated time is anchored to SIM_EPOCH, so the UTC stamps are synthetic.

Out: results/windowed_events_demo.jsonl (+ printed sample and summary)
"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
import python.util as util

FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
TRAIN, TEST = "all_in_one_ereno_train", "all_in_one_ereno_test"
SEED, NODES, NBR = 42, 14, 10
BENIGN_CAP = 500_000
K = 2                       # corroboration threshold (k>=2, the redundant operating point)
DELTA = 1.0                 # round window, seconds (~T0 of GOOSE, ~4.7k SV samples)
SIM_EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
OUT = "results/windowed_events_demo.jsonl"
PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 4}


def iso(t_rel):
    return (SIM_EPOCH + dt.timedelta(seconds=float(t_rel))).isoformat(timespec="microseconds")


def main():
    rng = np.random.default_rng(SEED)
    nc = util.normal_class

    print("[win] loading train...")
    Xtr_raw, ytr, cv = util.load_arff(f"{TRAIN}.csv")
    Xtr = util.filter_features(Xtr_raw, FEATURES); del Xtr_raw
    nidx = np.where(ytr == nc)[0]
    if len(nidx) > BENIGN_CAP:
        drop = rng.permutation(nidx)[BENIGN_CAP:]
        keep = np.ones(len(ytr), bool); keep[drop] = False
        Xtr, ytr = Xtr[keep], ytr[keep]
    attacks = sorted(int(c) for c in np.unique(ytr) if c != nc)
    benign_slices = np.array_split(rng.permutation(np.where(ytr == nc)[0]), NODES)

    # 14 specialists = 2 per attack (the N=14 redundant configuration)
    spec_attack = []
    for a in attacks:
        spec_attack += [a, a]
    spec_attack = spec_attack[:NODES]
    print(f"[win] training {NODES} specialists (2 per attack)...")
    boosters = []
    for cid in range(NODES):
        a = spec_attack[cid]
        a_idx = np.where(ytr == a)[0]
        peers = [c for c in range(NODES) if spec_attack[c] == a]
        shard = np.array_split(rng.permutation(a_idx), len(peers))[peers.index(cid)]
        idx = np.concatenate([shard, benign_slices[cid]])
        lab = (ytr[idx] != nc).astype(int)
        p = dict(PARAMS, scale_pos_weight=(lab == 0).sum() / max(lab.sum(), 1))
        boosters.append(xgb.train(p, xgb.DMatrix(Xtr[idx], label=lab), num_boost_round=NBR))
    del Xtr, ytr

    print("[win] loading test (keeping F1 as metadata)...")
    Xte_raw, yte, _ = util.load_arff(f"{TEST}.csv")
    util.normal_class = nc
    t_te = Xte_raw[:, 0].copy()                 # F1 = Time: metadata, NOT a feature
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw

    print("[win] scoring the whole test once, per specialist...")
    d = xgb.DMatrix(Xte); del Xte
    fired = np.vstack([(b.predict(d) >= 0.5) for b in boosters])     # (14, n_test) bool
    votes = fired.sum(axis=0).astype(np.int16)
    flagged = votes >= K

    # --- time slicing: round r consumes [t0 + r*DELTA, t0 + (r+1)*DELTA) ---
    order = np.argsort(t_te, kind="stable")     # file comes in per-scenario blocks
    t_sorted = t_te[order]
    t0 = float(t_sorted[0])
    win = np.floor((t_sorted - t0) / DELTA).astype(np.int64)
    flagged_s, votes_s, fired_s, y_s = flagged[order], votes[order], fired[:, order], yte[order]

    n_windows = int(win[-1]) + 1
    bounds = np.searchsorted(win, np.arange(n_windows + 1))
    print(f"[win] {len(t_sorted):,} samples -> {n_windows:,} windows of {DELTA}s "
          f"(span {t_sorted[-1]-t0:.0f}s)")

    os.makedirs("results", exist_ok=True)
    seq, n_events, n_nonempty, n_true = 0, 0, 0, 0
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in range(n_windows):
            lo, hi = bounds[r], bounds[r + 1]
            if hi == lo:
                continue                         # empty window: no traffic, no event
            n_nonempty += 1
            sel = slice(lo, hi)
            fl = flagged_s[sel]
            if not fl.any():
                continue
            if (y_s[sel][fl] != nc).any():
                n_true += 1
            det = np.where(fired_s[:, sel][:, fl].any(axis=1))[0]
            by_attack = {}
            for s in det:
                by_attack[cv[spec_attack[s]]] = by_attack.get(cv[spec_attack[s]], 0) + \
                    int(fired_s[s, sel][fl].sum())
            attack = max(by_attack, key=by_attack.get)
            seq += 1; n_events += 1
            ev = {
                "seq": seq,
                "ts": iso(t0 + (r + 1) * DELTA),          # emitted at window close
                "type": "intrusion_detected",
                "round": r,
                "window_start": iso(t0 + r * DELTA),
                "window_end": iso(t0 + (r + 1) * DELTA),
                "window_samples": int(hi - lo),
                "attack": attack,
                "k_votes": int(votes_s[sel][fl].max()),
                "n_flags": int(fl.sum()),
                "detector_nodes": [int(x) for x in det],
                "source_node": None,
                "confidence": None,
            }
            fh.write(json.dumps(ev) + "\n")
            if n_events <= 8:
                print(json.dumps(ev))

    print(f"\n[win] windows: {n_windows:,} total | {n_nonempty:,} with traffic "
          f"({100*n_nonempty/n_windows:.1f}%) | {n_events:,} emitted events")
    print(f"[win] events whose window truly contained an attack: {n_true:,}/{n_events:,} "
          f"({100*n_true/max(n_events,1):.1f}%)")
    print(f"[win] -> {OUT}")


if __name__ == "__main__":
    main()
