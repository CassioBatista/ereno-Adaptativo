#!/usr/bin/env python3
"""Demo: the full adaptive round-trip as a single monitor event timeline.

Standalone experiment - does NOT touch main_dist.py or the configs behind the paper's
results. Extends windowed_events_demo.py with the three event types interleaved on one
time axis (1 s windows over the ERENO test stream):

  FL (14 nodes) -> node_failure -> architecture_change FL->GL (fail-fast)
  -> further failures, shrinking 14 -> 3 (GL absorbs them)
  -> nodes return 3 -> 14 -> architecture_change GL->FL (careful: unanimity + dwell)

The detection pool follows the mode, which is the whole point:
  * FL: only the specialists of the ACTIVE nodes are in the aggregate, so each lost
    node removes its specialist and k>=2 loses redundancy;
  * GL: the RETAINED UNION seeded at the switch keeps every booster ever aggregated,
    including those of nodes that died later - detection stays flat while shrinking.

Time is metadata only (column F1); the 24 GRASP features are unchanged.
Out: results/adaptive_events_demo.jsonl (+ printed timeline and per-phase summary)
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
K = 2
DELTA = 1.0
SIM_EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
OUT = "results/adaptive_events_demo.jsonl"
PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 4}

# ---- scenario (round = 1 s window index) -------------------------------------
FAIL_AT = {600: 13, 900: 12, 1200: 11, 1500: 10, 1800: 9, 2100: 8,
           2400: 7, 2700: 6, 3000: 5, 3300: 4, 3600: 3}      # 14 -> 3 nodes
RETURN_AT = {4000: 3, 4040: 4, 4080: 5, 4120: 6, 4160: 7, 4200: 8,
             4240: 9, 4280: 10, 4320: 11, 4360: 12, 4400: 13}  # 3 -> 14 nodes
DETECT_LAG = 2      # fail-fast: FL->GL commits 2 rounds after the first failure
DWELL = 3           # careful recovery: GL->FL after full membership stays DWELL rounds


def iso(t):
    return (SIM_EPOCH + dt.timedelta(seconds=float(t))).isoformat(timespec="microseconds")


def main():
    rng = np.random.default_rng(SEED)
    nc = util.normal_class

    print("[adapt] loading train...")
    Xtr_raw, ytr, cv = util.load_arff(f"{TRAIN}.csv")
    Xtr = util.filter_features(Xtr_raw, FEATURES); del Xtr_raw
    nidx = np.where(ytr == nc)[0]
    if len(nidx) > BENIGN_CAP:
        drop = rng.permutation(nidx)[BENIGN_CAP:]
        keep = np.ones(len(ytr), bool); keep[drop] = False
        Xtr, ytr = Xtr[keep], ytr[keep]
    attacks = sorted(int(c) for c in np.unique(ytr) if c != nc)
    benign_slices = np.array_split(rng.permutation(np.where(ytr == nc)[0]), NODES)
    spec_attack = []
    for a in attacks:
        spec_attack += [a, a]
    spec_attack = spec_attack[:NODES]

    print(f"[adapt] training {NODES} specialists (2 per attack)...")
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

    print("[adapt] loading test (F1 kept as metadata)...")
    Xte_raw, yte, _ = util.load_arff(f"{TEST}.csv")
    util.normal_class = nc
    t_te = Xte_raw[:, 0].copy()
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    d = xgb.DMatrix(Xte); del Xte
    print("[adapt] scoring the test once, per specialist...")
    fired = np.vstack([(b.predict(d) >= 0.5) for b in boosters])

    order = np.argsort(t_te, kind="stable")
    ts_sorted = t_te[order]
    t0 = float(ts_sorted[0])
    win = np.floor((ts_sorted - t0) / DELTA).astype(np.int64)
    fired_s, y_s = fired[:, order], yte[order]
    n_windows = int(win[-1]) + 1
    bounds = np.searchsorted(win, np.arange(n_windows + 1))

    # ---- drive the scenario ---------------------------------------------------
    active = set(range(NODES))
    retained = set(range(NODES))      # union retained from the federated phase
    mode = "federated"
    switch_pending = None
    full_since = None
    seq = 0
    phase_stat = {}
    events = []

    for r in range(n_windows):
        ws, we = t0 + r * DELTA, t0 + (r + 1) * DELTA
        base = {"seq": 0, "ts": iso(we), "round": r,
                "window_start": iso(ws), "window_end": iso(we)}

        # -- membership changes (every one of them is an explicit event) --
        if r in FAIL_AT:
            n = FAIL_AT[r]
            active.discard(n)
            seq += 1
            events.append(dict(base, seq=seq, type="node_failure", mode=mode,
                               reason="node_failure", failed_nodes=[n],
                               active_nodes=sorted(active), n_active=len(active)))
            if mode == "federated" and switch_pending is None:
                switch_pending = r + DETECT_LAG       # fail-fast
        if r in RETURN_AT:
            n = RETURN_AT[r]
            active.add(n)
            seq += 1
            events.append(dict(base, seq=seq, type="node_recovery", mode=mode,
                               reason="recovery", recovered_nodes=[n],
                               active_nodes=sorted(active), n_active=len(active)))
            if len(active) == NODES and full_since is None:
                full_since = r
        if len(active) < NODES:
            full_since = None

        # -- mode transitions --
        if switch_pending is not None and r >= switch_pending and mode == "federated":
            mode = "gossip"
            switch_pending = None
            seq += 1
            events.append(dict(base, seq=seq, type="architecture_change", mode="gossip",
                               from_mode="federated", to_mode="gossip",
                               reason="node_failure",
                               failed_nodes=sorted(set(range(NODES)) - active),
                               active_nodes=sorted(active)))
        if mode == "gossip" and full_since is not None and r - full_since >= DWELL:
            mode = "federated"
            full_since = None
            seq += 1
            events.append(dict(base, seq=seq, type="architecture_change", mode="federated",
                               from_mode="gossip", to_mode="federated",
                               reason="recovery", failed_nodes=[],
                               active_nodes=sorted(active)))

        # -- detection pool follows the mode --
        pool = sorted(active) if mode == "federated" else sorted(retained)

        lo, hi = bounds[r], bounds[r + 1]
        if hi == lo:
            continue
        sel = slice(lo, hi)
        sub = fired_s[pool, sel]
        votes = sub.sum(axis=0)
        fl = votes >= K
        yy = y_s[sel]

        ph = ("FL 14" if mode == "federated" and len(active) == NODES and r < 600 else
              "GL shrinking" if mode == "gossip" and len(active) > 3 and r < 4000 else
              "GL at 3" if mode == "gossip" and len(active) <= 3 else
              "GL recovering" if mode == "gossip" else
              "FL restored")
        st = phase_stat.setdefault(ph, {"atk": 0, "tp": 0, "ben": 0, "fp": 0, "nodes": len(active)})
        st["atk"] += int((yy != nc).sum()); st["tp"] += int((fl & (yy != nc)).sum())
        st["ben"] += int((yy == nc).sum()); st["fp"] += int((fl & (yy == nc)).sum())
        st["nodes"] = len(active)

        if fl.any():
            det = [pool[i] for i in np.where(sub[:, fl].any(axis=1))[0]]
            by = {}
            for s in det:
                by[cv[spec_attack[s]]] = by.get(cv[spec_attack[s]], 0) + int(sub[pool.index(s)][fl].sum())
            seq += 1
            events.append(dict(base, seq=seq, type="intrusion_detected", mode=mode,
                               attack=max(by, key=by.get), k_votes=int(votes[fl].max()),
                               n_flags=int(fl.sum()),
                               detector_specialists=det, active_nodes=sorted(active),
                               source_node=None))

    events.sort(key=lambda e: e["seq"])
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")

    print("\n=== timeline (control-plane events + first intrusion after each) ===")
    shown = 0
    for i, e in enumerate(events):
        if e["type"] != "intrusion_detected":
            print(json.dumps(e))
            nxt = next((x for x in events[i + 1:] if x["type"] == "intrusion_detected"), None)
            if nxt:
                print("   next intrusion -> round %d  attack=%s  k_votes=%d  n_flags=%d  pool=%d nodes"
                      % (nxt["round"], nxt["attack"], nxt["k_votes"], nxt["n_flags"],
                         len(nxt["active_nodes"])))
            shown += 1
    print("\n=== per-phase detection (k>=%d) ===" % K)
    print("%16s %7s %10s %10s" % ("phase", "nodes", "recall%", "FPR%"))
    for ph in ["FL 14", "GL shrinking", "GL at 3", "GL recovering", "FL restored"]:
        s = phase_stat.get(ph)
        if not s or not s["atk"]:
            continue
        print("%16s %7d %10.2f %10.3f"
              % (ph, s["nodes"], 100 * s["tp"] / s["atk"], 100 * s["fp"] / max(s["ben"], 1)))
    kinds = {}
    for e in events:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    print("\n[adapt] events:", kinds, "-> ", OUT)


if __name__ == "__main__":
    main()
