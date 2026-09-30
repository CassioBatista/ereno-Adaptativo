#!/usr/bin/env python3
"""Two drivers of node reduction, as two separate event streams for the monitor.

  A. availability  — benign-only traffic. Nodes go inactive and come back; the only
                     alarms are FALSE POSITIVES, which is precisely the baseline the
                     monitor must learn to triage. Authority: ReSIDS itself (fail-fast).

  B. intrusion     — attack traffic with attribution. A node whose emitted traffic is
                     repeatedly and corroboratedly flagged is ISOLATED — but the IDS
                     never isolates: it reports, the monitor decides and commands, and
                     the removal appears as node_isolated (reason: intrusion).
                     Authority: the monitor.

Honest scope: `source_node` attribution does NOT exist in v2 (it is null in every real
event). Here each attack class is mapped to a synthetic emitter node so the driver can
be exercised end to end — that mapping is a v3 premise, not a measurement.

Out: results/events_A_availability.jsonl, results/events_B_intrusion.jsonl
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
SEED, NODES, NBR, K = 42, 14, 10, 2
BENIGN_CAP, DELTA = 500_000, 1.0
SIM_EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 4}

# A: scripted availability timeline (rounds are indices into the benign-only stream)
# NOTE: the benign-only stream has ~724 windows, so the whole arc must fit inside it —
# an earlier schedule put the last returns at rounds 760/820, which simply never
# happened, and the GL->FL return never fired.
A_FAIL = {100: 13, 200: 12, 300: 11}
A_BACK = {480: 11, 540: 12, 600: 13}
DETECT_LAG, DWELL, CMD_LATENCY = 2, 3, 3
# B: synthetic attribution (v3 premise) + isolation policy
ATTACK_SOURCE = {"injection": 11, "masquerade_fake_fault": 9, "masquerade_fake_normal": 8,
                 "random_replay": 12, "inverse_replay": 10, "high_StNum": 13,
                 "poisoned_high_rate": 7}
ISOLATE_AFTER = 5        # corroborated alarms attributed to the same node
ESCALATE = 10            # n_flags for an alarm to count as evidence


def iso(t):
    return (SIM_EPOCH + dt.timedelta(seconds=float(t))).isoformat(timespec="microseconds")


def main():
    rng = np.random.default_rng(SEED)
    nc = util.normal_class
    print("[2drv] loading train...")
    Xtr_raw, ytr, cv = util.load_arff(f"{TRAIN}.csv")
    Xtr = util.filter_features(Xtr_raw, FEATURES); del Xtr_raw
    nidx = np.where(ytr == nc)[0]
    if len(nidx) > BENIGN_CAP:
        drop = rng.permutation(nidx)[BENIGN_CAP:]
        keep = np.ones(len(ytr), bool); keep[drop] = False
        Xtr, ytr = Xtr[keep], ytr[keep]
    attacks = sorted(int(c) for c in np.unique(ytr) if c != nc)
    bslice = np.array_split(rng.permutation(np.where(ytr == nc)[0]), NODES)
    spec_attack = []
    for a in attacks:
        spec_attack += [a, a]
    spec_attack = spec_attack[:NODES]

    print("[2drv] training 14 specialists...")
    boosters = []
    for cid in range(NODES):
        a = spec_attack[cid]
        aidx = np.where(ytr == a)[0]
        peers = [c for c in range(NODES) if spec_attack[c] == a]
        shard = np.array_split(rng.permutation(aidx), len(peers))[peers.index(cid)]
        idx = np.concatenate([shard, bslice[cid]])
        lab = (ytr[idx] != nc).astype(int)
        p = dict(PARAMS, scale_pos_weight=(lab == 0).sum() / max(lab.sum(), 1))
        boosters.append(xgb.train(p, xgb.DMatrix(Xtr[idx], label=lab), num_boost_round=NBR))
    del Xtr, ytr

    print("[2drv] scoring test...")
    Xte_raw, yte, _ = util.load_arff(f"{TEST}.csv")
    util.normal_class = nc
    t_te = Xte_raw[:, 0].copy()
    Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
    d = xgb.DMatrix(Xte); del Xte
    fired = np.vstack([(b.predict(d) >= 0.5) for b in boosters])

    order = np.argsort(t_te, kind="stable")
    ts = t_te[order]; fired = fired[:, order]; y = yte[order]
    t0 = float(ts[0])
    win = np.floor((ts - t0) / DELTA).astype(np.int64)
    nw = int(win[-1]) + 1
    bnd = np.searchsorted(win, np.arange(nw + 1))

    benign_only, with_attack = [], []
    for w in range(nw):
        lo, hi = bnd[w], bnd[w + 1]
        if hi == lo:
            continue
        (benign_only if (y[lo:hi] == nc).all() else with_attack).append((w, lo, hi))
    print(f"[2drv] windows with traffic: {len(benign_only)} benign-only, {len(with_attack)} with attack")

    def alarm(lo, hi, pool):
        sub = fired[pool, lo:hi]
        votes = sub.sum(axis=0)
        fl = votes >= K
        if not fl.any():
            return None
        det = [pool[i] for i in np.where(sub[:, fl].any(axis=1))[0]]
        by = {}
        for s in det:
            by[cv[spec_attack[s]]] = by.get(cv[spec_attack[s]], 0) + int(sub[pool.index(s)][fl].sum())
        return dict(attack=max(by, key=by.get), k_votes=int(votes[fl].max()),
                    n_flags=int(fl.sum()), detector_specialists=det)

    # ---------- stream A: availability (benign only) ----------
    evs, seq, active, mode, pend, full_since = [], 0, set(range(NODES)), "federated", None, None
    retained = set(range(NODES))
    for r, (_, lo, hi) in enumerate(benign_only):
        ws, we = t0 + r * DELTA, t0 + (r + 1) * DELTA
        base = {"ts": iso(we), "round": r, "window_start": iso(ws), "window_end": iso(we)}
        if r in A_FAIL:
            active.discard(A_FAIL[r]); seq += 1
            evs.append(dict(base, seq=seq, type="node_failure", mode=mode, reason="node_failure",
                            failed_nodes=[A_FAIL[r]], active_nodes=sorted(active), n_active=len(active)))
            if mode == "federated" and pend is None:
                pend = r + DETECT_LAG
        if r in A_BACK:
            active.add(A_BACK[r]); seq += 1
            evs.append(dict(base, seq=seq, type="node_recovery", mode=mode, reason="recovery",
                            recovered_nodes=[A_BACK[r]], active_nodes=sorted(active), n_active=len(active)))
            if len(active) == NODES and full_since is None:
                full_since = r
        if len(active) < NODES:
            full_since = None
        if pend is not None and r >= pend and mode == "federated":
            mode, pend, seq = "gossip", None, seq + 1
            evs.append(dict(base, seq=seq, type="architecture_change", mode="gossip",
                            from_mode="federated", to_mode="gossip", reason="node_failure",
                            failed_nodes=sorted(set(range(NODES)) - active),
                            active_nodes=sorted(active), n_active=len(active)))
        # GL->FL is the MONITOR's call: ReSIDS only supplies the evidence (full membership
        # held for DWELL rounds); the monitor then commands, taking CMD_LATENCY rounds.
        # No autonomous fallback here on purpose — staying in GL is safe, only slower.
        if mode == "gossip" and full_since is not None and r - full_since >= DWELL + CMD_LATENCY:
            mode, full_since, seq = "federated", None, seq + 1
            evs.append(dict(base, seq=seq, type="architecture_change", mode="federated",
                            from_mode="gossip", to_mode="federated", reason="recovery",
                            failed_nodes=[], active_nodes=sorted(active), n_active=len(active),
                            detail=("MONITOR-COMMANDED: authority for GL->FL is the monitor's; "
                                    f"evidence = full membership for {DWELL} rounds")))
        pool = sorted(active) if mode == "federated" else sorted(retained)
        a = alarm(lo, hi, pool)
        if a:
            seq += 1
            evs.append(dict(base, seq=seq, type="intrusion_detected", mode=mode, **a,
                            active_nodes=sorted(active), n_active=len(active),
                            source_node=None, window_samples=int(hi - lo),
                            detail="benign-only window: this alarm is a FALSE POSITIVE"))
    with open("results/events_A_availability.jsonl", "w", encoding="utf-8") as fh:
        for e in evs:
            fh.write(json.dumps(e) + "\n")
    a_fp = sum(1 for e in evs if e["type"] == "intrusion_detected")
    print(f"[2drv] A: {len(evs)} events over {len(benign_only)} benign-only windows "
          f"({a_fp} false-positive alarms, {a_fp*100/max(len(benign_only),1):.1f}% of windows)")

    # ---------- stream B: intrusion -> monitor-commanded isolation ----------
    evs, seq, active, mode = [], 0, set(range(NODES)), "federated"
    retained, evidence, isolated = set(range(NODES)), {}, set()
    for r, (_, lo, hi) in enumerate(with_attack):
        ws, we = t0 + r * DELTA, t0 + (r + 1) * DELTA
        base = {"ts": iso(we), "round": r, "window_start": iso(ws), "window_end": iso(we)}
        pool = sorted(active) if mode == "federated" else sorted(retained)
        a = alarm(lo, hi, pool)
        if not a:
            continue
        src = ATTACK_SOURCE.get(a["attack"])
        src = src if src in active else None
        seq += 1
        evs.append(dict(base, seq=seq, type="intrusion_detected", mode=mode, **a,
                        active_nodes=sorted(active), n_active=len(active),
                        source_node=src, window_samples=int(hi - lo),
                        detail="notification only; containment is operator/monitor-gated"))
        if src is not None and a["k_votes"] >= K and a["n_flags"] >= ESCALATE:
            evidence[src] = evidence.get(src, 0) + 1
            if evidence[src] == ISOLATE_AFTER:
                active.discard(src); isolated.add(src); seq += 1
                evs.append(dict(base, seq=seq, type="node_isolated", mode=mode,
                                reason="intrusion", failed_nodes=[src], source_node=src,
                                active_nodes=sorted(active), n_active=len(active),
                                attack=a["attack"], k_votes=a["k_votes"],
                                detail=("monitor-commanded isolation after "
                                        f"{ISOLATE_AFTER} corroborated alarms attributed to node {src}")))
                # Intrusion NEVER triggers the autonomous fail-fast: only inactivity in FL
                # does. If the monitor wants GL after the membership shrank, it COMMANDS it.
                if mode == "federated" and len(isolated) >= 2:
                    mode, seq = "gossip", seq + 1
                    evs.append(dict(base, seq=seq, type="architecture_change", mode="gossip",
                                    from_mode="federated", to_mode="gossip", reason="intrusion",
                                    failed_nodes=sorted(isolated), active_nodes=sorted(active),
                                    n_active=len(active),
                                    detail=("MONITOR-COMMANDED switch (not fail-fast): intrusion "
                                            "never drives the autonomous transition")))
    with open("results/events_B_intrusion.jsonl", "w", encoding="utf-8") as fh:
        for e in evs:
            fh.write(json.dumps(e) + "\n")
    import collections
    print(f"[2drv] B: {len(evs)} events over {len(with_attack)} attack windows "
          f"-> {collections.Counter(e['type'] for e in evs)}")
    print(f"[2drv]    isolated nodes: {sorted(isolated)}  (evidence counts: {evidence})")


if __name__ == "__main__":
    main()
