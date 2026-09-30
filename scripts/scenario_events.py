#!/usr/bin/env python3
"""Generate a monitor event stream from a declarative SCENARIO file.

Supersedes scripts/two_driver_events.py, whose schedule was hard-coded — which is how an
earlier run silently lost the tail of the recovery arc: the last returns were scheduled
at rounds 760/820 while the benign-only stream only has ~724 windows, so they never
happened and the GL->FL return never fired. Here the schedule is data
(conf/scenarios/*.yaml, validated against schemas/scenario.schema.json AND against the
number of windows the selected stream actually has), so that failure mode is impossible.

Authority rule enforced (docs/decentralized_monitoring.md §1.1):
  * node inactivity while in FL -> GL is the ONLY autonomous action (decided_by=autonomous);
  * further losses in GL, intrusion-driven isolation and the GL->FL return are all
    monitor-governed (decided_by=monitor).

Usage:
  python scripts/scenario_events.py conf/scenarios/availability.yaml conf/scenarios/intrusion.yaml
Out: results/events_<name>.jsonl (one per scenario; the model is trained once for all)
"""
import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
import yaml
from jsonschema import Draft202012Validator

import python.util as util
from provenance import file_ref, write_manifest

FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
TRAIN, TEST = "all_in_one_ereno_train", "all_in_one_ereno_test"
SEED, NBR, BENIGN_CAP = 42, 10, 500_000
SIM_EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 4}
SCHEMA = "schemas/scenario.schema.json"


def iso(t):
    return (SIM_EPOCH + dt.timedelta(seconds=float(t))).isoformat(timespec="microseconds")


def load_scenario(path):
    sc = yaml.safe_load(open(path, encoding="utf-8"))
    Draft202012Validator(json.load(open(SCHEMA, encoding="utf-8"))).validate(sc)
    sc.setdefault("fusion_k", 2)
    sc.setdefault("data", {})
    sc["data"].setdefault("train", TRAIN)
    sc["data"].setdefault("test", TEST)
    sc["data"].setdefault("benign_cap", BENIGN_CAP)
    sc.setdefault("model", {})
    m = sc["model"]
    m.setdefault("seed", SEED)
    m.setdefault("num_boost_round", NBR)
    m.setdefault("max_depth", 4)
    m.setdefault("eta", 0.1)
    if "features" not in m:
        if "features_file" in m:
            m["features"] = json.load(open(m["features_file"], encoding="utf-8"))["features"]
        else:
            m["features"] = list(FEATURES)
    sc.setdefault("sim_epoch", "2026-01-01T00:00:00Z")
    sc.setdefault("timing", {})
    sc["timing"].setdefault("detect_lag", 2)
    sc["timing"].setdefault("dwell", 3)
    sc["timing"].setdefault("cmd_latency", 3)
    return sc


def validate_schedule(sc, n_windows):
    """The check that was missing: the schedule must FIT the stream, and the up/down
    state machine must be consistent."""
    errs, up = [], set(range(sc["nodes"]))
    for i, s in enumerate(sorted(sc["schedule"], key=lambda x: x["round"])):
        where = f"schedule[{i}] round={s['round']} {s['event']} node={s['node']}"
        if s["round"] >= n_windows:
            errs.append(f"{where}: round beyond the stream, which has {n_windows} windows")
        if s["node"] >= sc["nodes"]:
            errs.append(f"{where}: node index >= nodes ({sc['nodes']})")
        if s["event"] == "node_failure":
            if s["node"] not in up:
                errs.append(f"{where}: node is already down")
            up.discard(s["node"])
        else:
            if s["node"] in up:
                errs.append(f"{where}: node is already up")
            up.add(s["node"])
    if errs:
        raise SystemExit("[scenario] INVALID:\n  " + "\n  ".join(errs))
    if sc["schedule"]:
        last = max(s["round"] for s in sc["schedule"])
        need = last + sc["timing"]["dwell"] + sc["timing"]["cmd_latency"] + 1
        if len(up) == sc["nodes"] and need >= n_windows:
            print(f"[scenario] WARNING: the GL->FL return would land at round ~{need}, "
                  f"but the stream ends at {n_windows - 1} — it will not fire.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("--no-manifest", action="store_true",
                    help="skip provenance (hashing the ~1.6 GB inputs)")
    a = ap.parse_args()
    paths = a.scenarios
    scs = [load_scenario(p) for p in paths]
    print("[scenario] loaded:", ", ".join(s["name"] for s in scs))

    # every scenario in one call must share the data/model binding: the model is trained once
    key = lambda s: (s["data"]["train"], s["data"]["test"], s["data"]["benign_cap"],
                     tuple(s["model"]["features"]), s["model"]["seed"],
                     s["model"]["num_boost_round"], s["model"]["max_depth"],
                     s["model"]["eta"], s["nodes"])
    if len({key(s) for s in scs}) > 1:
        raise SystemExit("[scenario] scenarios in one call must share data/model/nodes; "
                         "run them separately")
    B, M, N = scs[0]["data"], scs[0]["model"], scs[0]["nodes"]
    feats = list(M["features"])
    params = {"max_depth": M["max_depth"], "eta": M["eta"],
              "objective": "binary:logistic", "seed": M["seed"], "nthread": 4}

    rng = np.random.default_rng(M["seed"])
    nc = util.normal_class
    print(f"[scenario] loading train ({B['train']}, {len(feats)} features)...")
    Xtr_raw, ytr, cv = util.load_arff(f"{B['train']}.csv")
    Xtr = util.filter_features(Xtr_raw, feats); del Xtr_raw
    nidx = np.where(ytr == nc)[0]
    if len(nidx) > B["benign_cap"]:
        drop = rng.permutation(nidx)[B["benign_cap"]:]
        keep = np.ones(len(ytr), bool); keep[drop] = False
        Xtr, ytr = Xtr[keep], ytr[keep]
    attacks = sorted(int(c) for c in np.unique(ytr) if c != nc)
    bslice = np.array_split(rng.permutation(np.where(ytr == nc)[0]), N)
    spec_attack = []
    for at in attacks:
        spec_attack += [at, at]
    spec_attack = spec_attack[:N]

    print(f"[scenario] training {N} specialists...")
    boosters = []
    for cid in range(N):
        at = spec_attack[cid]
        aidx = np.where(ytr == at)[0]
        peers = [c for c in range(N) if spec_attack[c] == at]
        shard = np.array_split(rng.permutation(aidx), len(peers))[peers.index(cid)]
        idx = np.concatenate([shard, bslice[cid]])
        lab = (ytr[idx] != nc).astype(int)
        p = dict(params, scale_pos_weight=(lab == 0).sum() / max(lab.sum(), 1))
        boosters.append(xgb.train(p, xgb.DMatrix(Xtr[idx], label=lab),
                                  num_boost_round=M["num_boost_round"]))
    del Xtr, ytr

    print("[scenario] scoring test...")
    Xte_raw, yte, _ = util.load_arff(f"{B['test']}.csv")
    util.normal_class = nc
    t_te = Xte_raw[:, 0].copy()
    Xte = util.filter_features(Xte_raw, feats); del Xte_raw
    d = xgb.DMatrix(Xte); del Xte
    fired = np.vstack([(b.predict(d) >= 0.5) for b in boosters])
    order = np.argsort(t_te, kind="stable")
    ts, fired, y = t_te[order], fired[:, order], yte[order]
    t0 = float(ts[0])

    inputs = None
    if not a.no_manifest:
        print("[scenario] hashing inputs (cached by size+mtime)...")
        inputs = {"train": file_ref(f"{B['train']}.csv", cache=True),
                  "test": file_ref(f"{B['test']}.csv", cache=True)}
        if "features_file" in M:
            inputs["features_file"] = file_ref(M["features_file"])

    for sc, path in zip(scs, paths):
        out, nwin, types = gen(sc, ts, fired, y, nc, cv, spec_attack, t0)
        if a.no_manifest:
            continue
        man = write_manifest(
            f"results/manifest_{sc['name']}.json",
            scenario_path=path, scenario=sc, inputs=inputs,
            model={k: M[k] for k in ("seed", "num_boost_round", "max_depth", "eta")}
            | {"features": M["features"], "n_features": len(M["features"]),
               "features_file": M.get("features_file"), "specialists": N,
               "benign_cap": B["benign_cap"], "fusion_k": sc["fusion_k"]},
            stream={"selector": sc["stream"], "window_seconds": sc["window_seconds"],
                    "windows": nwin, "sim_epoch": sc["sim_epoch"],
                    "timing": sc["timing"]},
            output={**file_ref(out), "events": sum(types.values()), "types": dict(types)})
        print(f"[scenario]   manifest -> results/manifest_{sc['name']}.json "
              f"(output sha256 {man['output']['sha256'][:12]}…)")


def gen(sc, ts, fired, y, nc, cv, spec_attack, t0):
    N, K, T = sc["nodes"], sc["fusion_k"], sc["timing"]
    win = np.floor((ts - t0) / sc["window_seconds"]).astype(np.int64)
    bnd = np.searchsorted(win, np.arange(int(win[-1]) + 2))
    sel = []
    for w in range(int(win[-1]) + 1):
        lo, hi = bnd[w], bnd[w + 1]
        if hi == lo:
            continue
        benign = bool((y[lo:hi] == nc).all())
        if sc["stream"] == "all" or (sc["stream"] == "benign_only") == benign:
            sel.append((w, lo, hi))        # keep the ORIGINAL window index for the clock
    print(f"[scenario] {sc['name']}: stream '{sc['stream']}' -> {len(sel)} windows")
    validate_schedule(sc, len(sel))

    fail = {s["round"]: s["node"] for s in sc["schedule"] if s["event"] == "node_failure"}
    back = {s["round"]: s["node"] for s in sc["schedule"] if s["event"] == "node_recovery"}
    attr = {k: v for k, v in (sc.get("attribution") or {}).items()}
    isol_cfg = sc.get("isolation") or {}

    evs, seq, active, retained = [], 0, set(range(N)), set(range(N))
    mode, pend, full_since, evidence, isolated = "federated", None, None, Counter(), set()
    # `r` is the scenario round (consecutive, so the schedule indexes it); `w` is the
    # window's position in the ORIGINAL trace, and the timestamps must come from w — the
    # selected windows are NOT contiguous in time (the two streams are disjoint views of
    # one trace, interleaved in it), so deriving the clock from r would fabricate a
    # timeline and break correlation with SOE/disturbance records.
    for r, (w, lo, hi) in enumerate(sel):
        ws, we = t0 + w * sc["window_seconds"], t0 + (w + 1) * sc["window_seconds"]
        base = {"ts": iso(we), "round": r, "window_start": iso(ws), "window_end": iso(we)}

        if r in fail:
            active.discard(fail[r]); seq += 1
            evs.append(dict(base, seq=seq, type="node_failure", mode=mode,
                            reason="node_failure",
                            failed_nodes=[fail[r]], active_nodes=sorted(active),
                            n_active=len(active)))
            if mode == "federated" and pend is None:
                pend = r + T["detect_lag"]
        if r in back:
            active.add(back[r]); seq += 1
            evs.append(dict(base, seq=seq, type="node_recovery", mode=mode,
                            reason="recovery",
                            recovered_nodes=[back[r]], active_nodes=sorted(active),
                            n_active=len(active)))
            if len(active) == N and full_since is None:
                full_since = r
        if len(active) < N:
            full_since = None

        # the ONE autonomous transition
        if pend is not None and r >= pend and mode == "federated":
            mode, pend, seq = "gossip", None, seq + 1
            evs.append(dict(base, seq=seq, type="architecture_change", mode="gossip",
                            from_mode="federated", to_mode="gossip", reason="node_failure",
                            decided_by="autonomous", failed_nodes=sorted(set(range(N)) - active),
                            active_nodes=sorted(active), n_active=len(active),
                            detail="autonomous fail-fast: inactivity while in FL"))
        # GL->FL: monitor's call, no autonomous fallback
        if mode == "gossip" and full_since is not None and r - full_since >= T["dwell"] + T["cmd_latency"]:
            mode, full_since, seq = "federated", None, seq + 1
            evs.append(dict(base, seq=seq, type="architecture_change", mode="federated",
                            from_mode="gossip", to_mode="federated", reason="recovery",
                            decided_by="monitor", failed_nodes=[],
                            active_nodes=sorted(active), n_active=len(active),
                            detail=f"monitor-commanded; evidence = full membership for {T['dwell']} rounds"))

        pool = sorted(active) if mode == "federated" else sorted(retained)
        sub = fired[pool, lo:hi]
        votes = sub.sum(axis=0)
        fl = votes >= K
        if not fl.any():
            continue
        det = [pool[i] for i in np.where(sub[:, fl].any(axis=1))[0]]
        by = Counter()
        for s in det:
            by[cv[spec_attack[s]]] += int(sub[pool.index(s)][fl].sum())
        atk = by.most_common(1)[0][0]
        # Attribution is about the EMITTER of the traffic, not about IDS membership: an
        # isolated node keeps publishing GOOSE/SV, because removing it from the
        # federation does not disconnect it from the network. Conditioning source_node on
        # `active` (as an earlier version did) erased the attribution of 74 of 80
        # injection alarms and hid exactly the fact the operator needs to see — that
        # IDS-level isolation did not stop the attack.
        src = attr.get(atk)
        src_isolated = src is not None and src in isolated
        seq += 1
        benign_win = bool((y[lo:hi] == nc).all())
        evs.append(dict(base, seq=seq, type="intrusion_detected", mode=mode, attack=atk,
                        k_votes=int(votes[fl].max()), n_flags=int(fl.sum()),
                        detector_specialists=det, active_nodes=sorted(active),
                        n_active=len(active), source_node=src, window_samples=int(hi - lo),
                        detail=("benign-only window: this alarm is a FALSE POSITIVE"
                                if benign_win else
                                (f"source node {src} is ALREADY ISOLATED and still emitting — "
                                 "IDS-level isolation does not block traffic; containment is a "
                                 "network action for the operator" if src_isolated else
                                 "notification only; containment is monitor/operator-gated"))))

        # evidence accrues only against nodes not yet isolated — an already-isolated
        # emitter still produces alarms, but there is nothing left for the monitor to
        # isolate; that case is reported in the alarm's detail instead.
        if src is not None and not src_isolated and isol_cfg \
                and int(votes[fl].max()) >= K \
                and int(fl.sum()) >= isol_cfg.get("escalate_flags", 10):
            evidence[src] += 1
            if evidence[src] == isol_cfg.get("isolate_after", 5):
                active.discard(src); isolated.add(src); seq += 1
                evs.append(dict(base, seq=seq, type="node_isolated", mode=mode,
                                reason="intrusion", decided_by="monitor",
                                failed_nodes=[src], source_node=src, attack=atk,
                                k_votes=int(votes[fl].max()), active_nodes=sorted(active),
                                n_active=len(active),
                                detail=(f"monitor-commanded isolation after "
                                        f"{isol_cfg.get('isolate_after', 5)} corroborated alarms")))
                if mode == "federated" and len(isolated) >= isol_cfg.get("command_switch_after", 2):
                    mode, seq = "gossip", seq + 1
                    evs.append(dict(base, seq=seq, type="architecture_change", mode="gossip",
                                    from_mode="federated", to_mode="gossip", reason="intrusion",
                                    decided_by="monitor", failed_nodes=sorted(isolated),
                                    active_nodes=sorted(active), n_active=len(active),
                                    detail="MONITOR-COMMANDED (not fail-fast): intrusion never "
                                           "drives the autonomous transition"))

    out = f"results/events_{sc['name']}.jsonl"
    os.makedirs("results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for e in evs:
            fh.write(json.dumps(e) + "\n")
    types = Counter(e["type"] for e in evs)
    print(f"[scenario] {sc['name']}: {len(evs)} events -> {dict(types)}")
    print(f"[scenario]   -> {out}")
    return out, len(sel), types


if __name__ == "__main__":
    main()
