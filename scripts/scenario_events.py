#!/usr/bin/env python3
"""Generate a monitor event stream from a declarative SCENARIO file.

Supersedes scripts/two_driver_events.py, whose schedule was hard-coded — which is how an
earlier run silently lost the tail of the recovery arc: the last returns were scheduled
at rounds 760/820 while the benign-only stream only has ~724 windows, so they never
happened and the GL->FL return never fired. Here the schedule is data
(conf/scenarios/*.yaml, validated against schemas/scenario.schema.json AND against the
number of windows the selected stream actually has), so that failure mode is impossible.

Authority rule enforced (docs/decentralized_monitoring.md §1.1). Two roles only: the
Disaster-FD monitor processes (one per node and per server) and the ReSIDS agent.
  * FL -> GL on node inactivity is commanded by the NODE-LOCAL Disaster-FD monitor, from
    local evidence only (fail-fast, no regional quorum): decided_by=monitor;
  * further losses in GL, intrusion-driven isolation and the GL->FL return are decided by
    the FEDERATED Disaster-FD: decided_by=monitor;
  * the agent acts alone ONLY through its watchdog, when its own node-local Disaster-FD
    process is unavailable for D rounds (scenario local_fd_available: false):
    decided_by=autonomous, reason=autonomous_fallback. Fail-safe direction only.

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
    sc["timing"].setdefault("watchdog", 3)
    sc.setdefault("local_fd_available", True)
    # Two cadences, both supplied by Disaster-FD. `window_seconds` is the deprecated
    # single-cadence form and maps onto the local time-based tick.
    cad = sc.setdefault("cadence", {})
    loc = cad.setdefault("local", {})
    if not loc:
        loc["period_s"] = float(sc.get("window_seconds", 1.0))
    if "samples" in loc:
        loc.setdefault("nominal_period_s", 1.0)
    cad.setdefault("federated", {}).setdefault("every", 1)
    sc["window_seconds"] = loc.get("period_s")      # None for a count-based tick
    sc.setdefault("traffic_time", "frame")
    prof = sc.setdefault("profile", {})
    prof.setdefault("name", "iec61850-goose-sv")
    prof.setdefault("label_set", "ereno-7")
    prof.setdefault("attribution", "protocol_source")
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
    # Training + scoring is shared with every other analysis through the cache
    # (scripts/scored_cache.py -> instance_streams.train_and_score, the same block that
    # used to be inlined here), so the streams and the analyses score one identical model.
    from scored_cache import get_scored
    ts, fired, y, nc, cv, spec_attack, cache = get_scored(paths[0])
    print(f"[scenario] scored test from {cache}")
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
                    "cadence": sc["cadence"], "traffic_time": sc["traffic_time"],
                    "profile": sc["profile"],
                    "windows": nwin, "sim_epoch": sc["sim_epoch"],
                    "timing": sc["timing"]},
            output={**file_ref(out), "events": sum(types.values()), "types": dict(types)})
        print(f"[scenario]   manifest -> results/manifest_{sc['name']}.json "
              f"(output sha256 {man['output']['sha256'][:12]}…)")


def local_ticks(sc, ts, t0):
    """Partition the scored samples by the NODE-LOCAL Disaster-FD tick.

    The agent infers no time from the traffic: a window is what it scored between two
    ticks. Time-based ticks (period_s) bucket by arrival time; count-based ticks (samples)
    bucket by arrival order, for data that carries no time (the tick is then simulated).
    Returns (bounds, number of ticks, period used for the tick instants)."""
    loc = sc["cadence"]["local"]
    if "samples" in loc:
        k, n = int(loc["samples"]), len(ts)
        n_w = (n + k - 1) // k
        return np.minimum(np.arange(n_w + 1) * k, n), n_w, float(loc.get("nominal_period_s", 1.0))
    period = float(loc["period_s"])
    win = np.floor((ts - t0) / period).astype(np.int64)
    n_w = int(win[-1]) + 1
    return np.searchsorted(win, np.arange(n_w + 1)), n_w, period


def gen(sc, ts, fired, y, nc, cv, spec_attack, t0):
    N, K, T = sc["nodes"], sc["fusion_k"], sc["timing"]
    local_fd = bool(sc.get("local_fd_available", True))
    every = int(sc["cadence"]["federated"]["every"])      # federated tick every N local rounds
    frame_time = sc.get("traffic_time", "frame") == "frame"
    bnd, n_w, period = local_ticks(sc, ts, t0)
    sel = []
    for w in range(n_w):
        lo, hi = bnd[w], bnd[w + 1]
        if hi == lo:
            continue
        benign = bool((y[lo:hi] == nc).all())
        if sc["stream"] == "all" or (sc["stream"] == "benign_only") == benign:
            sel.append((w, lo, hi))        # keep the ORIGINAL window index for the clock
    print(f"[scenario] {sc['name']}: stream '{sc['stream']}' -> {len(sel)} windows "
          f"(local tick {sc['cadence']['local']}, federated every {every})")
    validate_schedule(sc, len(sel))

    fail = {s["round"]: s["node"] for s in sc["schedule"] if s["event"] == "node_failure"}
    back = {s["round"]: s["node"] for s in sc["schedule"] if s["event"] == "node_recovery"}
    # attribution exists only if the profile has a source for it
    attr = ({} if sc["profile"]["attribution"] == "none"
            else {k: v for k, v in (sc.get("attribution") or {}).items()})
    isol_cfg = sc.get("isolation") or {}

    evs, seq, active, retained = [], 0, set(range(N)), set(range(N))
    mode, pend, full_since, evidence, isolated = "federated", None, None, Counter(), set()
    pending_iso = {}          # src -> (attack, k_votes) awaiting the next federated tick
    # `r` is the scenario round (consecutive, so the schedule indexes it); `w` is the
    # window's position in the ORIGINAL trace, and the timestamps must come from w — the
    # selected windows are NOT contiguous in time (the two streams are disjoint views of
    # one trace, interleaved in it), so deriving the clock from r would fabricate a
    # timeline and break correlation with SOE/disturbance records.
    for r, (w, lo, hi) in enumerate(sel):
        # window bounds = the local Disaster-FD tick instants (never inferred from traffic);
        # the frame times, when the profile has them, ride along as metadata
        ws, we = t0 + w * period, t0 + (w + 1) * period
        fed_tick = r % every == 0
        base = {"ts": iso(we), "round": r, "fed_round": r // every,
                "window_start": iso(ws), "window_end": iso(we)}
        if frame_time:
            base["traffic_time_start"] = iso(ts[lo])
            base["traffic_time_end"] = iso(ts[hi - 1])

        if r in fail:
            active.discard(fail[r]); seq += 1
            evs.append(dict(base, seq=seq, type="node_failure", mode=mode,
                            reason="node_failure",
                            failed_nodes=[fail[r]], active_nodes=sorted(active),
                            n_active=len(active)))
            if mode == "federated" and pend is None:
                # normal: the node-local Disaster-FD monitor commands FL->GL after
                # detect_lag; if that local process is down, the agent's watchdog waits
                # D more rounds before acting alone
                pend = r + T["detect_lag"] + (0 if local_fd else T["watchdog"])
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

        # FL->GL on inactivity: commanded by the NODE-LOCAL Disaster-FD monitor from local
        # evidence only (fail-fast, no regional quorum). The agent acts alone only through
        # its watchdog, when that local FD process itself is unavailable.
        if pend is not None and r >= pend and mode == "federated":
            mode, pend, seq = "gossip", None, seq + 1
            evs.append(dict(base, seq=seq, type="architecture_change", mode="gossip",
                            from_mode="federated", to_mode="gossip",
                            reason="node_failure" if local_fd else "autonomous_fallback",
                            decided_by="monitor" if local_fd else "autonomous",
                            failed_nodes=sorted(set(range(N)) - active),
                            active_nodes=sorted(active), n_active=len(active),
                            detail=("fail-fast FL->GL commanded by the node-local Disaster-FD "
                                    "monitor on local evidence (no regional quorum)"
                                    if local_fd else
                                    f"agent watchdog: node-local Disaster-FD monitor unresponsive "
                                    f"for D={T['watchdog']} rounds; fail-safe direction only")))
        # GL->FL: the federated Disaster-FD's call, with NO fallback of any kind. Taken only
        # on a federated tick; dwell and cmd_latency are counted in FEDERATED rounds.
        if mode == "gossip" and full_since is not None and fed_tick \
                and r - full_since >= (T["dwell"] + T["cmd_latency"]) * every:
            mode, full_since, seq = "federated", None, seq + 1
            evs.append(dict(base, seq=seq, type="architecture_change", mode="federated",
                            from_mode="gossip", to_mode="federated", reason="recovery",
                            decided_by="monitor", failed_nodes=[],
                            active_nodes=sorted(active), n_active=len(active),
                            detail=f"commanded by the federated Disaster-FD; evidence = full "
                                   f"membership for {T['dwell']} rounds"))

        pool = sorted(active) if mode == "federated" else sorted(retained)
        sub = fired[pool, lo:hi]
        votes = sub.sum(axis=0)
        fl = votes >= K
        if fl.any():
            seq, mode = alarm(sc, base, r, lo, hi, pool, sub, votes, fl, y, nc, cv, spec_attack,
                              attr, isol_cfg, K, evs, seq, mode, active, isolated, evidence,
                              pending_iso)
        # isolations decided by the federated Disaster-FD are taken on its tick only
        if pending_iso and fed_tick:
            seq, mode = take_isolations(base, isol_cfg, evs, seq, mode, active, isolated,
                                        pending_iso)

    out = f"results/events_{sc['name']}.jsonl"
    os.makedirs("results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for e in evs:
            fh.write(json.dumps(e) + "\n")
    types = Counter(e["type"] for e in evs)
    print(f"[scenario] {sc['name']}: {len(evs)} events -> {dict(types)}")
    print(f"[scenario]   -> {out}")
    return out, len(sel), types


def alarm(sc, base, r, lo, hi, pool, sub, votes, fl, y, nc, cv, spec_attack, attr, isol_cfg,
          K, evs, seq, mode, active, isolated, evidence, pending_iso):
    """One intrusion_detected for the window; accrue evidence against its source.

    The isolation itself is NOT taken here: it is a federated Disaster-FD decision, queued
    in pending_iso and executed on the next federated tick (take_isolations)."""
    det = [pool[i] for i in np.where(sub[:, fl].any(axis=1))[0]]
    by = Counter()
    for s in det:
        by[cv[spec_attack[s]]] += int(sub[pool.index(s)][fl].sum())
    atk = by.most_common(1)[0][0]
    # Attribution is about the EMITTER of the traffic, not about IDS membership: an
    # isolated node keeps publishing GOOSE/SV, because removing it from the federation
    # does not disconnect it from the network. Conditioning source_node on `active` (as an
    # earlier version did) erased the attribution of 74 of 80 injection alarms and hid
    # exactly the fact that needs to be seen — that IDS-level isolation did not stop the
    # attack.
    src = attr.get(atk)
    src_isolated = src is not None and src in isolated
    n_flags, n_win = int(fl.sum()), int(hi - lo)
    seq += 1
    benign_win = bool((y[lo:hi] == nc).all())
    evs.append(dict(base, seq=seq, type="intrusion_detected", mode=mode, attack=atk,
                    k_votes=int(votes[fl].max()), n_flags=n_flags,
                    detector_specialists=det, active_nodes=sorted(active),
                    n_active=len(active), source_node=src, window_samples=n_win,
                    detail=("benign-only window: this alarm is a FALSE POSITIVE"
                            if benign_win else
                            (f"source node {src} is ALREADY ISOLATED and still emitting — "
                             "IDS-level isolation does not block traffic; containment is a "
                             "network action outside the IDS" if src_isolated else
                             "notification only; isolation is decided by the federated "
                             "Disaster-FD"))))

    # Evidence accrues only against nodes not yet isolated. The escalation test is the
    # flagged FRACTION when configured and the volume (n_flags) otherwise; both are
    # calibrated at one cadence (scripts/cadence_sweep.py: safe when the tick stretches,
    # not when it shortens).
    if "escalate_fraction" in isol_cfg:
        escalates = n_flags / n_win >= isol_cfg["escalate_fraction"]
    else:
        escalates = n_flags >= isol_cfg.get("escalate_flags", 10)
    if src is not None and not src_isolated and src not in pending_iso and isol_cfg \
            and int(votes[fl].max()) >= K and escalates:
        evidence[src] += 1
        if evidence[src] >= isol_cfg.get("isolate_after", 5):
            pending_iso[src] = (atk, int(votes[fl].max()))
    return seq, mode


def take_isolations(base, isol_cfg, evs, seq, mode, active, isolated, pending_iso):
    """Execute the isolations queued for this federated tick (and the FL->GL they may
    lead to). With federated.every = 1 this happens in the same round as the alarm that
    completed the evidence, exactly as before the two cadences existed."""
    for src, (atk, kv) in list(pending_iso.items()):
        del pending_iso[src]
        active.discard(src); isolated.add(src); seq += 1
        evs.append(dict(base, seq=seq, type="node_isolated", mode=mode,
                        reason="intrusion", decided_by="monitor",
                        failed_nodes=[src], source_node=src, attack=atk,
                        k_votes=kv, active_nodes=sorted(active), n_active=len(active),
                        detail=(f"isolation commanded by the federated Disaster-FD after "
                                f"{isol_cfg.get('isolate_after', 5)} corroborated alarms")))
        if mode == "federated" and len(isolated) >= isol_cfg.get("command_switch_after", 2):
            mode, seq = "gossip", seq + 1
            evs.append(dict(base, seq=seq, type="architecture_change", mode="gossip",
                            from_mode="federated", to_mode="gossip", reason="intrusion",
                            decided_by="monitor", failed_nodes=sorted(isolated),
                            active_nodes=sorted(active), n_active=len(active),
                            detail="commanded by the federated Disaster-FD (intrusion, "
                                   "not inactivity: never a fail-fast transition)"))
    return seq, mode


if __name__ == "__main__":
    main()
