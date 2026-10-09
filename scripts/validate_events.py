#!/usr/bin/env python3
"""Conformance check for ReSIDS monitor artifacts.

Validates an event stream (JSONL) against schemas/event.schema.json, or a scenario file
against schemas/scenario.schema.json, and additionally checks the invariants a consumer
is entitled to rely on but which JSON Schema cannot express:

  * seq is strictly increasing, starts at 1 and has NO gaps (a gap in a file means the
    trail lost events; over the wire it means the monitor must reconcile with
    GET /events?since=<highest CONTIGUOUS seq> - see docs/API.md §2);
  * window_start < window_end, and ts >= window_end (an event is emitted at the close of
    the window it refers to);
  * `decided_by: autonomous` appears ONLY where the authority rule allows it: the agent's
    watchdog FL->GL (reason autonomous_fallback), when its node-local Disaster-FD monitor is
    unavailable. Every other decision is Disaster-FD's (decided_by: monitor)
    (docs/decentralized_monitoring.md §1.1);
  * 1.9.0 streams (decided_by: agent): the agent decides every transition from the
    Disaster-FD trust level (reason trust_level, to_mode = the band of trust_level) or
    falls back to local on a stale TL (reason tl_stale); monitor/autonomous decisions and
    the intrusion events, reserved for future work, do not appear (docs/TRUST_LEVEL.md);
  * per-type required fields (failed_nodes on node_failure, recovered_nodes on
    node_recovery, from/to_mode on architecture_change, source_node on node_isolated).

Usage:
  python scripts/validate_events.py results/events_availability.jsonl
  python scripts/validate_events.py conf/scenarios/availability.yaml
Exit code 0 = conformant.
"""
import json
import sys
from collections import Counter

import yaml
from jsonschema import Draft202012Validator

EVENT_SCHEMA = "schemas/event.schema.json"
SCENARIO_SCHEMA = "schemas/scenario.schema.json"
REQUIRED = {"node_failure": ["failed_nodes"], "node_recovery": ["recovered_nodes"],
            "architecture_change": ["from_mode", "to_mode"],
            "node_isolated": ["source_node"],
            "intrusion_detected": ["attack", "k_votes", "n_flags"]}


def fail(errs, msg):
    errs.append(msg)


def check_scenario(path):
    sc = yaml.safe_load(open(path, encoding="utf-8"))
    v = Draft202012Validator(json.load(open(SCENARIO_SCHEMA, encoding="utf-8")))
    errs = [f"schema: {e.message}" for e in v.iter_errors(sc)]
    up = set(range(sc.get("nodes", 0)))
    for i, s in enumerate(sorted(sc.get("schedule", []), key=lambda x: x["round"])):
        w = f"schedule[{i}] round={s['round']} {s['event']} node={s['node']}"
        if s["node"] >= sc.get("nodes", 0):
            fail(errs, f"{w}: node index out of range")
        if s["event"] == "node_failure":
            if s["node"] not in up:
                fail(errs, f"{w}: node already down")
            up.discard(s["node"])
        else:
            if s["node"] in up:
                fail(errs, f"{w}: node already up")
            up.add(s["node"])
    print(f"scenario '{sc.get('name')}' | stream={sc.get('stream')} "
          f"nodes={sc.get('nodes')} schedule={len(sc.get('schedule', []))}")
    print("NOTE: rounds are validated against the actual stream length by "
          "scripts/scenario_events.py, which knows how many windows the stream has.")
    return errs


def check_events(path):
    evs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    v = Draft202012Validator(json.load(open(EVENT_SCHEMA, encoding="utf-8")))
    errs = []
    for e in evs:
        for err in v.iter_errors(e):
            fail(errs, f"seq {e.get('seq')}: {err.message[:110]}")
    seqs = [e["seq"] for e in evs]
    if seqs != sorted(seqs):
        fail(errs, "seq is not increasing")
    if seqs and seqs[0] != 1:
        fail(errs, f"seq starts at {seqs[0]}, expected 1")
    gaps = [b for a, b in zip(seqs, seqs[1:]) if b != a + 1]
    if gaps:
        fail(errs, f"{len(gaps)} gap(s) in seq, first before seq {gaps[0]}")
    for e in evs:
        for f in REQUIRED.get(e["type"], []):
            if f not in e:
                fail(errs, f"seq {e['seq']}: {e['type']} missing '{f}'")
        if e.get("window_start") and e.get("window_end"):
            if not (e["window_start"] < e["window_end"] <= e["ts"]):
                fail(errs, f"seq {e['seq']}: window/ts ordering violated")
        # traffic time is metadata inside the tick window, never the window itself
        if e.get("traffic_time_start") and e.get("traffic_time_end"):
            if not (e["traffic_time_start"] <= e["traffic_time_end"]):
                fail(errs, f"seq {e['seq']}: traffic_time_start after traffic_time_end")
            if e.get("window_start") and not (
                    e["window_start"] <= e["traffic_time_start"]
                    and e["traffic_time_end"] < e["window_end"]):
                fail(errs, f"seq {e['seq']}: traffic time outside its tick window")
        if e.get("fed_round") is not None and e["fed_round"] > e["round"]:
            fail(errs, f"seq {e['seq']}: fed_round {e['fed_round']} exceeds the local round")
        # decided_by belongs to ACTIONS only, and 'autonomous' to exactly one of them
        is_action = e["type"] in ("architecture_change", "node_isolated")
        if e.get("decided_by") is not None and not is_action:
            fail(errs, f"seq {e['seq']}: {e['type']} is an observation, not an action — "
                       f"it must not carry decided_by")
        if is_action and e.get("decided_by") is None:
            fail(errs, f"seq {e['seq']}: {e['type']} is an action and must state decided_by")
        # Every decision belongs to Disaster-FD (decided_by=monitor). The agent acts alone
        # only through its watchdog -- FL->GL, reason autonomous_fallback -- when its own
        # node-local Disaster-FD process is unavailable. Anything else claiming autonomy,
        # or a fallback in the unsafe direction, is a violation.
        if e.get("decided_by") == "autonomous":
            ok = (e["type"] == "architecture_change" and e.get("to_mode") == "gossip"
                  and e.get("reason") == "autonomous_fallback")
            if not ok:
                fail(errs, f"seq {e['seq']}: decided_by=autonomous not allowed for "
                           f"{e['type']}/{e.get('to_mode')}/{e.get('reason')} — the agent acts "
                           f"alone only as the watchdog FL->GL (reason autonomous_fallback)")
        if e.get("reason") == "autonomous_fallback" and e.get("decided_by") != "autonomous":
            fail(errs, f"seq {e['seq']}: reason autonomous_fallback must be decided_by "
                       f"autonomous (it is the agent's watchdog, not a Disaster-FD decision)")
        # 1.9.0: Disaster-FD supplies the trust level, the agent decides (docs/TRUST_LEVEL.md)
        if e.get("decided_by") == "agent":
            if e["type"] != "architecture_change" or e.get("reason") not in ("trust_level", "tl_stale"):
                fail(errs, f"seq {e['seq']}: decided_by=agent only on architecture_change with "
                           f"reason trust_level or tl_stale")
        if e.get("reason") in ("trust_level", "tl_stale") and e.get("decided_by") != "agent":
            fail(errs, f"seq {e['seq']}: reason {e['reason']} must be decided_by agent")
        if e.get("reason") == "tl_stale" and e.get("to_mode") != "local":
            fail(errs, f"seq {e['seq']}: a stale trust level falls back to local, not {e.get('to_mode')}")
        if e.get("reason") == "trust_level":
            tl, b = e.get("trust_level"), e.get("tl_bands") or {"gl": 30, "fl": 50}
            if tl is None:
                fail(errs, f"seq {e['seq']}: reason trust_level without the trust_level it used")
            else:
                band = "federated" if tl >= b["fl"] else "gossip" if tl >= b["gl"] else "local"
                if band != e.get("to_mode"):
                    fail(errs, f"seq {e['seq']}: TL {tl} selects {band}, event switches to {e.get('to_mode')}")
    if any(e.get("decided_by") == "agent" for e in evs):
        for e in evs:
            if e.get("decided_by") in ("monitor", "autonomous"):
                fail(errs, f"seq {e['seq']}: decided_by={e['decided_by']} in a 1.9.0 stream, where "
                           f"the agent decides every transition")
            if e["type"] in ("intrusion_detected", "node_isolated"):
                fail(errs, f"seq {e['seq']}: {e['type']} is reserved for future work from 1.9.0")
    print(f"events: {len(evs)} | types: {dict(Counter(e['type'] for e in evs))}")
    print(f"seq range: {seqs[0] if seqs else '-'}..{seqs[-1] if seqs else '-'} | "
          f"decided_by: {dict(Counter(e.get('decided_by') for e in evs))}")
    return errs


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bad = 0
    for path in sys.argv[1:]:
        print(f"\n=== {path} ===")
        try:
            errs = check_scenario(path) if path.endswith((".yaml", ".yml")) else check_events(path)
        except FileNotFoundError:
            print("MISSING — generate it first (scripts/scenario_events.py)")
            bad += 1
            continue
        if errs:
            bad += 1
            print(f"FAIL ({len(errs)} problem(s)):")
            for m in errs[:15]:
                print("   ", m)
            if len(errs) > 15:
                print(f"    ... and {len(errs)-15} more")
        else:
            print("CONFORMANT")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
