#!/usr/bin/env python3
"""Derive the command log an event stream IMPLIES, and check the surface covers it.

Every event with decided_by in {monitor, operator} was caused by a command. This script
walks a stream backwards from the events to the commands that must have produced them,
emitting a conformant log (docs/COMMANDS.md) with the envelope each one needs: idempotency
key, authority epoch, expiry, and the `expect` block the monitor would have asserted from
the state it had observed up to that point.

The point is not to pretend the commands were really issued — nothing commands anything
yet (docs/COMMANDS.md 9). The point is falsifiable: if any monitor-decided event cannot be
expressed as one of the three commands, the surface is INCOMPLETE and this script says so
instead of quietly skipping it.

  python scripts/commands_from_stream.py results/events_intrusion.jsonl
Out: results/commands_<stem>.jsonl  (one {command_obj, result} per line)
"""
import argparse
import datetime as dt
import json
import os
import uuid

# A command is issued on the monitor's view, so it is dated BEFORE the event it causes.
# One round of command latency is the scenario's own `timing.cmd_latency` premise.
ISSUE_LEAD = dt.timedelta(seconds=1.0)
EPOCH = 3  # a plausible deployment epoch; what matters is that it never decreases


def command_for(ev, state):
    """The command that must have produced this event, or None if the event is not an action."""
    t = ev["type"]
    if t == "node_isolated":
        return ("isolate_node",
                {"node": ev["source_node"], "reason": ev.get("reason") or "intrusion",
                 "evidence_seq": state["evidence"].get(ev["source_node"], [])[-3:] or None})
    if t == "architecture_change":
        return ("set_mode", {"to": ev["to_mode"], "reason": ev.get("reason") or "scheduled"})
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    a = ap.parse_args()
    evs = [json.loads(l) for l in open(a.stream, encoding="utf-8")]
    stem = os.path.splitext(os.path.basename(a.stream))[0].replace("events_", "")

    # the monitor's view, accumulated as it polls: what it would put in `expect`
    state = {"mode": None, "n_active": None, "seq": 0, "evidence": {}}
    rows, uncovered = [], []
    cmd_seq = 0

    for ev in evs:
        # the monitor has read everything up to this event (contiguous seq)
        if ev["type"] == "intrusion_detected" and ev.get("source_node") is not None:
            state["evidence"].setdefault(ev["source_node"], []).append(ev["seq"])

        who = ev.get("decided_by")
        if who in ("monitor", "operator"):
            made = command_for(ev, state)
            if made is None:
                uncovered.append((ev["seq"], ev["type"]))
                state["seq"] = ev["seq"]
                continue
            verb, params = made
            if params.get("evidence_seq") is None:
                params.pop("evidence_seq", None)
            issued = dt.datetime.fromisoformat(ev["ts"]) - ISSUE_LEAD
            cmd_seq += 1
            cmd = {
                "command_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{stem}/{ev['seq']}")),
                "issued_at": issued.isoformat(timespec="microseconds"),
                "not_after": (issued + dt.timedelta(seconds=30)).isoformat(timespec="microseconds"),
                "authority_epoch": EPOCH,
                "issuer": {"kind": who, "id": "mon-a" if who == "monitor" else "operator"},
                "command": verb,
                "params": params,
                # the view the monitor decided on: the state BEFORE this event, which is
                # exactly what makes 409 expectation_failed meaningful
                "expect": {"mode": state["mode"], "n_active": state["n_active"],
                           "seq": state["seq"]},
            }
            warn = []
            if verb == "isolate_node":
                warn.append(f"node {params['node']} remains able to emit traffic: "
                            "IDS-level isolation does not block it")
            res = {
                "command_id": cmd["command_id"],
                "cmd_seq": cmd_seq,
                "received_at": ev["ts"],
                "outcome": "applied",
                "replayed": False,
                "effective_round": ev["round"],
                "event_seq": ev["seq"],
                "warnings": warn,
                "state": {"mode": ev.get("mode"), "round": ev["round"],
                          "n_active": ev.get("n_active"), "authority_epoch": EPOCH},
            }
            rows.append({"command_obj": cmd, "result": res})

        # update the view AFTER handling, so `expect` holds the pre-event state
        if ev.get("mode"):
            state["mode"] = ev["mode"]
        if ev.get("n_active") is not None:
            state["n_active"] = ev["n_active"]
        state["seq"] = ev["seq"]

    os.makedirs("results", exist_ok=True)
    out = f"results/commands_{stem}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    n_auto = sum(1 for e in evs if e.get("decided_by") == "autonomous")
    verbs = {}
    for r in rows:
        v = r["command_obj"]["command"]
        verbs[v] = verbs.get(v, 0) + 1
    print(f"[commands] {stem}: {len(evs)} events -> {len(rows)} commands {verbs}")
    print(f"[commands]   autonomous actions (no command, by design): {n_auto}")
    print(f"[commands]   -> {out}")
    if uncovered:
        print(f"[commands] SURFACE INCOMPLETE: {len(uncovered)} monitor-decided event(s) "
              f"map to no command: {uncovered}")
        return 1
    print("[commands]   every monitor-decided event maps to a command: surface SUFFICIENT "
          "for this stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
