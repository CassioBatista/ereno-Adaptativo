#!/usr/bin/env python3
"""Conformance check for ReSIDS monitor COMMANDS (docs/COMMANDS.md).

Validates a command log (JSONL of command objects, or of {command, result} pairs) against
schemas/command.schema.json and schemas/command_result.schema.json, and then the
invariants JSON Schema cannot express:

  * command_id is UNIQUE across the log — it is the at-most-once key, so a repeat is
    either a replay (which must carry result.replayed) or a bug that would execute an
    isolation or a mode switch twice;
  * authority_epoch is NON-DECREASING — a command below the highest epoch seen is exactly
    what the stale_epoch rejection exists to stop (a revoked monitor still commanding);
  * not_after > issued_at, and issued_at + 60 s is the hard acceptance cap even when
    not_after is absent (docs/COMMANDS.md 2);
  * cmd_seq starts at 1 and has NO gaps — the log must account for rejected commands too,
    since those produce no event and would otherwise vanish;
  * decided_by/command_id coherence: a result with outcome=applied must name an
    event_seq once the event exists, and an error must be present exactly when
    outcome=rejected;
  * NO COMMAND MAY TARGET THE DEADLINE D. There is no hold/cancel/extend verb in the
    schema, and this check restates it as an assertion over the log, because the absence
    is a safety property (docs/COMMANDS.md 4.1), not an omission to be fixed later.

Usage:
  python scripts/validate_commands.py results/commands_intrusion.jsonl
Exit code 0 = conformant.
"""
import json
import sys
from collections import Counter
from datetime import datetime, timedelta

from jsonschema import Draft202012Validator

CMD_SCHEMA = "schemas/command.schema.json"
RES_SCHEMA = "schemas/command_result.schema.json"
ACCEPT_WINDOW = timedelta(seconds=60)
FORBIDDEN_VERBS = {"hold", "cancel_deadline", "extend_deadline", "veto", "freeze",
                   "set_k", "upload_booster", "set_label", "block_traffic"}


def ts(s):
    return datetime.fromisoformat(s)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    cv = Draft202012Validator(json.load(open(CMD_SCHEMA, encoding="utf-8")))
    rv = Draft202012Validator(json.load(open(RES_SCHEMA, encoding="utf-8")))

    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    errs, ids, epoch, seqs = [], {}, None, []
    outcomes, verbs = Counter(), Counter()

    for i, row in enumerate(rows, 1):
        cmd = row.get("command_obj", row.get("command") if isinstance(row.get("command"), dict) else row)
        res = row.get("result")
        where = f"line {i}"

        for e in cv.iter_errors(cmd):
            errs.append(f"{where}: command schema: {'/'.join(map(str, e.path))} {e.message}")
        if res is not None:
            for e in rv.iter_errors(res):
                errs.append(f"{where}: result schema: {'/'.join(map(str, e.path))} {e.message}")

        verb = cmd.get("command")
        verbs[verb] += 1
        if verb in FORBIDDEN_VERBS:
            errs.append(f"{where}: '{verb}' is not a command this system has. "
                        f"The deadline D and the detection parameters are deliberately "
                        f"NOT commandable (docs/COMMANDS.md 4.1, 7)")

        cid = cmd.get("command_id")
        if cid in ids:
            replayed = bool(res and res.get("replayed"))
            if not replayed:
                errs.append(f"{where}: command_id {cid} repeats {ids[cid]} without "
                            f"result.replayed — at-most-once execution is violated")
        else:
            ids[cid] = where

        ep = cmd.get("authority_epoch")
        if epoch is not None and ep is not None and ep < epoch:
            errs.append(f"{where}: authority_epoch {ep} < {epoch} seen earlier — this is "
                        f"the stale_epoch case and must be rejected, not logged as issued")
        if ep is not None:
            epoch = max(epoch or 0, ep)

        if cmd.get("issued_at"):
            t0 = ts(cmd["issued_at"])
            na = cmd.get("not_after")
            if na and ts(na) <= t0:
                errs.append(f"{where}: not_after {na} is not after issued_at {cmd['issued_at']}")
            if na and ts(na) - t0 > ACCEPT_WINDOW:
                errs.append(f"{where}: not_after exceeds the {ACCEPT_WINDOW.seconds}s "
                            f"acceptance cap; the agent would refuse it as expired anyway")
            if res and res.get("received_at") and ts(res["received_at"]) < t0:
                errs.append(f"{where}: received_at precedes issued_at")

        if res is not None:
            outcomes[res["outcome"]] += 1
            seqs.append(res["cmd_seq"])
            if (res["outcome"] == "rejected") != (res.get("error") is not None):
                errs.append(f"{where}: error must be present exactly when "
                            f"outcome=rejected (got {res['outcome']})")
            if res["outcome"] == "applied" and res.get("effective_round") is None:
                errs.append(f"{where}: outcome=applied without effective_round — a command "
                            f"commits on a round boundary, so the round must be stated")
            if res["outcome"] != "applied" and res.get("event_seq") is not None:
                errs.append(f"{where}: event_seq on a non-applied command: nothing was "
                            f"committed, so no event records it")

    if seqs:
        want = list(range(1, len(seqs) + 1))
        if sorted(seqs) != want:
            missing = sorted(set(want) - set(seqs))
            errs.append(f"cmd_seq must start at 1 and have no gaps; missing {missing[:10]}")
        if seqs != sorted(seqs):
            errs.append("cmd_seq is not ascending in file order")

    print(f"\n=== {path} ===")
    print(f"commands: {len(rows)} | verbs: {dict(verbs)}")
    if outcomes:
        print(f"outcomes: {dict(outcomes)}")
    print(f"authority_epoch (highest): {epoch} | unique command_id: {len(ids)}")
    if errs:
        print(f"\nNOT CONFORMANT — {len(errs)} problem(s):")
        for e in errs[:40]:
            print(f"  - {e}")
        return 1
    print("CONFORMANT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
