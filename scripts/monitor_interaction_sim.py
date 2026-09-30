#!/usr/bin/env python3
"""Replay harness: ReSIDS <-> external monitor interaction, in process.

Feeds the event timeline produced by adaptive_events_demo.py through the hybrid
transport we settled on -- CoAP Observe (node -> monitor, push on event) plus a
periodic ping on GET /status (monitor -> node, liveness + last_seq) -- and exercises
the four behaviours that actually matter. No sockets: the point is the LOGIC, which
is where the design decisions live. The wire format (aiocoap) would be a later step.

Scenarios
  1 clean          every notification is delivered
  2 lossy link     a fraction of notifications is dropped -> the monitor must notice
                   the gap by comparing status.last_seq and recover with /events?since=
  3 management partition  monitor unreachable around the failure -> the node asks for a
                   command, gets none, and falls back to autonomous fail-fast after D
  4 monitor-commanded  monitor reachable -> it answers, and the switch happens on command

Out: results/monitor_interaction_sim.txt (+ printed report)
"""
import json
import os
import random
import sys
from collections import defaultdict

IN = "results/adaptive_events_demo.jsonl"
OUT = "results/monitor_interaction_sim.txt"
PING_EVERY = 10      # rounds between monitor pings (GET /status)
DEADLINE_D = 5       # rounds the node waits for a command before autonomous fallback
CMD_LATENCY = 3      # rounds the monitor takes to answer
FAILFAST = 2         # autonomous fail-fast latency (rounds), as in the v2 manager
ESCALATE_FLAGS = 10  # n_flags threshold for the monitor to treat an event as an incident


class Node:
    """The ReSIDS side: append-only event log, /status and /events?since=."""

    def __init__(self, events):
        self.all = events
        self.log = []                      # events already emitted (the JSONL trail)
        self.mode = "federated"
        self.round = 0

    def emit(self, ev):
        self.log.append(ev)

    @property
    def last_seq(self):
        return self.log[-1]["seq"] if self.log else 0

    def status(self):                      # GET /status
        return {"current_mode": self.mode, "round": self.round,
                "last_seq": self.last_seq}

    def events_since(self, seq):           # GET /events?since=
        return [e for e in self.log if e["seq"] > seq]


class Monitor:
    """The external monitor: Observe subscriber + periodic ping + triage."""

    def __init__(self):
        self.seen = set()
        self.max_seq = 0
        self.gaps_found = 0
        self.recovered = 0
        self.incidents = 0
        self.blind_rounds = 0
        self.pings_ok = 0
        self.pings_failed = 0

    def on_notify(self, ev):               # CoAP Observe notification
        self.seen.add(ev["seq"])
        self.max_seq = max(self.max_seq, ev["seq"])
        if ev["type"] == "intrusion_detected" and ev.get("n_flags", 0) >= ESCALATE_FLAGS:
            self.incidents += 1

    @property
    def contiguous(self):
        """Highest seq such that everything up to it has been seen. Comparing only
        last_seq would miss INTERIOR holes: once later notifications arrive, the tail
        is up to date while earlier events are still missing."""
        c = 0
        while c + 1 in self.seen:
            c += 1
        return c

    def ping(self, node, link_up):         # GET /status
        if not link_up:
            self.pings_failed += 1
            self.blind_rounds += PING_EVERY
            return
        self.pings_ok += 1
        st = node.status()
        if st["last_seq"] > self.contiguous:   # a hole anywhere, not just at the tail
            self.gaps_found += 1
            for ev in node.events_since(self.contiguous):
                if ev["seq"] not in self.seen:
                    self.recovered += 1
                    self.on_notify(ev)


def run(name, events, loss=0.0, partition=(), commanded=False, seed=7):
    rng = random.Random(seed)
    node, mon = Node(events), Monitor()
    by_round = defaultdict(list)
    for e in events:
        if e["type"] != "architecture_change":      # the switch is re-decided here
            by_round[e["round"]].append(e)
    trigger = next(e["round"] for e in events if e["type"] == "node_failure")
    max_round = max(e["round"] for e in events)

    switch_round = None
    switch_kind = None
    asked_at = None
    delivered = lost = 0

    for r in range(max_round + 1):
        node.round = r
        up = not any(a <= r <= b for a, b in partition)

        for ev in by_round.get(r, []):
            node.emit(ev)
            if up and rng.random() >= loss:
                mon.on_notify(ev)
                delivered += 1
            else:
                lost += 1

        # -- switch decision at the failure trigger --
        if r == trigger:
            asked_at = r if commanded else None
        if switch_round is None and asked_at is not None:
            if up and r - asked_at >= CMD_LATENCY:
                switch_round, switch_kind = r, "commanded"
            elif r - asked_at >= DEADLINE_D:
                switch_round, switch_kind = r, "deadline fallback (autonomous)"
        if switch_round is None and not commanded and r - trigger >= FAILFAST and r >= trigger:
            switch_round, switch_kind = r, "autonomous fail-fast"
        if switch_round == r:
            node.mode = "gossip"
            ev = {"seq": node.last_seq + 1, "type": "architecture_change", "round": r,
                  "from_mode": "federated", "to_mode": "gossip",
                  "reason": "node_failure", "decided_by": switch_kind}
            node.emit(ev)
            if up:
                mon.on_notify(ev); delivered += 1
            else:
                lost += 1

        if r % PING_EVERY == 0:
            mon.ping(node, up)

    total = len(node.log)
    return {"scenario": name, "emitted": total, "delivered": delivered, "lost": lost,
            "seen": len(mon.seen), "gaps_found": mon.gaps_found, "recovered": mon.recovered,
            "missing": total - len(mon.seen), "incidents": mon.incidents,
            "pings_ok": mon.pings_ok, "pings_failed": mon.pings_failed,
            "switch_round": switch_round, "switch_kind": switch_kind,
            "switch_latency": switch_round - trigger if switch_round else None}


def main():
    if not os.path.exists(IN):
        sys.exit(f"missing {IN} - run scripts/adaptive_events_demo.py first")
    events = [json.loads(l) for l in open(IN, encoding="utf-8")]
    trigger = next(e["round"] for e in events if e["type"] == "node_failure")
    runs = [
        run("1 clean", events),
        run("2 lossy link (20%)", events, loss=0.20),
        run("3 mgmt partition", events, partition=[(trigger - 5, trigger + 40)], commanded=True),
        run("4 monitor-commanded", events, commanded=True),
    ]
    lines = []
    lines.append("failure trigger at round %d | ping every %d rounds | D=%d | cmd latency=%d"
                 % (trigger, PING_EVERY, DEADLINE_D, CMD_LATENCY))
    lines.append("")
    lines.append("%-22s %8s %9s %6s %7s %9s %8s %8s" %
                 ("scenario", "emitted", "delivered", "lost", "gaps", "recovered", "missing", "incid."))
    for r in runs:
        lines.append("%-22s %8d %9d %6d %7d %9d %8d %8d" %
                     (r["scenario"], r["emitted"], r["delivered"], r["lost"],
                      r["gaps_found"], r["recovered"], r["missing"], r["incidents"]))
    lines.append("")
    lines.append("%-22s %12s %8s   %s" % ("scenario", "switch@round", "latency", "decided by"))
    for r in runs:
        lines.append("%-22s %12s %8s   %s" % (r["scenario"], r["switch_round"],
                                              r["switch_latency"], r["switch_kind"]))
    lines.append("")
    for r in runs:
        lines.append("%-22s pings ok=%d failed=%d" % (r["scenario"], r["pings_ok"], r["pings_failed"]))
    txt = "\n".join(lines)
    print(txt)
    os.makedirs("results", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(txt + "\n")
    print("\n->", OUT)


if __name__ == "__main__":
    main()
