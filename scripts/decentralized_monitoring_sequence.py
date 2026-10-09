#!/usr/bin/env python3
"""Sequence diagram: decentralized observability (RD + Observe + since=), Disaster-FD roles.
API 1.9.0 (docs/TRUST_LEVEL.md): the colocated Disaster-FD sends the trust level, the agent
decides each switch (decided_by: agent); a stale TL sends it to local. Intrusion events,
isolation and the command surface are future work. Earlier versions below.

Architecture of 2026-10-02 (docs/decentralized_monitoring.md 1.1): two roles only,
Disaster-FD and the ReSIDS agent. Four lifelines: the agent (one per monitored node), its
node-local Disaster-FD monitor, the CoAP Resource Directory (RFC 9176), and the federated
Disaster-FD that reconciles all agents. Five phases: discovery, subscription, steady state,
gap recovery by contiguous seq, and partition / local-FD failure.
1.6.0: Disaster-FD is the time reference, in two cadences -- `round` (local tick) and
`fed_round` (federated tick); an event's window is what the agent scored between two local
ticks; federated decisions (isolation, GL->FL) land on a federated tick.
1.7.0: the domain profile (label set, attribution, traffic time) is reported in /status;
`attack` comes from its label set and `source_node` from its attribution (null with none).

Solid arrow = request / notification, dashed = response or ACK, red X = lost message.
Out: results/decentralized_monitoring_sequence.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

X = {"AG": 1.45, "LF": 4.45, "RD": 7.6, "FF": 11.1}
STY = {"AG": dict(fc="#e8eef7", ec="#3b5b92"),
       "LF": dict(fc="#e6f4ea", ec="#3a7d4f"),
       "RD": dict(fc="#f2f2f2", ec="#6b6b6b"),
       "FF": dict(fc="#f3ecf7", ec="#6a3d9a")}
LABEL = {"AG": "ReSIDS agent\n(one per monitored node)",
         "LF": "Disaster-FD\n(node-local)",
         "RD": "Resource Directory\n(RFC 9176)",
         "FF": "Disaster-FD\n(federated)"}
RED, GREEN, PURPLE = "#b2182b", "#3a7d4f", "#6a3d9a"

rows = [
    ("banner", "API 1.9.0  ·  DTLS 1.3 / coaps:// 5684  ·  +~11 B per record  ·  Observe (RFC 7641)  ·  RD (RFC 9176)  ·  TLS path: RFC 8323 (per profile)"),
    ("note", "TRANSPORT IS AN OPEN DECISION — this is the CoAP binding; REST/TLS is the alternative for the Disaster-FD path"),
    ("note", "time reference from Disaster-FD:  round = index of its 5-s probe cycle  ·  the trust level arrives on the host-local channel, never over the network"),
    ("phase", "1 · Discovery  —  rare (only on provisioning or re-addressing)"),
    ("msg", "AG", "RD", "POST /rd?ep=node-07&lt=3600&base=coaps://10.0.1.17:5684", "req"),
    ("msg", "RD", "AG", "2.01 Created", "ack"),
    ("msg", "FF", "RD", "GET /rd-lookup/ep?ep=node-07", "req"),
    ("msg", "RD", "FF", "base = coaps://10.0.1.17:5684   (cached)", "ack"),
    ("phase", "2 · Subscription  —  once"),
    ("msg", "FF", "AG", "GET /events   (Observe: 0)", "req"),
    ("msg", "AG", "FF", "2.05 Content  ·  Observe registered", "ack"),
    ("phase", "3 · Steady state  —  trust level from Disaster-FD, decisions by the agent"),
    ("msg", "FF", "AG", "GET /status   —  once, at start-up: state bootstrap", "req"),
    ("msg", "AG", "FF", "2.05 {api_version, current_mode, round, last_seq, profile{…}, trust{trust_level, band, candidate, stale}}", "ack"),
    ("note", "the event types share this one channel — each CON is ACKed; the ACKs are omitted below"),
    ("cmd", "LF", "AG", "trust_level {target: aggregator, trusted: false, TL 30}   (one per probe)", GREEN),
    ("msg", "AG", "FF", "notify · node_failure {failed_nodes:[0]}        — relayed observation, no decided_by", "req"),
    ("self", "AG", "band gossip held S = 5 s  →  commit FL→GL", "#3b5b92"),
    ("msg", "AG", "FF", "notify · architecture_change {FL→GL, reason: trust_level, trust_level: 30}   [decided_by: agent]", "req"),
    ("cmd", "LF", "AG", "trust_level {target: aggregator, trusted: true, TL 80}", GREEN),
    ("self", "AG", "band federated held S = 5 s  →  commit GL→FL", "#3b5b92"),
    ("msg", "AG", "FF", "notify · architecture_change {GL→FL, reason: trust_level, trust_level: 80}   [decided_by: agent]", "req"),
    ("note", "every transition is decided by the agent from its own monitor's TL — Disaster-FD supplies evidence, never a mode"),
    ("note", "pilot run, 14 clients: 4 switches each, all 14 within 4.9 s of each other  ·  intrusion events and isolation: future work"),
    ("note", "no event in the window → nothing is sent"),
    ("msg", "FF", "AG", "GET /events?since=388   (periodic reconciliation)", "req"),
    ("msg", "AG", "FF", "2.05  []   →  alive  ·  no gap  ·  nothing to recover", "ack"),
    ("phase", "4 · Gap recovery  —  one round trip, on the contiguous seq"),
    ("lost", "AG", "FF", "notify · node_failure {failed_nodes:[3]}", "req"),
    ("msg", "FF", "AG", "GET /events?since=386   (386 = highest CONTIGUOUS seq, not the highest seen)", "req"),
    ("msg", "AG", "FF", "2.05  [387, 388]   →  gap revealed AND filled  (duplicates dropped by seq)", "ack"),
    ("note", "using the highest seq seen would hide interior holes: 33 events lost at 20% loss, 0 with contiguous"),
    ("phase", "5 · Partition and local-FD failure  —  neither stops the fail-fast"),
    ("lost", "FF", "AG", "GET /events?since=…   (times out)", "req"),
    ("msg", "FF", "RD", "GET /rd-lookup/ep   —  address changed, or really gone?", "req"),
    ("msg", "RD", "FF", "registration still valid  →  partition, not a dead node", "ack"),
    ("self", "LF", "co-located with the agent: on the same side of every partition → its TL drops → the agent goes local", GREEN),
    ("self", "LF", "✗  if the colocated FD process itself crashes or hangs …", RED),
    ("self", "AG", "… no TL for F = 15 s → the agent goes local  [reason: tl_stale, decided_by: agent]", RED),
    ("note", "replaces the watchdog D of 1.5.0 · in the pilot run the largest gap between observations is 4.8 s, so it never fires"),
]

H = {"msg": 0.56, "lost": 0.56, "cmd": 0.56, "note": 0.46, "banner": 0.50, "phase": 0.52,
     "self": 0.6}
y, laid = 0.95, []
for kind, *rest in rows:
    y -= H[kind]
    laid.append((kind, y, rest))
    y -= 0.10
bottom = y - 0.25

fig, ax = plt.subplots(figsize=(14.2, 1.3 - bottom))
ax.set_xlim(0, 12.7)
ax.set_ylim(bottom, 1.95)
ax.axis("off")

for key in ("AG", "LF", "RD", "FF"):
    st, x = STY[key], X[key]
    ax.add_patch(FancyBboxPatch((x - 1.35, 1.18), 2.7, 0.6,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=st["fc"], ec=st["ec"], lw=1.7))
    ax.text(x, 1.48, LABEL[key], ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=st["ec"])
    ax.plot([x, x], [1.18, bottom + 0.12], ls=(0, (4, 4)), color="0.6", lw=1)


def arrow(y, x0, x1, text, kind, lost=False, color=None):
    xe = x0 + (x1 - x0) * (0.55 if lost else 1.0)
    solid = kind in ("req", "cmd")
    col = RED if lost else (color or ("0.1" if solid else "0.45"))
    ax.annotate("", xy=(xe, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.9 if solid else 1.2,
                                ls="-" if solid else (0, (5, 3))))
    if lost:
        ax.plot(xe, y, marker="x", ms=13, mew=3, color=RED)
    ax.text((x0 + xe) / 2, y + 0.11, text, ha="center", va="bottom",
            fontsize=10.3 if solid else 9.8,
            color=(color or "0.15") if solid else "0.4",
            style="normal" if solid else "italic",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


for kind, y0, rest in laid:
    if kind == "banner":
        ax.add_patch(FancyBboxPatch((0.4, y0 - 0.02), 11.9, 0.36,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc="#f2f2f2", ec="0.7", lw=0.9))
        ax.text(6.35, y0 + 0.16, rest[0], ha="center", va="center", fontsize=10.3, color="0.3")
    elif kind == "phase":
        ax.add_patch(FancyBboxPatch((0.15, y0 - 0.02), 12.4, 0.36,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc="#eef3fa", ec="#9bb4d4", lw=1.0))
        ax.text(0.35, y0 + 0.16, rest[0], ha="left", va="center", fontsize=11,
                fontweight="bold", color="#2c4d7c")
    elif kind == "note":
        ax.text(6.35, y0 + 0.14, rest[0], ha="center", va="center", fontsize=10,
                color="0.45", style="italic")
    elif kind == "self":
        who, text, col = rest
        x = X[who]
        width = min(12.55 - (x - 0.45), 0.072 * len(text) + 0.5)
        ax.add_patch(FancyBboxPatch((x - 0.45, y0 - 0.04), width, 0.44,
                     boxstyle="round,pad=0.10,rounding_size=0.08",
                     fc="#fdeceb" if col == RED else "#eef7f0",
                     ec="#e6a3a0" if col == RED else "#9fcdb0", lw=1.1))
        ax.text(x - 0.30, y0 + 0.18, text, ha="left", va="center", fontsize=10.3, color=col)
    elif kind == "cmd":
        frm, to, text, col = rest
        arrow(y0, X[frm], X[to], text, "cmd", color=col)
    else:
        frm, to, text, k = rest
        arrow(y0, X[frm], X[to], text, k, lost=(kind == "lost"))

ax.set_title("Decentralized observability with Disaster-FD (API 1.9.0): Resource Directory + Observe + since=",
             fontsize=15, fontweight="bold")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/decentralized_monitoring_sequence.png", dpi=170, bbox_inches="tight")
fig.savefig("results/decentralized_monitoring_sequence.pdf", bbox_inches="tight")
print("ok -> results/decentralized_monitoring_sequence.{png,pdf}")
