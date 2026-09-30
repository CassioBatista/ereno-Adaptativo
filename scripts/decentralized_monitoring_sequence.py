#!/usr/bin/env python3
"""Sequence diagram: decentralized observability (RD + Observe + ping).

Three participants: a ReSIDS agent (one per IED), the CoAP Resource Directory
(RFC 9176) and the external Monitor. Covers the five phases of docs/decentralized_
monitoring.md: discovery, subscription, steady state, gap recovery by contiguous seq,
and management partition with deadline fallback.

Solid arrow = request / notification, dashed = response or ACK, red X = lost message.
Out: results/decentralized_monitoring_sequence.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

AG, RD, MO = 1.7, 6.2, 11.0
STY = {"AG": dict(fc="#e8eef7", ec="#3b5b92"),
       "RD": dict(fc="#f3ecf7", ec="#6a3d9a"),
       "MO": dict(fc="#e6f4ea", ec="#3a7d4f")}
X = {"AG": AG, "RD": RD, "MO": MO}

rows = [
    ("banner", "DTLS 1.3 / coaps:// 5684  ·  +~11 B per record (1.2 floor: +29 B)  ·  Observe (RFC 7641)  ·  RD (RFC 9176)  ·  TLS path: RFC 8323 where IEC 62351-3 governs"),
    ("note", "TRANSPORT IS AN OPEN DECISION — this is the CoAP binding; REST/TLS is the alternative for the decision monitor"),
    ("phase", "1 · Discovery  —  rare (only on provisioning or re-addressing)"),
    ("msg", "AG", "RD", "POST /rd?ep=SUB1_PROT_G&lt=3600&base=coaps://10.0.1.17:5684", "req"),
    ("msg", "RD", "AG", "2.01 Created", "ack"),
    ("msg", "MO", "RD", "GET /rd-lookup/ep?ep=SUB1_PROT_G", "req"),
    ("msg", "RD", "MO", "base = coaps://10.0.1.17:5684   (monitor caches it)", "ack"),
    ("phase", "2 · Subscription  —  once"),
    ("msg", "MO", "AG", "GET /events   (Observe: 0)", "req"),
    ("msg", "AG", "MO", "2.05 Content  ·  Observe registered", "ack"),
    ("phase", "3 · Steady state  —  events on occurrence, reconciliation on policy"),
    ("msg", "MO", "AG", "GET /status   —  once, at start-up: state bootstrap", "req"),
    ("msg", "AG", "MO", "2.05 {current_mode, round, active_nodes, last_seq}", "ack"),
    ("msg", "AG", "MO", "notify (CON) · intrusion_detected {attack, k_votes, n_flags, window_start/end}", "req"),
    ("msg", "MO", "AG", "ACK", "ack"),
    ("note", "the five event types share this one channel — each CON is ACKed; the ACKs are omitted below"),
    ("msg", "AG", "MO", "notify · node_failure {failed_nodes:[13]}        — observation, no decided_by", "req"),
    ("msg", "AG", "MO", "notify · architecture_change {FL→GL, reason: node_failure}   [decided_by: AUTONOMOUS]", "req"),
    ("msg", "AG", "MO", "notify · node_recovery {recovered_nodes:[13]}    — observation, no decided_by", "req"),
    ("msg", "AG", "MO", "notify · node_isolated {source_node:9, reason: intrusion}   [decided_by: monitor]", "req"),
    ("msg", "AG", "MO", "notify · architecture_change {GL→FL, reason: recovery}   [decided_by: monitor]", "req"),
    ("note", "only ONE action in the whole system is autonomous: the fail-fast FL→GL on inactivity"),
    ("note", "no event in the window → nothing is sent   (measured: 88.9% of rounds carry no event at all)"),
    ("msg", "MO", "AG", "GET /events?since=388   (periodic reconciliation)", "req"),
    ("msg", "AG", "MO", "2.05  []   →  alive  ·  no gap  ·  nothing to recover", "ack"),
    ("phase", "4 · Gap recovery  —  one round trip, on the contiguous seq"),
    ("lost", "AG", "MO", "notify · node_failure {failed_nodes:[3]}", "req"),
    ("msg", "MO", "AG", "GET /events?since=386   (386 = highest CONTIGUOUS seq, not the highest seen)", "req"),
    ("msg", "AG", "MO", "2.05  [387, 388]   →  gap revealed AND filled  (duplicates dropped by seq)", "ack"),
    ("note", "using the highest seq seen would hide interior holes: 33 events lost at 20% loss, 0 with contiguous"),
    ("phase", "5 · Management partition  —  the deadline fallback removes the SPOF"),
    ("lost", "MO", "AG", "GET /events?since=…   (times out)", "req"),
    ("msg", "MO", "RD", "GET /rd-lookup/ep   —  address changed, or really gone?", "req"),
    ("msg", "RD", "MO", "registration still valid  →  partition, not a dead node", "ack"),
    ("self", "AG", "switch_pending → no command within D rounds → autonomous fail-fast  (FL→GL ONLY; GL→FL has no fallback)"),
    ("note", "the IDS keeps switching without the monitor   (measured: fail-fast 2 rounds · commanded 3 · fallback 5)"),
]

H = {"msg": 0.56, "lost": 0.56, "note": 0.46, "banner": 0.50, "phase": 0.52, "self": 0.62}
y, laid = 0.0, []
for kind, *rest in rows:
    y -= H[kind]
    laid.append((kind, y, rest))
    y -= 0.10
bottom = y - 0.25

fig, ax = plt.subplots(figsize=(13.4, 1.15 - bottom))
ax.set_xlim(0, 12.7)
ax.set_ylim(bottom, 1.85)
ax.axis("off")

for key, label in [("AG", "ReSIDS agent\n(one per IED)"),
                   ("RD", "Resource Directory\n(RFC 9176)"),
                   ("MO", "Monitor\n(external)")]:
    st, x = STY[key], X[key]
    ax.add_patch(FancyBboxPatch((x - 1.55, 1.10), 3.1, 0.58,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=st["fc"], ec=st["ec"], lw=1.7))
    ax.text(x, 1.39, label, ha="center", va="center", fontsize=12,
            fontweight="bold", color=st["ec"])
    ax.plot([x, x], [1.10, bottom + 0.12], ls=(0, (4, 4)), color="0.6", lw=1)


def arrow(y, x0, x1, text, kind, lost=False):
    xe = x0 + (x1 - x0) * (0.55 if lost else 1.0)
    solid = kind == "req"
    ax.annotate("", xy=(xe, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>",
                                color="#b2182b" if lost else ("0.1" if solid else "0.45"),
                                lw=1.8 if solid else 1.2,
                                ls="-" if solid else (0, (5, 3))))
    if lost:
        ax.plot(xe, y, marker="x", ms=13, mew=3, color="#b2182b")
    ax.text((x0 + xe) / 2, y + 0.11, text, ha="center", va="bottom",
            fontsize=10.5 if solid else 9.8,
            color="0.15" if solid else "0.4", style="normal" if solid else "italic",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


for kind, y0, rest in laid:
    if kind == "banner":
        ax.add_patch(FancyBboxPatch((0.9, y0 - 0.02), 10.9, 0.36,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc="#f2f2f2", ec="0.7", lw=0.9))
        ax.text(6.35, y0 + 0.16, rest[0], ha="center", va="center", fontsize=10.5, color="0.3")
    elif kind == "phase":
        ax.add_patch(FancyBboxPatch((0.5, y0 - 0.02), 11.8, 0.36,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc="#eef3fa", ec="#9bb4d4", lw=1.0))
        ax.text(0.75, y0 + 0.16, rest[0], ha="left", va="center", fontsize=11,
                fontweight="bold", color="#2c4d7c")
    elif kind == "note":
        ax.text(6.35, y0 + 0.14, rest[0], ha="center", va="center", fontsize=10,
                color="0.45", style="italic")
    elif kind == "self":
        x = X[rest[0]]
        ax.add_patch(FancyBboxPatch((x - 0.45, y0 - 0.04), 7.6, 0.46,
                     boxstyle="round,pad=0.10,rounding_size=0.08",
                     fc="#fff8ec", ec="#e0c27a", lw=1.1))
        ax.text(x - 0.30, y0 + 0.19, rest[1], ha="left", va="center", fontsize=10.5,
                color="#8a5a00")
    else:
        frm, to, text, k = rest
        arrow(y0, X[frm], X[to], text, k, lost=(kind == "lost"))

ax.set_title("Decentralized observability: Resource Directory + Observe + since= reconciliation",
             fontsize=15, fontweight="bold")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/decentralized_monitoring_sequence.png", dpi=170, bbox_inches="tight")
fig.savefig("results/decentralized_monitoring_sequence.pdf", bbox_inches="tight")
print("ok -> results/decentralized_monitoring_sequence.{png,pdf}")
