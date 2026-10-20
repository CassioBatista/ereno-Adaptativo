#!/usr/bin/env python3
"""Sequence diagram: the API between the colocated Disaster-FD monitor and the ReSIDS agent
along Trace 2 (docs/disaster_fd_trace2_scenario.md), API 1.9.0, same cutoffs as Trace 1.

REST binding over loopback (docs/TRUST_LEVEL.md §5, docs/API.md §2): no network discovery,
since the monitor is on the agent's host and knows its endpoint by configuration; start-up
check with GET /health and GET /status; one POST /trust_level per observation (204); the
agent's decisions readable at GET /status. The monitor probes its targets (aggregator,
ring predecessor and successor) and detects their failures. No learning traffic, no
intrusion handling. The client shown is client 8 (peers 7 and 9): it keeps gossiping when
clients 4 and 5 are lost (45 min), falls to local when its predecessor is lost (60 min), and
returns to federated with the aggregator (75 min, TL 65).

Solid arrow = probe / publication, dashed = reply, red X = lost probe.
Out: results/tl_trace2_sequence.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

X = {"TG": 1.9, "FD": 6.2, "AG": 10.6}
STY = {"TG": dict(fc="#f2f2f2", ec="#6b6b6b"),
       "FD": dict(fc="#e6f4ea", ec="#3a7d4f"),
       "AG": dict(fc="#e8eef7", ec="#3b5b92")}
LABEL = {"TG": "Targets\n(aggregator · peers 7 and 9)",
         "FD": "Disaster-FD\n(monitor M08)",
         "AG": "ReSIDS agent\n(client 8)"}
RED, BLUE, GREY = "#b2182b", "#3b5b92", "#3a7d4f"

rows = [
    ("banner", "TL = aggregator 50 + predecessor 15 + successor 15  ·  TL < 30 local  ·  "
               "30 ≤ TL < 50 gossip  ·  TL ≥ 50 federated  ·  settle S = 5 s"),
    ("phase", "0 min · start-up  —  colocated: no network discovery, the agent's endpoint is configured"),
    ("self", "FD", "agent endpoint from configuration: http://127.0.0.1:8722", GREY),
    ("msg", "FD", "AG", "GET /health", "req"),
    ("msg", "AG", "FD", "200 {status: ok}", "ack"),
    ("msg", "FD", "AG", "GET /status", "req"),
    ("msg", "AG", "FD", "200 {api_version 1.9.0, current_mode local, tl_bands {gl 30, fl 50}, trust {stale}}", "ack"),
    ("lost", "FD", "TG", "probe aggregator, peer 7, peer 9  (all isolated)", "req"),
    ("msg", "FD", "AG", "POST /trust_level {TL 0}", "req"),
    ("msg", "AG", "FD", "204", "ack"),
    ("phase", "15 min · all targets reachable"),
    ("msg", "FD", "TG", "probe aggregator, peer 7, peer 9   (every 5 s)", "req"),
    ("msg", "TG", "FD", "replies", "ack"),
    ("msg", "FD", "AG", "POST /trust_level {target aggregator, trusted, TL 80}", "req"),
    ("msg", "AG", "FD", "204", "ack"),
    ("self", "AG", "federated held 5 s  →  commit local → federated  (architecture_change, seq 1)", BLUE),
    ("phase", "30 min · aggregator lost"),
    ("lost", "FD", "TG", "probe aggregator  (timeout)", "req"),
    ("msg", "FD", "AG", "POST /trust_level {target aggregator, untrusted, TL 30}", "req"),
    ("msg", "AG", "FD", "204", "ack"),
    ("self", "AG", "gossip held 5 s  →  commit federated → gossip  (seq 2)", BLUE),
    ("phase", "45 min · clients 4 and 5 lost  —  peers of client 8 unchanged: TL stays 30, still gossip"),
    ("phase", "60 min · clients 2, 3, 6 and 7 lost  (region 2–7 down)"),
    ("lost", "FD", "TG", "probe peer 7  (timeout)", "req"),
    ("msg", "FD", "AG", "POST /trust_level {target predecessor, untrusted, TL 15}", "req"),
    ("msg", "AG", "FD", "204", "ack"),
    ("self", "AG", "local held 5 s  →  commit gossip → local  (seq 3; keeps detecting with its retained pool)", BLUE),
    ("phase", "75 min · aggregator restored  (clients 2–7 still lost)"),
    ("msg", "FD", "TG", "probe aggregator", "req"),
    ("msg", "TG", "FD", "reply", "ack"),
    ("msg", "FD", "AG", "POST /trust_level {target aggregator, trusted, TL 65}", "req"),
    ("msg", "AG", "FD", "204", "ack"),
    ("self", "AG", "federated held 5 s  →  commit local → federated  (seq 4)", BLUE),
    ("msg", "FD", "AG", "GET /status", "req"),
    ("msg", "AG", "FD", "200 {current_mode federated, trust {TL 65, band federated, candidate null}, last_seq 4}", "ack"),
    ("note", "Disaster-FD detects failures and publishes the trust level; the agent decides the mode. "
             "Its decisions are readable at GET /status and GET /events"),
    ("note", "CoAP binding: the agent subscribes instead, GET /trust_level with Observe on the monitor, "
             "and receives one notification per observation"),
    ("note", "Simulation: a trace player takes the monitor's place and issues the same POST /trust_level, "
             "replaying the Disaster-FD trace in time order"),
]

H = {"msg": 0.56, "lost": 0.56, "note": 0.46, "banner": 0.50, "phase": 0.52, "self": 0.6}
y, laid = 0.95, []
for kind, *rest in rows:
    y -= H[kind]
    laid.append((kind, y, rest))
    y -= 0.10
bottom = y - 0.25

fig, ax = plt.subplots(figsize=(13.0, 1.3 - bottom))
ax.set_xlim(0, 12.7)
ax.set_ylim(bottom, 1.95)
ax.axis("off")

for key in X:
    st, x = STY[key], X[key]
    ax.add_patch(FancyBboxPatch((x - 1.5, 1.18), 3.0, 0.6,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=st["fc"], ec=st["ec"], lw=1.7))
    ax.text(x, 1.48, LABEL[key], ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=st["ec"])
    ax.plot([x, x], [1.18, bottom + 0.12], ls=(0, (4, 4)), color="0.6", lw=1)


def arrow(y, x0, x1, text, kind, lost=False):
    xe = x0 + (x1 - x0) * (0.55 if lost else 1.0)
    solid = kind == "req"
    col = RED if lost else ("0.1" if solid else "0.45")
    ax.annotate("", xy=(xe, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.9 if solid else 1.2,
                                ls="-" if solid else (0, (5, 3))))
    if lost:
        ax.plot(xe, y, marker="x", ms=13, mew=3, color=RED)
    w = 0.078 * len(text)
    xt = min(max((x0 + xe) / 2, 0.15 + w / 2), 12.6 - w / 2)
    ax.text(xt, y + 0.11, text, ha="center", va="bottom", fontsize=10.3 if solid else 9.8,
            color="0.15" if solid else "0.4", style="normal" if solid else "italic",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


for kind, y0, rest in laid:
    if kind == "banner":
        ax.add_patch(FancyBboxPatch((0.15, y0 - 0.02), 12.4, 0.36,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc="#f2f2f2", ec="0.7", lw=0.9))
        ax.text(6.35, y0 + 0.16, rest[0], ha="center", va="center", fontsize=10.1, color="0.3")
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
        width = 0.082 * len(text) + 0.5
        x_left = max(0.2, min(x - 0.45, 12.55 - width))
        ax.add_patch(FancyBboxPatch((x_left, y0 - 0.04), min(12.55 - x_left, width), 0.44,
                     boxstyle="round,pad=0.10,rounding_size=0.08",
                     fc="#e6f4ea" if col == GREY else "#eef3fa",
                     ec="#9cc7aa" if col == GREY else "#9bb4d4", lw=1.1))
        ax.text(x_left + 0.15, y0 + 0.18, text, ha="left", va="center", fontsize=10.3, color=col)
    else:
        frm, to, text, k = rest
        arrow(y0, X[frm], X[to], text, k, lost=(kind == "lost"))

ax.set_title("Trace 2: the Disaster-FD trust-level API (colocated monitor, REST over loopback)",
             fontsize=15, fontweight="bold")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/tl_trace2_sequence.png", dpi=170, bbox_inches="tight")
fig.savefig("results/tl_trace2_sequence.pdf", bbox_inches="tight")
print("ok -> results/tl_trace2_sequence.{png,pdf}")
