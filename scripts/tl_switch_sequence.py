#!/usr/bin/env python3
"""Sequence diagram: the trust-level switch of API 1.9.0 (docs/TRUST_LEVEL.md).

Disaster-FD detects failures and publishes a trust level (TL); the agent decides its mode
from it. Four lifelines: the monitored targets (aggregator and ring neighbours), the
Disaster-FD monitor colocated with the client, the ReSIDS agent, and the regional
Disaster-FD, which only observes the agent's events.

The values are those of the 14-client pilot run (scripts/tl_policy_replay.py): client 13 at
the 30-min fault (TL 80 -> 30 -> 15 within 0.4 s; with S = 0 it would switch twice), the
45-min partial recovery (TL 30) and the 60-min return of the aggregator (TL 80).
Supersedes scripts/monitor_commanded_sequence.py (API 1.7.0, Disaster-FD-commanded switch).

Solid arrow = probe / observation / notification, dashed = reply, red X = lost probe.
Out: results/tl_switch_sequence.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

X = {"TG": 1.5, "FD": 4.6, "AG": 7.9, "RF": 11.2}
STY = {"TG": dict(fc="#f2f2f2", ec="#6b6b6b"),
       "FD": dict(fc="#e6f4ea", ec="#3a7d4f"),
       "AG": dict(fc="#e8eef7", ec="#3b5b92"),
       "RF": dict(fc="#f3ecf7", ec="#6a3d9a")}
LABEL = {"TG": "Targets\n(aggregator · ring peers)",
         "FD": "Disaster-FD\n(colocated monitor)",
         "AG": "ReSIDS agent\n(client 13)",
         "RF": "Disaster-FD\n(regional, observer)"}
RED, GREEN, BLUE = "#b2182b", "#3a7d4f", "#3b5b92"

rows = [
    ("banner", "API 1.9.0  ·  TL < 30 local  ·  30 ≤ TL < 50 gossip  ·  TL ≥ 50 federated  ·  "
               "settle S = 5 s  ·  stale after F = 15 s"),
    ("note", "TL = sum of the impact factors of the trusted targets: aggregator 50, predecessor 15, successor 15  →  TL ∈ {0, 15, 30, 50, 65, 80}"),
    ("phase", "1 · Steady state  —  federated, one probe per target every 5 s"),
    ("msg", "FD", "TG", "probe  (aggregator, peer 12, peer 14)", "req"),
    ("msg", "TG", "FD", "replies", "ack"),
    ("msg", "FD", "AG", "trust_level {TL 80, trusted_system}   (host-local: POST /trust_level)", "req"),
    ("self", "AG", "band federated = current mode  →  nothing to do", BLUE),
    ("phase", "2 · Network instability (30 min)  —  the TL passes through an intermediate band"),
    ("lost", "FD", "TG", "probe aggregator  (times out)", "req"),
    ("msg", "FD", "AG", "trust_level {target: aggregator, trusted: false, TL 30}", "req"),
    ("self", "AG", "candidate gossip, since t", BLUE),
    ("lost", "FD", "TG", "probe peer 14  (times out, 0.4 s later)", "req"),
    ("msg", "FD", "AG", "trust_level {target: successor, trusted: false, TL 15}", "req"),
    ("self", "AG", "candidate local  (gossip held 0.4 s < S: never committed)", BLUE),
    ("self", "AG", "local held S = 5 s  →  commit federated → local", BLUE),
    ("msg", "AG", "RF", "notify · architecture_change {federated→local, reason: trust_level, TL 15}   [decided_by: agent]", "req"),
    ("note", "pilot run: with S = 5 s every client makes exactly the 4 nominal switches (5–8 with S = 0); the 14 clients switch within 4.9 s of each other"),
    ("phase", "3 · Partial recovery (45 min) and return of the aggregator (60 min)"),
    ("msg", "FD", "AG", "trust_level {TL 30}   — both peers trusted again", "req"),
    ("self", "AG", "gossip held 5 s  →  commit local → gossip", BLUE),
    ("msg", "AG", "RF", "notify · architecture_change {local→gossip, reason: trust_level, TL 30}   [decided_by: agent]", "req"),
    ("msg", "FD", "AG", "trust_level {TL 80}   — aggregator trusted again", "req"),
    ("self", "AG", "federated held 5 s  →  commit gossip → federated  (re-accepting the hub on persistent evidence only)", BLUE),
    ("msg", "AG", "RF", "notify · architecture_change {gossip→federated, reason: trust_level, TL 80}   [decided_by: agent]", "req"),
    ("phase", "4 · Monitor failure  —  a silent monitor is not evidence of a healthy hub"),
    ("self", "FD", "✗  colocated monitor crashes or hangs: no observation", RED),
    ("self", "AG", "no TL for F = 15 s  →  commit → local   (reason: tl_stale, decided_by: agent)", RED),
    ("msg", "AG", "RF", "notify · architecture_change {federated→local, reason: tl_stale}", "req"),
    ("note", "replaces the watchdog D of 1.5.0 (FL→GL only); never fires in the pilot run (largest gap between observations 4.8 s)"),
    ("note", "no command reaches the agent: Disaster-FD cannot set a mode  ·  intrusion notification and isolation: future work (Byzantine model)"),
]

H = {"msg": 0.56, "lost": 0.56, "note": 0.46, "banner": 0.50, "phase": 0.52, "self": 0.6}
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

for key in X:
    st, x = STY[key], X[key]
    ax.add_patch(FancyBboxPatch((x - 1.35, 1.18), 2.7, 0.6,
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
    w = 0.078 * len(text)                               # approx. label width (axis units)
    xt = min(max((x0 + xe) / 2, 0.15 + w / 2), 12.6 - w / 2)   # keep long labels inside
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
        x_left = min(x - 0.45, 12.55 - (0.082 * len(text) + 0.5))
        width = min(12.55 - x_left, 0.082 * len(text) + 0.5)
        ax.add_patch(FancyBboxPatch((x_left, y0 - 0.04), width, 0.44,
                     boxstyle="round,pad=0.10,rounding_size=0.08",
                     fc="#fdeceb" if col == RED else "#eef3fa",
                     ec="#e6a3a0" if col == RED else "#9bb4d4", lw=1.1))
        ax.text(x_left + 0.15, y0 + 0.18, text, ha="left", va="center", fontsize=10.3, color=col)
    else:
        frm, to, text, k = rest
        arrow(y0, X[frm], X[to], text, k, lost=(kind == "lost"))

ax.set_title("Trust-level switching (API 1.9.0): Disaster-FD supplies the evidence, the agent decides",
             fontsize=15, fontweight="bold")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/tl_switch_sequence.png", dpi=170, bbox_inches="tight")
fig.savefig("results/tl_switch_sequence.pdf", bbox_inches="tight")
print("ok -> results/tl_switch_sequence.{png,pdf}")
