#!/usr/bin/env python3
"""Sequence diagram: the Disaster-FD-commanded FL->GL, with the agent's watchdog.

Architecture of 2026-10-02 (docs/decentralized_monitoring.md 1.1): two roles only,
Disaster-FD (one monitor process per node and per server, federated by region) and the
ReSIDS agent. Three lifelines: the agent, its node-local Disaster-FD monitor, and the
federated Disaster-FD.

  alt A  local FD alive  -> it commands FL->GL from LOCAL evidence (decided_by=monitor)
  alt B  local FD down   -> the agent's watchdog switches after D rounds
                            (decided_by=autonomous, reason=autonomous_fallback)
  later  GL->FL          -> federated FD decides (fan-out, unanimity), NO fallback

Round numbers are the measured ones from conf/scenarios/{availability,fd_watchdog}.yaml.
Out: results/monitor_commanded_sequence.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

XA, XL, XF = 2.0, 6.7, 11.4                       # agent, node-local FD, federated FD
STY = {"A": dict(fc="#e8eef7", ec="#3b5b92"),
       "L": dict(fc="#e6f4ea", ec="#3a7d4f"),
       "F": dict(fc="#f3ecf7", ec="#6a3d9a")}
CMD = "#3a7d4f"
RED = "#b2182b"

y = 0.85          # start just under the participant boxes (they sit at y = 1.04)
ops, frames = [], []


def adv(h):
    global y
    y -= h


def note(x, text, h=0.52, color="0.4", fc="#f7f7f2", ec="0.75"):
    adv(h)
    ops.append(("note", x, y, text, color, fc, ec))
    adv(0.12)


def center(text, h=0.45, color="0.45"):
    adv(h)
    ops.append(("center", y, text, color))
    adv(0.12)


def msg(x0, x1, text, color="0.1", style="-"):
    adv(0.56)
    ops.append(("arrow", x0, x1, y, text, color, style))
    adv(0.10)


# ---------------------------------------------------------------- common prefix
note(XL, "round r: inactivity observed by the node-local Disaster-FD\n"
         "local evidence only — no regional quorum", h=0.62)

# ---------------------------------------------------------------- alt A
adv(0.45); top = y
msg(XL, XA, "set_mode {to: gossip, reason: node_failure}    (local IPC)", CMD)
msg(XA, XL, "result {applied, effective at the next round boundary}", "0.45", "ack")
note(XA, "commit FL→GL @ r + detect_lag\n(measured: round 102)", h=0.62)
msg(XA, XL, "architecture_change {FL→GL}   decided_by: monitor")
msg(XL, XF, "event federated to the regional view", "0.45", "ack")
frames.append((top, y - 0.08, "alt", "[node-local Disaster-FD alive]", CMD))

# ---------------------------------------------------------------- alt B
adv(0.55); top = y
note(XL, "✗  local FD process crashed / hung", color=RED, fc="#fdeceb", ec="#e6a3a0")
note(XA, "watchdog: no command from its own FD for D rounds", h=0.52)
note(XA, "commit FL→GL @ r + detect_lag + D\n(measured: round 105, D = 3)", h=0.62)
msg(XA, XF, "architecture_change {FL→GL, reason: autonomous_fallback}   decided_by: autonomous",
    RED)
center("the ONLY autonomous action — towards the safe mode only; the event is read on the "
       "next since= poll", color=RED)
frames.append((top, y - 0.08, "alt", "[node-local Disaster-FD down]", RED))

# ---------------------------------------------------------------- later: GL -> FL
adv(0.55); top = y
note(XF, "membership full for dwell rounds\n(regional evidence)", h=0.62)
msg(XF, XA, "set_mode {to: federated, reason: recovery}    (fan-out to all 15 · unanimity)",
    "#6a3d9a")
msg(XA, XF, "result {applied}", "0.45", "ack")
center("no fallback of any kind: if the federation cannot decide, the system rests in GL")
frames.append((top, y - 0.08, "later", "[GL→FL — federated Disaster-FD decides]", "#6a3d9a"))

bottom = y - 0.3

# ---------------------------------------------------------------- render
fig, ax = plt.subplots(figsize=(14.2, 1.2 - bottom))
ax.set_xlim(0, 13.4)
ax.set_ylim(bottom, 1.75)
ax.axis("off")

for x, key, label in [(XA, "A", "ReSIDS agent"),
                      (XL, "L", "Disaster-FD\n(node-local monitor)"),
                      (XF, "F", "Disaster-FD\n(federated, regional)")]:
    st = STY[key]
    ax.add_patch(FancyBboxPatch((x - 1.6, 1.04), 3.2, 0.58,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=st["fc"], ec=st["ec"], lw=1.7))
    ax.text(x, 1.33, label, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=st["ec"])
    ax.plot([x, x], [1.04, bottom + 0.1], ls=(0, (4, 4)), color="0.6", lw=1)

for yt, yb, tag, label, col in frames:
    ax.add_patch(Rectangle((0.15, yb), 13.1, yt - yb, fill=False, ec=col, lw=1.3,
                           ls=(0, (6, 3)), zorder=0))
    ax.add_patch(FancyBboxPatch((0.15, yt - 0.02), 0.9, 0.3,
                 boxstyle="round,pad=0.01,rounding_size=0.04", fc=col, ec=col, zorder=1))
    ax.text(0.6, yt + 0.13, tag, ha="center", va="center", fontsize=10, color="white",
            fontweight="bold", zorder=2)
    ax.text(1.2, yt + 0.13, label, ha="left", va="center", fontsize=10, color=col,
            style="italic", zorder=2)

for op in ops:
    if op[0] == "arrow":
        _, x0, x1, yy, text, color, style = op
        ack = style == "ack"
        ax.annotate("", xy=(x1, yy), xytext=(x0, yy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2 if ack else 1.9,
                                    ls=(0, (5, 3)) if ack else "-"))
        ax.text((x0 + x1) / 2, yy + 0.10, text, ha="center", va="bottom",
                fontsize=9.6 if ack else 10.3, color=color if ack else "0.12",
                style="italic" if ack else "normal",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    elif op[0] == "note":
        _, x, yy, text, color, fc, ec = op
        ax.text(x, yy + 0.12, text, ha="center", va="center", fontsize=9.6, color=color,
                style="italic", bbox=dict(boxstyle="round,pad=0.28", fc=fc, ec=ec, lw=0.8))
    elif op[0] == "center":
        _, yy, text, color = op
        ax.text(6.7, yy + 0.1, text, ha="center", va="center", fontsize=9.8, color=color,
                style="italic")

ax.set_title("FL→GL commanded by the node-local Disaster-FD, with the agent's watchdog",
             fontsize=14.5, fontweight="bold")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/monitor_commanded_sequence.png", dpi=165, bbox_inches="tight")
fig.savefig("results/monitor_commanded_sequence.pdf", bbox_inches="tight")
print("ok -> results/monitor_commanded_sequence.{png,pdf}")
