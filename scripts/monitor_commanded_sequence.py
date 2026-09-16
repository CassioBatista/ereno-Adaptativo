#!/usr/bin/env python3
"""Sequence diagram: monitor-COMMANDED FL->GL switch with deadline fallback.

Architecture-only figure (the mechanism is NOT implemented; v2 default stays
autonomous). ReSIDS detects a peer timeout, REPORTS to the monitor and WAITS in a
PENDING state (deadline D). Two outcomes: (alt A) the monitor commands the switch
via server-side RPC set_mode; (alt B) the deadline expires with no command (monitor
unreachable / partition) and ReSIDS falls back to autonomous fail-fast self-protect.
Out: results/monitor_commanded_sequence.{png,pdf}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

XL, XR = 2.6, 9.0
RES = dict(fc="#e8eef7", ec="#3b5b92")
MON = dict(fc="#e6f4ea", ec="#3a7d4f")
CMD = "#6a3d9a"                                   # command (actuation) colour

y = 0.0
ops = []           # draw ops with resolved y
alts = []          # (y_top, y_bot, label, colour)

def adv(h):
    global y; y -= h

def note_over(side, text, h=0.5):
    adv(h); x = XL if side == "L" else XR
    ops.append(("note_over", x, y, text)); adv(0.12)

def note_center(text, h=0.45):
    adv(h); ops.append(("note_center", (XL+XR)/2, y, text)); adv(0.12)

def req(text, ack, color="0.1"):
    adv(0.55); ops.append(("arrow", XL, XR, y, text, color, "-"));
    adv(0.46); ops.append(("arrow", XR, XL, y, ack, "0.45", "ack")); adv(0.12)

def cmd(text, result):
    adv(0.55); ops.append(("arrow", XR, XL, y, text, CMD, "-"))
    adv(0.46); ops.append(("arrow", XL, XR, y, result, CMD, "ack")); adv(0.12)

# --- common prefix ---
note_over("L", "peer timeout (T rounds) detected → PENDING; start deadline timer (D rounds)")
req("POST /telemetry   ·   node_failure {failed_node, round}", "2.01 Created")
req("POST /attributes   ·   switch_pending {FL→GL, since round r}", "2.04 Changed")

# --- alt A: commanded within deadline ---
adv(0.6); a_top = y
note_over("R", "operator / policy decides")
cmd("RPC  set_mode {to: gossip, reason: node_failure}", "RPC result {ok, mode: gossip}")
note_over("L", "commit FL→GL @ round r'")
req("POST /telemetry   ·   architecture_change {FL→GL, reason: commanded}", "2.01 Created")
a_bot = y - 0.05; alts.append((a_top, a_bot, "alt  [command within deadline  (≤ D rounds)]", CMD))

# --- alt B: deadline fallback ---
adv(0.6); b_top = y
note_over("L", "no command in D rounds  (monitor unreachable / partition)")
note_over("L", "→ autonomous fail-fast self-protect: commit FL→GL @ round r+D")
req("POST /telemetry   ·   architecture_change {FL→GL, reason: node_failure (autonomous fallback)}",
    "2.01 Created")
b_bot = y - 0.05; alts.append((b_top, b_bot, "alt  [deadline expires — no command]", "#b2182b"))

bottom = y - 0.25
fig, ax = plt.subplots(figsize=(12.2, 0.9 - bottom))
ax.set_xlim(0, 12); ax.set_ylim(bottom, 1.7); ax.axis("off")

# participant boxes + lifelines
for x, label, st in [(XL, "ReSIDS\n(adaptive IDS)", RES),
                     (XR, "Monitor\n(IoT Hub / ThingsBoard + operator/policy)", MON)]:
    ax.add_patch(FancyBboxPatch((x - 1.9, 1.02), 3.8, 0.55,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=st["fc"], ec=st["ec"], lw=1.6))
    ax.text(x, 1.29, label, ha="center", va="center", fontsize=10, fontweight="bold",
            color=st["ec"])
    ax.plot([x, x], [1.02, bottom + 0.1], ls=(0, (4, 4)), color="0.6", lw=1)

# alt frames (behind arrows)
for yt, yb, label, col in alts:
    ax.add_patch(Rectangle((XL - 2.0, yb), (XR - XL) + 4.0, yt - yb,
                 fill=False, ec=col, lw=1.3, ls=(0, (6, 3)), zorder=0))
    ax.add_patch(FancyBboxPatch((XL - 2.0, yt - 0.02), 3.6, 0.3,
                 boxstyle="round,pad=0.01,rounding_size=0.04",
                 fc=col, ec=col, zorder=1))
    ax.text(XL - 1.85, yt + 0.13, label.split("[")[0].strip(), ha="left", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=2)
    ax.text(XL + 1.75, yt + 0.13, "[" + label.split("[", 1)[1], ha="left", va="center",
            fontsize=8, color=col, style="italic", zorder=2)

def draw_arrow(x0, x1, yy, text, color, style):
    ls = (0, (5, 3)) if style == "ack" else "-"
    lw = 1.2 if style == "ack" else 1.8
    ax.annotate("", xy=(x1, yy), xytext=(x0, yy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls))
    it = style == "ack"
    ax.text((XL+XR)/2, yy + 0.10, text, ha="center", va="bottom",
            fontsize=7.8 if it else 8.5, color=color if it else "0.1",
            style="italic" if it else "normal")

for op in ops:
    if op[0] == "arrow":
        _, x0, x1, yy, text, color, style = op
        draw_arrow(x0, x1, yy, text, color, style)
    elif op[0] == "note_over":
        _, x, yy, text = op
        ax.text(x, yy + 0.1, text, ha="center", va="center", fontsize=7.8, color="0.4",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.25", fc="#f7f7f2", ec="0.75", lw=0.8))
    elif op[0] == "note_center":
        _, x, yy, text = op
        ax.text(x, yy + 0.1, text, ha="center", va="center", fontsize=8, color="0.5",
                style="italic")

ax.set_title("Monitor-commanded FL→GL switch, with deadline fallback (architecture)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/monitor_commanded_sequence.png", dpi=165)
fig.savefig("results/monitor_commanded_sequence.pdf")
print("ok -> results/monitor_commanded_sequence.{png,pdf}")
