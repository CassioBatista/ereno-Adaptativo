#!/usr/bin/env python3
"""CoAP message sequence at the ReSIDS <-> Monitor event interface (Paper 2 figure).

Only the CoAP messages between the two parties (no ReSIDS/monitor internals):
ReSIDS is the CoAP client / IoT device; the Monitor is the CoAP server (IoT hub /
ThingsBoard). Solid arrow = request (POST, CON), dashed = ACK. Events go on
/telemetry (time-series); current state on /attributes.

Includes the intrusion_detected event (notification only -- ReSIDS reports, the
operator/monitor decides containment; ReSIDS never isolates/quarantines).
Out: results/coap_sequence.{png,pdf}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

XL, XR = 2.3, 8.7                     # lifeline x for ReSIDS / Monitor
RES = dict(fc="#e8eef7", ec="#3b5b92")
MON = dict(fc="#e6f4ea", ec="#3a7d4f")

# rows: ("msg", req, ack, note) or ("note", text) or ("banner", text)
rows = [
    ("banner", "CoAP device API  ·  POST  ·  Confirmable (CON)  ·  DTLS"),
    ("msg", "POST /attributes   ·   state: {mode: federated, active_nodes}", "2.04 Changed", None),
    ("msg", "POST /telemetry   ·   intrusion_detected {attack, k_votes, source_node: null}",
            "2.01 Created", "notification only — ReSIDS reports; operator/monitor decides (no isolation)"),
    ("msg", "POST /telemetry   ·   node_failure {failed_node, round}", "2.01 Created", None),
    ("msg", "POST /telemetry   ·   architecture_change {FL→GL, reason: node_failure}", "2.01 Created", None),
    ("note", "(network shrinks — further node_failure / intrusion_detected events pushed as they occur)"),
    ("msg", "POST /telemetry   ·   architecture_change {GL→FL, reason: recovery}", "2.01 Created", None),
    ("msg", "POST /attributes   ·   state: {mode: federated, active_nodes}", "2.04 Changed", None),
]

# vertical layout
y = 0.0
ys = []
H_MSG, H_ACK, H_NOTE, H_BAN = 0.62, 0.5, 0.5, 0.5
for kind, *rest in rows:
    if kind == "msg":
        y -= H_MSG; y_req = y
        y -= H_ACK; y_ack = y
        ys.append(("msg", y_req, y_ack, rest))
    elif kind == "note":
        y -= H_NOTE; ys.append(("note", y, None, rest))
    elif kind == "banner":
        y -= H_BAN; ys.append(("banner", y, None, rest))
    y -= 0.12
bottom = y - 0.2

fig, ax = plt.subplots(figsize=(12.6, 0.9 - bottom))
ax.set_xlim(0, 11); ax.set_ylim(bottom, 1.7)
ax.axis("off")

# participant boxes + lifelines
for x, label, st in [(XL, "ReSIDS\n(adaptive IDS)", RES),
                     (XR, "Monitor\n(IoT Hub / ThingsBoard)", MON)]:
    ax.add_patch(FancyBboxPatch((x - 2.15, 1.02), 4.3, 0.55,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=st["fc"], ec=st["ec"], lw=1.6))
    ax.text(x, 1.29, label, ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=st["ec"])
    ax.plot([x, x], [1.02, bottom + 0.1], ls=(0, (4, 4)), color="0.6", lw=1)

def req(y, text):
    ax.annotate("", xy=(XR, y), xytext=(XL, y),
                arrowprops=dict(arrowstyle="-|>", color="0.1", lw=1.8))
    ax.text((XL + XR) / 2, y + 0.12, text, ha="center", va="bottom", fontsize=11)

def ack(y, text):
    ax.annotate("", xy=(XL, y), xytext=(XR, y),
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.2, ls=(0, (5, 3))))
    ax.text((XL + XR) / 2, y + 0.10, text, ha="center", va="bottom", fontsize=9.8,
            color="0.4", style="italic")

for kind, a, b, rest in ys:
    if kind == "banner":
        ax.add_patch(FancyBboxPatch(((XL + XR) / 2 - 3.1, a - 0.02), 6.2, 0.34,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc="#f2f2f2", ec="0.7", lw=0.9))
        ax.text((XL + XR) / 2, a + 0.15, rest[0], ha="center", va="center",
                fontsize=10, color="0.35")
    elif kind == "note":
        ax.text((XL + XR) / 2, a + 0.1, rest[0], ha="center", va="center",
                fontsize=10, color="0.5", style="italic")
    else:
        text, ackt, sidenote = rest
        req(a, text)
        ack(b, ackt)
        if sidenote:
            ax.text(XR + 0.15, b - 0.02, sidenote, ha="right", va="top", fontsize=9,
                    color="#8a5a00", style="italic", wrap=True,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fff8ec", ec="#e0c27a", lw=0.8))

ax.set_title("ReSIDS ↔ Monitor event interface (CoAP)", fontsize=15, fontweight="bold")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/coap_sequence.png", dpi=170)
fig.savefig("results/coap_sequence.pdf")
print("ok -> results/coap_sequence.{png,pdf}")
