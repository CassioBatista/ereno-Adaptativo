#!/usr/bin/env python3
"""What the external monitor sees: the whole event sequence on one time axis.

Renders results/adaptive_events_demo.jsonl the way a monitor dashboard would show it:

  top    control plane — aggregation mode as a background band, active-node count as a
         step line, and the control events (node_failure, node_recovery,
         architecture_change) as markers on the timeline;
  bottom data plane — every intrusion_detected alarm as a point at its window, height =
         n_flags (alarm volume, log), colour = attack class, with the escalation
         threshold the monitor uses for triage.

Out: results/monitor_dashboard_view.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

IN = "results/adaptive_events_demo.jsonl"
ESCALATE = 10
MODE_C = {"federated": "#dbe6f5", "gossip": "#dcefe1"}
ATK_C = {"injection": "#b2182b", "random_replay": "#2166ac", "high_StNum": "#762a83",
         "poisoned_high_rate": "#e08214", "inverse_replay": "#1b7837",
         "masquerade_fake_fault": "#d6604d", "masquerade_fake_normal": "#8c510a"}

evs = [json.loads(l) for l in open(IN, encoding="utf-8")]
ctrl = [e for e in evs if e["type"] != "intrusion_detected"]
intr = [e for e in evs if e["type"] == "intrusion_detected"]
rmax = max(e["round"] for e in evs)

# mode spans from the architecture_change events
spans, cur, start = [], "federated", 0
for e in [x for x in ctrl if x["type"] == "architecture_change"]:
    spans.append((start, e["round"], cur))
    cur, start = e["to_mode"], e["round"]
spans.append((start, rmax, cur))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.2, 7.0), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.25]})

# ── top: control plane ────────────────────────────────────────────────────────
for a, b, m in spans:
    ax1.axvspan(a, b, color=MODE_C[m], zorder=0)
    ax1.text((a + b) / 2, 15.4, m.upper(), ha="center", va="top", fontsize=10,
             fontweight="bold", color="#3b5b92" if m == "federated" else "#3a7d4f")

xs, ys, n = [0], [14], 14
for e in ctrl:
    if e["type"] in ("node_failure", "node_recovery"):
        xs += [e["round"], e["round"]]; ys += [n, e["n_active"]]; n = e["n_active"]
xs.append(rmax); ys.append(n)
ax1.plot(xs, ys, color="0.25", lw=2, zorder=3, label="active nodes")

for e in ctrl:
    r = e["round"]
    if e["type"] == "node_failure":
        ax1.plot(r, e["n_active"], "v", color="#b2182b", ms=9, zorder=4)
    elif e["type"] == "node_recovery":
        ax1.plot(r, e["n_active"], "^", color="#1b7837", ms=9, zorder=4)
    else:
        ax1.axvline(r, color="#6a3d9a", ls="--", lw=1.8, zorder=2)
        ax1.annotate(f"{e['from_mode'][:2].upper()}→{e['to_mode'][:2].upper()}\nr{r}",
                     (r, 3.2), fontsize=9, color="#6a3d9a", fontweight="bold",
                     ha="left" if r < rmax / 2 else "right")
ax1.set_ylim(0, 16.5)
ax1.set_ylabel("active nodes")
ax1.set_title("What the monitor sees: control plane", fontsize=11.5, loc="left")
ax1.grid(alpha=0.3)
ax1.legend(handles=[Line2D([], [], color="0.25", lw=2, label="active nodes"),
                    Line2D([], [], marker="v", color="#b2182b", ls="", label="node_failure"),
                    Line2D([], [], marker="^", color="#1b7837", ls="", label="node_recovery"),
                    Line2D([], [], color="#6a3d9a", ls="--", label="architecture_change")],
           loc="lower right", fontsize=9, ncol=2, framealpha=0.95)

# ── bottom: data plane ────────────────────────────────────────────────────────
for a, b, m in spans:
    ax2.axvspan(a, b, color=MODE_C[m], zorder=0)
for atk in sorted({e["attack"] for e in intr}):
    pts = [(e["round"], max(e["n_flags"], 1)) for e in intr if e["attack"] == atk]
    ax2.scatter([p[0] for p in pts], [p[1] for p in pts], s=16,
                color=ATK_C.get(atk, "0.4"), label=f"{atk} ({len(pts)})", zorder=3, alpha=0.85)
ax2.axhline(ESCALATE, color="#b2182b", ls=":", lw=1.6, zorder=2)
ax2.text(rmax, ESCALATE * 1.25, f"escalation threshold  n_flags = {ESCALATE}",
         ha="right", fontsize=9, color="#b2182b")
ax2.set_yscale("log")
ax2.set_xlabel("round  (1 s window each)")
ax2.set_ylabel("n_flags  (alarm volume, log)")
ax2.set_title(f"data plane — {len(intr)} intrusion_detected alarms", fontsize=11.5, loc="left")
ax2.grid(alpha=0.3, which="both")
ax2.legend(loc="upper center", fontsize=8, ncol=4, framealpha=0.95)
ax2.set_xlim(0, rmax)

esc = sum(1 for e in intr if e["n_flags"] >= ESCALATE)
fig.suptitle(f"Monitor view of the full sequence — {len(evs)} events "
             f"({len(intr)} alarms, {esc} escalated; {len(ctrl)} control-plane)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs("results", exist_ok=True)
fig.savefig("results/monitor_dashboard_view.png", dpi=170, bbox_inches="tight")
fig.savefig("results/monitor_dashboard_view.pdf", bbox_inches="tight")
print("ok -> results/monitor_dashboard_view.{png,pdf}")
