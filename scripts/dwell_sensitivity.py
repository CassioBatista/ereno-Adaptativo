#!/usr/bin/env python3
"""Sensitivity of GL->FL recovery to dwell_rounds (control plane only).

dwell_rounds does NOT change detection F1 (GL holds flat regardless); it controls
(i) recovery latency after the network heals and (ii) resistance to flapping when
the network is briefly stable then fails again. This sweep drives the
DistributedArchManager state machine (no Flower) over two scenarios:
  A (clean recovery): node fails @10, returns @24 and stays up.
  B (unstable):       node fails @10, returns @20, fails again @22
                      (only a 2-round full-participation window).
cooldown_rounds=2 fixed, N=14. Out: results/dwell_sensitivity.csv + .png/.pdf
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.arch_manager import DistributedArchManager
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, COOLDOWN, ROUNDS = 14, 2, 40
SCEN = {
    "A_clean":     [{"round": 10, "nodes": [3]}, {"round": 24, "nodes": [3], "up": True}],
    "B_unstable":  [{"round": 10, "nodes": [3]}, {"round": 20, "nodes": [3], "up": True},
                    {"round": 22, "nodes": [3]}],
}
DWELLS = [1, 2, 3, 4, 5, 8]


def run(faults, dwell):
    mgr = DistributedArchManager(n_nodes=N, initial_mode="federated",
                                 dwell_rounds=dwell, cooldown_rounds=COOLDOWN, faults=faults)
    commits = []
    for r in range(1, ROUNDS + 1):
        active = mgr.get_active_clients(r)
        reported = set(active) if active is not None else set(range(N))
        mgr.observe(r, reported)
        rs = mgr.consume_switch_reason()
        if rs:
            commits.append((r, mgr.get_mode(r + 1), rs["reason"]))
    return commits


rows = []
for scen, faults in SCEN.items():
    for d in DWELLS:
        c = run(faults, d)
        n_switches = len(c)
        rec = next((r for (r, mode, reason) in c if reason == "recovery"), None)
        # flap = a recovery to FL followed by another FL->GL commit afterwards
        flap = any(c[i][2] == "recovery" and i + 1 < len(c) for i in range(len(c)))
        rows.append((scen, d, rec if rec else "-", n_switches, "yes" if flap else "no"))

print(f"{'scenario':11}{'dwell':>6}{'recovery@':>11}{'switches':>10}{'flap?':>7}")
for scen, d, rec, sw, flap in rows:
    print(f"{scen:11}{d:>6}{str(rec):>11}{sw:>10}{flap:>7}")

os.makedirs("results", exist_ok=True)
with open("results/dwell_sensitivity.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["scenario", "dwell", "recovery_round", "n_switches", "flap"])
    w.writerows(rows)

# figure: recovery round vs dwell (scenario A) + flap count (scenario B)
a = [(d, rec) for (s, d, rec, sw, fl) in rows if s == "A_clean" and rec != "-"]
b = [(d, sw) for (s, d, rec, sw, fl) in rows if s == "B_unstable"]
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot([d for d, _ in a], [r for _, r in a], "-o", color="#1b7837", lw=2.2, ms=8,
         label="A: recovery round (clean)")
ax1.set_xlabel("dwell_rounds"); ax1.set_ylabel("GL$\\rightarrow$FL recovery round", color="#1b7837")
ax1.tick_params(axis="y", labelcolor="#1b7837"); ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.plot([d for d, _ in b], [s for _, s in b], "--s", color="#b2182b", lw=2.2, ms=8,
         label="B: total switches (unstable)")
ax2.set_ylabel("switches in unstable scenario (flap = 3)", color="#b2182b")
ax2.tick_params(axis="y", labelcolor="#b2182b")
ax1.set_title("dwell_rounds: later recovery (A) vs. flap resistance (B)\n"
              "F1 is unaffected — this is timing/stability only")
ax1.axvline(3, color="0.6", ls=":", lw=1); ax1.text(3.05, ax1.get_ylim()[0], " deployed", fontsize=8, color="0.4")
fig.tight_layout()
fig.savefig("results/dwell_sensitivity.png", dpi=170)
fig.savefig("results/dwell_sensitivity.pdf")
print("\n[dwell] CSV + figure -> results/dwell_sensitivity.*")
