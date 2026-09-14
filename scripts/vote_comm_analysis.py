#!/usr/bin/env python3
"""Communication cost of vote dissemination piggybacked on the GLow gossip.

Redo of the comm analysis WITH the control-plane votes. Votes ride the same GLow
gossip messages as boosters; the added payload is a small who-voted digest per
message (unsigned = ceil(N/8)-byte bitmap; signed = bitmap + one ~96 B aggregate
signature for the Byzantine regime). Overhead = GL messages x digest, compared to
the measured GL booster traffic (results/escalabilidade.csv). Also reports the
decentralized-quorum latency (rounds to network-wide quorum on a ring), the *other*
cost of dissemination. Out: results/vote_comm.csv + .png/.pdf
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.topology import build_topology
from fd.vote_diffusion import VoteDiffusion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGG_SIG = 96  # bytes, one aggregate (e.g. BLS) signature per message (signed regime)

rows = []
with open("results/escalabilidade.csv") as fh:
    for r in csv.DictReader(fh):
        N = int(r["N"]); msgs = int(r["GL_comm"]); gcb = int(r["GL_comm_bytes"])
        d_uns = VoteDiffusion.digest_bytes(N, signed=False)
        d_sig = VoteDiffusion.digest_bytes(N, signed=True, agg_sig_bytes=AGG_SIG)
        ov_uns = msgs * d_uns
        ov_sig = msgs * d_sig
        # decentralized quorum latency (ring), all cast
        vd = VoteDiffusion(build_topology("ring", N))
        for node in range(N):
            vd.cast(node)
        lat = vd.rounds_to_network_quorum(N // 2 + 1)
        rows.append({
            "N": N, "GL_msgs": msgs, "GL_comm_MB": gcb / 1e6,
            "digest_unsigned_B": d_uns, "digest_signed_B": d_sig,
            "vote_overhead_unsigned_pct": 100 * ov_uns / gcb,
            "vote_overhead_signed_pct": 100 * ov_sig / gcb,
            "quorum_latency_rounds_ring": lat,
        })

# print + CSV
hdr = ["N", "GL_msgs", "GL_comm_MB", "digest_unsigned_B", "digest_signed_B",
       "vote_overhead_unsigned_pct", "vote_overhead_signed_pct", "quorum_latency_rounds_ring"]
print(f"{'N':>4}{'GLcomm(MB)':>12}{'unsigned%':>11}{'signed%':>10}{'quorum@ring':>13}")
for r in rows:
    print(f"{r['N']:>4}{r['GL_comm_MB']:>12.1f}{r['vote_overhead_unsigned_pct']:>11.3f}"
          f"{r['vote_overhead_signed_pct']:>10.3f}{str(r['quorum_latency_rounds_ring']):>13}")
os.makedirs("results", exist_ok=True)
with open("results/vote_comm.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in hdr})

# figure: overhead % (left) + quorum latency (right) vs N
Ns = [r["N"] for r in rows]
fig, ax1 = plt.subplots(figsize=(8.5, 5))
ax1.plot(Ns, [r["vote_overhead_signed_pct"] for r in rows], "-o", color="#b2182b",
         lw=2.2, ms=7, label="comm overhead, signed (+agg sig)")
ax1.plot(Ns, [r["vote_overhead_unsigned_pct"] for r in rows], "--s", color="#e08214",
         lw=2, ms=7, label="comm overhead, unsigned (bitmap)")
ax1.set_xlabel("Number of peers ($N$)")
ax1.set_ylabel("vote comm overhead (% of GL traffic)")
ax1.set_ylim(0, max(1.0, max(r["vote_overhead_signed_pct"] for r in rows) * 1.3))
ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(Ns, [r["quorum_latency_rounds_ring"] for r in rows], "-^", color="#1b7837",
         lw=2.2, ms=7, label="quorum latency (ring)")
ax2.set_ylabel("decentralized-quorum latency (rounds)", color="#1b7837")
ax2.tick_params(axis="y", labelcolor="#1b7837")
ax1.set_title("Vote dissemination over GLow: tiny comm overhead, latency ~$N$\n"
              "(piggyback digest on booster gossip)")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=9)
fig.tight_layout()
fig.savefig("results/vote_comm.png", dpi=170); fig.savefig("results/vote_comm.pdf")
print("\n[vote-comm] -> results/vote_comm.{csv,png,pdf}")
