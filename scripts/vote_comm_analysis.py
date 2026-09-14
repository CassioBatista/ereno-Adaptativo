#!/usr/bin/env python3
"""Communication cost of the CONTROL PLANE piggybacked on the GLow gossip.

Two control-plane signals ride the same GLow gossip messages as boosters:
  * mode votes      — who votes to switch FL<->GL (a who-voted bitmap, ceil(N/8) B);
  * peer suspicions — which nodes are locally timed-out as down (a down-set bitmap,
                      ceil(N/8) B), from fd/peer_failure.py.
So the piggyback digest per gossip message is 2*ceil(N/8) B (unsigned), or
+~96 B for one aggregate signature in the Byzantine (signed) regime. Overhead is
compared to the measured GL booster traffic (results/escalabilidade.csv). Also
reports the two decentralized latencies: mode-quorum convergence and peer-failure
network agreement (timeout + diffusion). Out: results/vote_comm.csv + .png/.pdf
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.topology import build_topology
from fd.vote_diffusion import VoteDiffusion
from fd.peer_failure import PeerFailureDetector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGG_SIG = 96      # bytes, one aggregate (e.g. BLS) signature / message (signed regime)
PF_TIMEOUT = 2    # peer-failure local timeout (rounds)


def peer_failure_latency(n):
    det = PeerFailureDetector(build_topology("ring", n), timeout=PF_TIMEOUT)
    down, fail_at = n // 2, 3
    for r in range(1, 6 * n):
        alive = set(range(n)) if r < fail_at else set(range(n)) - {down}
        det.observe_and_step(r, alive)
        if det.network_knows(down, alive):
            return r - fail_at
    return None


rows = []
with open("results/escalabilidade.csv") as fh:
    for r in csv.DictReader(fh):
        N = int(r["N"]); msgs = int(r["GL_comm"]); gcb = int(r["GL_comm_bytes"])
        bmp = math.ceil(N / 8)
        d_uns = 2 * bmp                 # mode-vote bitmap + suspicion down-set bitmap
        d_sig = 2 * bmp + AGG_SIG
        vd = VoteDiffusion(build_topology("ring", N))
        for node in range(N):
            vd.cast(node)
        rows.append({
            "N": N, "GL_msgs": msgs, "GL_comm_MB": gcb / 1e6,
            "digest_unsigned_B": d_uns, "digest_signed_B": d_sig,
            "ctrl_overhead_unsigned_pct": 100 * msgs * d_uns / gcb,
            "ctrl_overhead_signed_pct": 100 * msgs * d_sig / gcb,
            "mode_quorum_latency": vd.rounds_to_network_quorum(N // 2 + 1),
            "peer_failure_latency": peer_failure_latency(N),
        })

hdr = ["N", "GL_msgs", "GL_comm_MB", "digest_unsigned_B", "digest_signed_B",
       "ctrl_overhead_unsigned_pct", "ctrl_overhead_signed_pct",
       "mode_quorum_latency", "peer_failure_latency"]
print(f"{'N':>4}{'GLcomm(MB)':>12}{'unsigned%':>11}{'signed%':>10}{'modeQ':>7}{'peerFail':>9}")
for r in rows:
    print(f"{r['N']:>4}{r['GL_comm_MB']:>12.1f}{r['ctrl_overhead_unsigned_pct']:>11.3f}"
          f"{r['ctrl_overhead_signed_pct']:>10.3f}{str(r['mode_quorum_latency']):>7}"
          f"{str(r['peer_failure_latency']):>9}")
os.makedirs("results", exist_ok=True)
with open("results/vote_comm.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in hdr})

Ns = [r["N"] for r in rows]
fig, ax1 = plt.subplots(figsize=(8.8, 5))
ax1.plot(Ns, [r["ctrl_overhead_signed_pct"] for r in rows], "-o", color="#b2182b",
         lw=2.2, ms=7, label="comm overhead, signed (+agg sig)")
ax1.plot(Ns, [r["ctrl_overhead_unsigned_pct"] for r in rows], "--s", color="#e08214",
         lw=2, ms=7, label="comm overhead, unsigned (2 bitmaps)")
ax1.set_xlabel("Number of peers ($N$)")
ax1.set_ylabel("control-plane comm overhead (% of GL traffic)")
ax1.set_ylim(0, max(1.0, max(r["ctrl_overhead_signed_pct"] for r in rows) * 1.3))
ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(Ns, [r["mode_quorum_latency"] for r in rows], "-^", color="#1b7837",
         lw=2.2, ms=7, label="mode-quorum latency (ring)")
ax2.plot(Ns, [r["peer_failure_latency"] for r in rows], ":D", color="#2166ac",
         lw=2.2, ms=7, label="peer-failure latency (ring, T=%d)" % PF_TIMEOUT)
ax2.set_ylabel("decentralized latency (rounds)")
ax1.set_title("Control plane over GLow (mode votes + peer suspicions):\n"
              "tiny comm overhead; cost is latency ~$N$")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8.5)
fig.tight_layout()
fig.savefig("results/vote_comm.png", dpi=170); fig.savefig("results/vote_comm.pdf")
print("\n[ctrl-comm] -> results/vote_comm.{csv,png,pdf}")
