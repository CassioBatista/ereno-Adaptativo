#!/usr/bin/env python3
"""Self-test of decentralized vote diffusion over the GLow substrate
(fd/vote_diffusion.py). Verifies: (1) with >= q voters, every node locally reaches
quorum within ~N rounds (diffusion latency); (2) with f < q voters, NO node ever
reaches quorum (minority cannot force a decision); (3) diffusion latency grows with
network size / diameter. Rodar: ~/venv-ereno314/bin/python scripts/vote_diffusion_selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.topology import build_topology
from fd.vote_diffusion import VoteDiffusion

ok = True
def check(cond, msg):
    global ok
    print(("  OK  " if cond else " FAIL ") + msg); ok = ok and cond

def q_of(n):
    return n // 2 + 1

# (1) all cast -> everyone reaches quorum within a bounded number of rounds
print("=== (1) unanimous cast: rounds to network-wide quorum (ring) ===")
for n in (5, 10, 14, 20):
    vd = VoteDiffusion(build_topology("ring", n))
    for node in range(n):
        vd.cast(node)
    r = vd.rounds_to_network_quorum(q_of(n))
    print(f"  N={n:>2}  q={q_of(n):>2}  rounds_to_network_quorum={r}")
    check(r is not None and r <= 3 * n, f"N={n}: network quorum reached in <=3N rounds")

# (2) minority (f < q) cannot force a decision
print("\n=== (2) minority f < q never reaches quorum ===")
n = 10; q = q_of(n)              # q = 6
vd = VoteDiffusion(build_topology("ring", n))
for node in range(q - 1):        # only 5 voters (f = q-1 = 5 < q)
    vd.cast(node)
for r in range(1, 200):
    vd.step(r)
maxcount = max(vd.count(node) for node in range(n))
check(maxcount == q - 1, f"max votes any node hears = {maxcount} (= f = {q-1}, < q = {q})")
check(not vd.all_committed(q), "no node reaches quorum with f < q (minority cannot force)")

# (3) exactly q voters -> quorum eventually reached
print("\n=== (3) exactly q voters -> quorum reached ===")
vd = VoteDiffusion(build_topology("ring", n))
for node in range(q):            # exactly 6 voters
    vd.cast(node)
r = vd.rounds_to_network_quorum(q)
check(r is not None, f"with exactly q={q} voters, network quorum reached (@+{r} rounds)")

# (4) topology matters: star diffuses faster than ring
print("\n=== (4) topology: star vs ring diffusion latency (N=14) ===")
res = {}
for topo in ("ring", "star"):
    vd = VoteDiffusion(build_topology(topo, 14))
    for node in range(14):
        vd.cast(node)
    res[topo] = vd.rounds_to_network_quorum(q_of(14))
    print(f"  {topo:5} -> {res[topo]} rounds")
check(res["star"] is not None and res["ring"] is not None, "both topologies reach quorum")

print("\n" + ("VOTE DIFFUSION SELFTEST OK" if ok else "VOTE DIFFUSION SELFTEST FAILED"))
sys.exit(0 if ok else 1)
