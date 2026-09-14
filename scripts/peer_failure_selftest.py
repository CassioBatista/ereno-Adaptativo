#!/usr/bin/env python3
"""Self-test of local peer-to-peer failure detection over GLow (fd/peer_failure.py).
Verifies: (1) a node's up-neighbours locally time it out after `timeout` silent
rounds; (2) the suspicion diffuses via GLow so the whole network learns; (3) only
degree(j) nodes DIRECTLY observe j (sparse-graph limit on observer-quorum).
Rodar: ~/venv-ereno314/bin/python scripts/peer_failure_selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.topology import build_topology
from fd.peer_failure import PeerFailureDetector

ok = True
def check(cond, msg):
    global ok
    print(("  OK  " if cond else " FAIL ") + msg); ok = ok and cond

N, T, DOWN, FAIL_AT, ROUNDS = 10, 2, 5, 3, 60
topo = build_topology("ring", N)
det = PeerFailureDetector(topo, timeout=T)
neigh = topo.neighbors(DOWN)
print(f"ring N={N}, node {DOWN} fails at round {FAIL_AT}, timeout T={T}; neighbours of {DOWN} = {neigh}")

local_r = None; network_r = None
for r in range(1, ROUNDS + 1):
    alive = set(range(N)) if r < FAIL_AT else set(range(N)) - {DOWN}
    det.observe_and_step(r, alive)
    if local_r is None and any(det.knows_down(i, DOWN) for i in neigh):
        local_r = r
    if network_r is None and det.network_knows(DOWN, alive):
        network_r = r

print(f"local suspicion by a neighbour @round {local_r}  (failure @ {FAIL_AT}, so latency {local_r-FAIL_AT} = T-ish)")
print(f"network-wide agreement @round {network_r}  (total latency {network_r-FAIL_AT} rounds)")

check(local_r is not None and (local_r - FAIL_AT) >= T - 1, "neighbour times out after ~T silent rounds")
check(network_r is not None, "suspicion diffuses to the whole network (all nodes learn j is down)")
check(network_r >= local_r, "network agreement comes after (or with) local detection")
check(det.direct_observers(DOWN) == len(neigh) == 2,
      f"only degree(j)={det.direct_observers(DOWN)} nodes DIRECTLY observe j (< quorum) -> observer-quorum infeasible in sparse graph")

# a node NOT neighbour of DOWN only learns via diffusion, never directly
far = next(i for i in range(N) if i not in neigh and i != DOWN)
det2 = PeerFailureDetector(build_topology("ring", N), timeout=T)
for r in range(1, T + 2):
    det2.observe_and_step(r, set(range(N)) - {DOWN})
check(not det2.knows_down(far, DOWN),
      f"far node {far} does NOT know j down before diffusion (only neighbours detect directly)")

print("\n" + ("PEER FAILURE SELFTEST OK" if ok else "PEER FAILURE SELFTEST FAILED"))
sys.exit(0 if ok else 1)
