#!/usr/bin/env python3
"""Self-test of local peer-to-peer failure detection over GLow (fd/peer_failure.py).

Covers: (1) min_witnesses=1 (crash-trusted) detection + diffusion; (2) min_witnesses=2
on a ring (BOTH neighbours must independently time out j) -> more robust, slightly
later; (3) a single FALSE suspicion (one flaky/lying neighbour, j alive) is REJECTED
under min_witnesses=2 (resists a single bad observer); (4) the agreement quorum
(>= q nodes accept j down) IS reachable via diffusion -- the neighbours diffuse and
quorum forms, even though only degree(j) nodes observe j directly.
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

N, T, DOWN, FAIL_AT, R = 10, 2, 5, 3, 80
q = N // 2 + 1

def run(min_w):
    det = PeerFailureDetector(build_topology("ring", N), timeout=T, min_witnesses=min_w)
    net_r = agree_r = None
    for r in range(1, R + 1):
        alive = set(range(N)) if r < FAIL_AT else set(range(N)) - {DOWN}
        det.observe_and_step(r, alive)
        if net_r is None and det.network_knows(DOWN, alive):
            net_r = r
        if agree_r is None and det.agreement_quorum(DOWN, alive, q):
            agree_r = r
    return det, net_r, agree_r

print(f"ring N={N}, node {DOWN} fails @round {FAIL_AT}, T={T}, q={q}, neighbours={build_topology('ring',N).neighbors(DOWN)}")

# (1) min_witnesses = 1
_, net1, agree1 = run(1)
print(f"[min_w=1] network_knows @{net1}  agreement_quorum(>= q) @{agree1}")
check(net1 is not None, "min_w=1: network learns j is down")
check(agree1 is not None, "min_w=1: agreement quorum (>= q nodes accept) IS reached via diffusion")

# (2) min_witnesses = 2 (both neighbours needed) -> more conservative
_, net2, agree2 = run(2)
print(f"[min_w=2] network_knows @{net2}  agreement_quorum(>= q) @{agree2}")
check(net2 is not None, "min_w=2: still reaches network agreement (both neighbours detect a real crash)")
check(net2 >= net1, "min_w=2 is >= as late as min_w=1 (more conservative)")

# (3) single FALSE suspicion (j alive) rejected under min_w=2
det = PeerFailureDetector(build_topology("ring", N), timeout=T, min_witnesses=2)
flaky = build_topology("ring", N).neighbors(DOWN)[0]      # one neighbour lies
det.heard[flaky].setdefault(DOWN, set()).add(flaky)        # inject a single false suspicion
for r in range(1, R + 1):
    det.observe_and_step(r, set(range(N)))                  # j is ALIVE the whole time
knows_any = any(det.knows_down(i, DOWN) for i in range(N))
check(not knows_any, "min_w=2: a single false suspicion (1 observer) is REJECTED (resists one flaky/lying node)")

# (4) explicit: agreement quorum reachable though only degree(j) observe directly
det, _, agr = run(1)
print(f"[agreement] only degree(j)={det.direct_observers(DOWN)} observe j directly, "
      f"yet >= q={q} nodes agree by diffusion @round {agr}")
check(agr is not None, "agreement quorum reachable via neighbour diffusion (sparse graph OK for crash)")

print("\n" + ("PEER FAILURE SELFTEST OK" if ok else "PEER FAILURE SELFTEST FAILED"))
sys.exit(0 if ok else 1)
