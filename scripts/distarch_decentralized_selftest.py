#!/usr/bin/env python3
"""Self-test of the DECENTRALIZED detection path in DistributedArchManager (Gap 1+2).

Unlike the central-oracle path (distarch_selftest.py), the decisions here are
decentralized. The two directions are ASYMMETRIC by design:
  * FL->GL is FAIL-FAST: commit as soon as a down peer is locally corroborated by
    min_witnesses observers (~T rounds), NO network consensus on the way down;
  * GL->FL is CAREFUL: per-node local dwell + a diffused recovery vote to quorum.
This test drives the manager over a fault schedule and checks:

  (1) FL->GL fires ~T rounds after the failure (fail-fast), not instantly and not
      after a network quorum;
  (2) the fail-fast latency is strictly BELOW the network-consensus latency
      (contrast with the peer_failure model's quorum/net-know times);
  (3) the control-plane digest (Gap 2) is accounted per round (ctrl_bytes grows,
      digest_hex non-empty) and stays tiny;
  (4) recovery (GL->FL) still respects dwell + full participation.

Rodar: ~/venv-ereno314/bin/python scripts/distarch_decentralized_selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.arch_manager import DistributedArchManager
from fd.peer_failure import PeerFailureDetector
from fd.topology import build_topology

ok = True
def check(cond, msg):
    global ok
    print(("  OK  " if cond else " FAIL ") + msg); ok = ok and cond

N, T, DOWN, FAIL_AT, ROUNDS = 14, 2, 7, 10, 40


def model_bounds(n, t, down, fail_at, q):
    """Standalone peer_failure latencies with IDENTICAL failure timing:
      * quorum_know : rounds until >= q nodes KNOW j is down (agreement quorum);
      * net_know    : rounds until EVERY up node knows (full agreement).
    The live VOTE-quorum commit must sit in [quorum_know, net_know]: it needs a
    quorum of knowers AND their votes to converge at one node."""
    det = PeerFailureDetector(build_topology("ring", n), timeout=t)
    quorum_know = net_know = None
    for r in range(1, 8 * n):
        alive = set(range(n)) if r < fail_at else set(range(n)) - {down}
        det.observe_and_step(r, alive)
        if quorum_know is None and det.agreement_quorum(down, alive, q):
            quorum_know = r - fail_at
        if net_know is None and det.network_knows(down, alive):
            net_know = r - fail_at
        if net_know is not None:
            break
    return quorum_know, net_know


def run(recover_at=None):
    """Drive the manager: node DOWN fails @FAIL_AT; optionally recovers @recover_at."""
    mgr = DistributedArchManager(
        n_nodes=N, initial_mode="federated", dwell_rounds=3, cooldown_rounds=2,
        topology=build_topology("ring", N), peer_timeout=T, min_witnesses=1)
    timeline, digest_seen = [], False
    for r in range(1, ROUNDS + 1):
        alive = set(range(N))
        if r >= FAIL_AT and (recover_at is None or r < recover_at):
            alive -= {DOWN}
        mgr.observe(r, alive)                    # aggregate_fit feeds reported=alive
        timeline.append(mgr.get_mode(r + 1))     # committed mode for the NEXT round
        if mgr.digest_hex():
            digest_seen = True
    return mgr, timeline, digest_seen


print(f"ring N={N}, node {DOWN} fails @round {FAIL_AT}, T={T}, "
      f"neighbours={build_topology('ring', N).neighbors(DOWN)}")

# (1)+(2) FL->GL is FAIL-FAST: fires ~T rounds after the failure (local
# corroboration by min_witnesses), NOT after a network quorum diffusion.
mgr, timeline, digest_seen = run()
switch_round = next((i + 1 for i, m in enumerate(timeline) if m == "gossip"), None)
q = (N - 1) // 2 + 1                              # quorum among the 13 active nodes
Lq, Lnet = model_bounds(N, T, DOWN, FAIL_AT, q)  # network-consensus latencies (for contrast)
Llive = None if switch_round is None else switch_round - FAIL_AT
print(f"[fail-fast] switch committed for round {switch_round}  live latency={Llive}  "
      f"(vs network-quorum_know={Lq}, net_know={Lnet} — fail-fast must beat these)")
# determinism: a second identical run gives the same endogenous latency
_, timeline2, _ = run()
switch2 = next((i + 1 for i, m in enumerate(timeline2) if m == "gossip"), None)
check(switch_round is not None, "FL->GL commit does happen")
check(Llive is not None and 1 <= Llive <= T + 2,
      f"FL->GL is fail-fast: fires ~T={T} rounds after failure (latency {Llive}), not instant")
check(switch2 == switch_round, "endogenous latency is deterministic (reproducible)")
check(Llive is not None and Lq is not None and Llive < Lq,
      f"fail-fast latency ({Llive}) < network-quorum latency ({Lq}) — no consensus on the way down")

# (3) control-plane digest accounted (Gap 2), tiny
per_round = mgr.ctrl_bytes / ROUNDS
print(f"[digest] ctrl_bytes={mgr.ctrl_bytes} over {ROUNDS} rounds "
      f"(~{per_round:.1f} B/round)  digest_size={mgr._digest_size()} B/msg")
check(mgr.ctrl_bytes > 0, "control-plane bytes accounted per round (Gap 2)")
check(digest_seen, "digest_hex non-empty during the run (rides real GLow messages)")
check(mgr._digest_size() == 2 * ((N + 7) // 8), "digest = 2 bitmaps (vote+suspicion)")

# (4) recovery GL->FL after node returns + dwell
mgr2, tl2, _ = run(recover_at=FAIL_AT + (Llive or 5) + 2)
gl_start = next((i for i, m in enumerate(tl2) if m == "gossip"), None)   # enter GL
rec = None
if gl_start is not None:
    for i in range(gl_start + 1, len(tl2)):        # first FL after the GL phase
        if tl2[i] == "federated":
            rec = i + 1; break
print(f"[recovery] entered GL for round {None if gl_start is None else gl_start + 1}, "
      f"GL->FL recovery committed for round {rec}  timeline tail={tl2[-8:]}")
check(gl_start is not None, "recovery run does enter GL first")
check(rec is not None and rec > (gl_start + 1),
      "GL->FL recovery commit happens after node returns + dwell")

# (5) NO FALSE RECOVERY (flap): with several nodes down (that partition the ring)
# and NEVER returning, recovery must NOT fire — unanimity of local health is
# unreachable while any node is missing, so the mode stays gossip.
def run_cascade(nrounds=60):
    mgr = DistributedArchManager(
        n_nodes=N, initial_mode="federated", dwell_rounds=3, cooldown_rounds=2,
        topology=build_topology("ring", N), peer_timeout=T, min_witnesses=1)
    modes = []
    for r in range(1, nrounds + 1):
        alive = set(range(N)) - {d for d, fr in ((3, 10), (7, 16), (11, 22)) if r >= fr}
        mgr.observe(r, alive)
        modes.append(mgr.get_mode(r + 1))
    return modes
casc = run_cascade()
casc_switch = next((i + 1 for i, m in enumerate(casc) if m == "gossip"), None)
false_recovery = casc_switch is not None and any(m == "federated" for m in casc[casc_switch:])
print(f"[no-flap] cascade FL->GL @{casc_switch}; any false GL->FL afterwards? {false_recovery}")
check(casc_switch is not None, "cascade enters GL (fail-fast)")
check(not false_recovery,
      "NO false recovery while nodes stay down (unanimity blocks partial membership)")

print("\n" + ("DISTARCH DECENTRALIZED SELFTEST OK" if ok else
              "DISTARCH DECENTRALIZED SELFTEST FAILED"))
sys.exit(0 if ok else 1)
