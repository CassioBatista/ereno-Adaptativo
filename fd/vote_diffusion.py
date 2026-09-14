"""Vote / suspicion dissemination reusing the GLow gossip substrate.

Control-plane votes ride the SAME head-election neighbourhood-union diffusion that
GLow uses to spread boosters (see fd/strategy/glow_strategy.py): each round the
elected head unions its up-neighbours' vote sets into its own; round-robin over
active nodes. A node reaches a decision LOCALLY when its accumulated vote set holds
>= q distinct votes -- a *decentralized* quorum, no central tally. The vote is a
small digest piggybacked on the GLow gossip message (see comm_bytes_per_round).

This makes the quorum's cost explicit and honest:
  * communication: a few bytes per existing GLow gossip message (piggyback);
  * latency: votes take ~diffusion time (~N rounds on a ring) to reach every node,
    unlike the instantaneous central tally used by DistributedArchManager today.
"""
from __future__ import annotations

import math

from fd.topology import Topology


class VoteDiffusion:
    """Diffuses per-node votes over a topology with GLow head-election semantics."""

    def __init__(self, topology: Topology) -> None:
        self.topo = topology
        # node -> set of distinct voter ids it has heard (the "vote set")
        self.heard: dict[int, set[int]] = {n: set() for n in topology.all_nodes()}
        self._msgs = 0  # cumulative gossip exchanges that carried the vote digest

    def cast(self, voter: int) -> None:
        """A node casts its own vote (immediately known to itself)."""
        if voter in self.heard:
            self.heard[voter].add(voter)

    def step(self, server_round: int) -> None:
        """One GLow round: the elected head unions its up-neighbours' vote sets."""
        active = [n for n in self.topo.all_nodes() if self.topo.is_up(n)]
        if not active:
            return
        head = active[(server_round - 1) % len(active)]
        nbrs = self.topo.up_neighbors(head)
        merged = set(self.heard[head])
        for nb in nbrs:
            merged |= self.heard[nb]
        self.heard[head] = merged
        self._msgs += len(nbrs)  # head <- each neighbour carries a digest

    def count(self, node: int) -> int:
        return len(self.heard.get(node, ()))

    def committed(self, node: int, q: int) -> bool:
        return self.count(node) >= q

    def all_committed(self, q: int) -> bool:
        active = [n for n in self.topo.all_nodes() if self.topo.is_up(n)]
        return bool(active) and all(self.committed(n, q) for n in active)

    def rounds_to_network_quorum(self, q: int, start_round: int = 1,
                                 max_rounds: int = 1000) -> int | None:
        """Rounds (from start_round) until EVERY active node locally has >= q votes.
        Assumes all active nodes have cast. Returns None if not reached."""
        r = start_round
        while r < start_round + max_rounds:
            if self.all_committed(q):
                return r - start_round
            self.step(r)
            r += 1
        return None

    # ── communication model (piggyback digest) ─────────────────────────────────

    @staticmethod
    def digest_bytes(n_nodes: int, signed: bool, agg_sig_bytes: int = 96) -> int:
        """Size of the vote digest carried per gossip message.
        unsigned: a who-voted bitmap of ceil(N/8) bytes.
        signed:   bitmap + one aggregate signature (e.g. BLS ~96 B)."""
        bmp = math.ceil(n_nodes / 8)
        return bmp + (agg_sig_bytes if signed else 0)

    def messages(self) -> int:
        """Cumulative gossip exchanges that carried a vote digest."""
        return self._msgs
