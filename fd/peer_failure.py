"""Local peer-to-peer failure detection over the GLow gossip substrate.

Each node locally times out its own up-neighbours: a neighbour that is silent for
`timeout` consecutive gossip rounds is *locally suspected* (a crash is objectively
observable by whoever exchanges with the node). Suspicions then ride the SAME GLow
head-election neighbourhood-union diffusion as boosters and votes (see
fd/vote_diffusion.py), so every node learns of the failure.

Design honesty — quorum-of-observers is topology-limited:
  Only a node's *neighbours* can directly time it out. In a sparse overlay (ring,
  degree 2) at most `degree(j)` nodes ever DIRECTLY observe j, which is far below a
  majority quorum. So for a CRASH (non-Byzantine, v2) the suspicion is TRUSTED and
  propagated (min_witnesses=1 suffices). A Byzantine FALSE accusation of a peer
  cannot be resisted by "q independent observers" in a sparse graph -> needs a
  different corroboration scheme (accused-node defence / witnesses / reputation),
  deferred to v3 together with valued/weighted votes.
"""
from __future__ import annotations

from fd.topology import Topology


class PeerFailureDetector:
    def __init__(self, topology: Topology, timeout: int = 2, min_witnesses: int = 1) -> None:
        self.topo = topology
        self.T = max(1, int(timeout))
        self.min_w = max(1, int(min_witnesses))          # 1 = crash-trusted (v2)
        # consecutive missed exchanges: observer i -> {neighbour j: misses}
        self.missed: dict[int, dict[int, int]] = {i: {} for i in topology.all_nodes()}
        # per-target suspicion sets diffused via GLow: i -> {target j: {suspectors}}
        self.heard: dict[int, dict[int, set[int]]] = {i: {} for i in topology.all_nodes()}

    def direct_observers(self, j: int) -> int:
        """How many nodes can EVER directly time out j (its degree)."""
        return len(self.topo.neighbors(j))

    def observe_and_step(self, server_round: int, alive: set[int]) -> None:
        """One GLow round: (1) local per-neighbour timeout, (2) diffuse suspicions."""
        # (1) local detection — each up node times out its silent up-neighbours
        for i in self.topo.all_nodes():
            if i not in alive:
                continue
            for j in self.topo.neighbors(i):
                if j not in alive:
                    self.missed[i][j] = self.missed[i].get(j, 0) + 1
                    if self.missed[i][j] >= self.T:               # i now suspects j
                        self.heard[i].setdefault(j, set()).add(i)
                else:
                    self.missed[i][j] = 0

        # (2) diffuse suspicions via GLow head-election neighbourhood union
        active = [n for n in self.topo.all_nodes() if n in alive]
        if not active:
            return
        head = active[(server_round - 1) % len(active)]
        for nb in self.topo.up_neighbors(head):
            for j, suspectors in self.heard[nb].items():
                self.heard[head].setdefault(j, set()).update(suspectors)

    def knows_down(self, node_i: int, target_j: int) -> bool:
        """Node i considers j down once it has heard >= min_witnesses suspicions."""
        return len(self.heard.get(node_i, {}).get(target_j, ())) >= self.min_w

    def network_knows(self, target_j: int, alive: set[int]) -> bool:
        """Every up node (except j) considers j down."""
        others = [i for i in self.topo.all_nodes() if i in alive and i != target_j]
        return bool(others) and all(self.knows_down(i, target_j) for i in others)
