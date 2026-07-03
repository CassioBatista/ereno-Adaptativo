"""Topology — loads and represents the agent graph for Gossip Learning.

Supported topology types (conf/topologies/*.yaml):
  star    — hub-and-spoke; equivalent to classic federated (default)
  ring    — each node connected to its two neighbours
  chain   — linear sequence, first and last have one neighbour only
  graph   — arbitrary adjacency list from YAML
"""

from __future__ import annotations

import yaml


# ── Data class ────────────────────────────────────────────────────────────────

class Topology:
    """Graph representation of the agent network.

    Attributes
    ----------
    name        : topology identifier (e.g. 'ring', 'star')
    num_nodes   : total number of agents / Flower supernodes
    _adjacency  : dict[node_id -> list[neighbor_ids]]
    _status     : dict[node_id -> bool]  (True = up)
    """

    def __init__(self, name: str, adjacency: dict[int, list[int]]) -> None:
        self.name       = name
        self._adjacency = adjacency
        self._status    = {n: True for n in adjacency}

    @property
    def num_nodes(self) -> int:
        return len(self._adjacency)

    def neighbors(self, node_id: int) -> list[int]:
        return self._adjacency.get(node_id, [])

    def up_neighbors(self, node_id: int) -> list[int]:
        return [n for n in self.neighbors(node_id) if self._status.get(n, False)]

    def is_up(self, node_id: int) -> bool:
        return self._status.get(node_id, False)

    def set_status(self, node_id: int, up: bool) -> None:
        self._status[node_id] = up

    def all_nodes(self) -> list[int]:
        return list(self._adjacency.keys())

    def __repr__(self) -> str:
        return f"Topology(name={self.name!r}, num_nodes={self.num_nodes})"


# ── Builders ──────────────────────────────────────────────────────────────────

def _build_star(n: int) -> dict[int, list[int]]:
    """Node 0 is the hub; all others connect only to 0."""
    adj: dict[int, list[int]] = {0: list(range(1, n))}
    for i in range(1, n):
        adj[i] = [0]
    return adj


def _build_ring(n: int) -> dict[int, list[int]]:
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}


def _build_chain(n: int) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {}
    for i in range(n):
        neighbours = []
        if i > 0:
            neighbours.append(i - 1)
        if i < n - 1:
            neighbours.append(i + 1)
        adj[i] = neighbours
    return adj


def _build_from_adjacency(raw: dict) -> dict[int, list[int]]:
    """Parse arbitrary adjacency dict from YAML (keys may be strings)."""
    return {int(k): [int(v) for v in vs] for k, vs in raw.items()}


# ── Public API ─────────────────────────────────────────────────────────────────

def build_topology(topology_type: str, num_nodes: int) -> Topology:
    """Create a topology programmatically without a YAML file.

    Parameters
    ----------
    topology_type : 'star' | 'ring' | 'chain'
    num_nodes     : number of agents
    """
    match topology_type:
        case "star":
            adj = _build_star(num_nodes)
        case "ring":
            adj = _build_ring(num_nodes)
        case "chain":
            adj = _build_chain(num_nodes)
        case _:
            raise ValueError(
                f"Tipo de topologia desconhecido: '{topology_type}'. "
                f"Use 'star', 'ring' ou 'chain', ou carregue via YAML para topologias arbitrárias."
            )
    return Topology(name=topology_type, adjacency=adj)


def load_topology(yaml_path: str) -> Topology:
    """Load topology from a YAML file.

    YAML format:
        name: ring
        nodes: 5
        # optional — if absent, topology_type + nodes are used to auto-build
        adjacency:
          0: [1, 4]
          1: [0, 2]
          2: [1, 3]
          3: [2, 4]
          4: [3, 0]
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    name      = data.get("name", "custom")
    num_nodes = int(data.get("nodes", 0))

    if "adjacency" in data:
        adj = _build_from_adjacency(data["adjacency"])
    elif num_nodes > 0:
        adj = _build_star(num_nodes) if name == "star" \
            else _build_ring(num_nodes) if name == "ring" \
            else _build_chain(num_nodes) if name == "chain" \
            else _build_ring(num_nodes)   # fallback
    else:
        raise ValueError(
            f"YAML de topologia '{yaml_path}' deve conter 'adjacency' ou 'nodes'."
        )

    return Topology(name=name, adjacency=adj)


def resolve_topology(mode: str, topology_arg: str | None, num_clients: int) -> Topology:
    """Entry point used by main_dist.py.

    For 'federated': always returns a star topology (num_clients nodes).
    For 'gossip':    loads from YAML path or builds by type name.
    """
    if mode == "federated":
        return build_topology("star", num_clients)

    if topology_arg is None:
        raise ValueError("Para modo 'gossip', informe a topologia via --topology.")

    # topology_arg can be a YAML path or a type name (ring, chain, star)
    if topology_arg.endswith(".yaml") or "/" in topology_arg:
        return load_topology(topology_arg)

    return build_topology(topology_arg, num_clients)
