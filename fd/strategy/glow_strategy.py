"""GlowStrategy — Gossip Learning strategy for Flower 1.31.

Each round one node is elected as HEAD (round-robin over active nodes).
The head receives its neighbours' current models, aggregates them,
and returns an updated model. Non-head nodes validate for the head.

Aggregation modes:
  inplace  — weighted average of neighbour models (weight = num_samples)
  score    — weight proportional to neighbour's reported local accuracy
  xgb      — merge XGBoost boosters by concatenating trees (bagging)
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from flwr.common import (
    Parameters, Scalar, NDArrays,
    ndarrays_to_parameters, parameters_to_ndarrays,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from fd.topology import Topology


def resolve_node_index(
    cid:       str,
    all_cids:  list[str],
    cid_map:   dict[str, int],
    num_nodes: int,
) -> int:
    """Map a server-side proxy cid to a logical node/partition index 0..N-1.

    Under flwr's simulation the proxy cids are opaque node ids, NOT the
    partition ids. Resolution order:
      1. cid_map — learned from FitRes/EvaluateRes metrics['cid'] reported
         by the clients (the true partition id);
      2. small digit cids ("0".."N-1") — in-process tests;
      3. stable sort of all known cids — deterministic fallback before any
         results have been seen (consistent within a run, but the index is
         not guaranteed to equal the data partition id).
    """
    if cid in cid_map:
        return cid_map[cid]
    if cid.isdigit() and int(cid) < num_nodes:
        return int(cid)
    ordered = sorted(all_cids, key=lambda c: (0, int(c)) if c.isdigit() else (1, c))
    return ordered.index(cid) % num_nodes


def learn_cid_map(cid_map: dict[str, int], results) -> None:
    """Update cid_map from client-reported metrics['cid'] (true partition id)."""
    for proxy, res in results:
        metrics = getattr(res, "metrics", None)
        if metrics and "cid" in metrics:
            cid_map[proxy.cid] = int(metrics["cid"])


def dedup_union(params_list: list[Parameters | None]) -> Parameters:
    """União de conjuntos de boosters com deduplicação por conteúdo.

    Cada Parameters pode carregar 1+ tensores (boosters serializados); a
    união preserva a ordem de chegada e descarta cópias byte a byte —
    re-treinos determinísticos (mesmos dados/seed) deduplicam naturalmente,
    limitando o pool de cada nó ao nº de nós da rede.
    """
    seen: set[bytes] = set()
    arrays: list[np.ndarray] = []
    for p in params_list:
        if p is None:
            continue
        for a in parameters_to_ndarrays(p):
            if a.size == 0:
                continue
            key = a.tobytes()
            if key not in seen:
                seen.add(key)
                arrays.append(a)
    return ndarrays_to_parameters(arrays)


class GlowStrategy(Strategy):
    """Gossip Learning strategy with round-robin head election.

    Parameters
    ----------
    topology        : Topology object with adjacency information
    aggregation     : 'inplace' (weighted avg) | 'score' (accuracy-weighted) | 'xgb' (booster merge)
    initial_parameters : starting Parameters for all nodes
    """

    def __init__(
        self,
        topology:            Topology,
        aggregation:         str = "inplace",
        initial_parameters:  Parameters | None = None,
    ) -> None:
        self.topology           = topology
        self.aggregation        = aggregation
        self._initial_parameters = initial_parameters

        # per-node model state (Parameters keyed by node index 0..N-1)
        self.pool_parameters: dict[int, Parameters] = {}
        self._last_accuracies: dict[int, float]     = {}
        # proxy cid -> partition/node index, learned from client metrics
        # (may be replaced by a shared dict from HybridStrategy)
        self.cid_map: dict[str, int] = {}

    def _node_index(self, cid: str, all_cids: list[str]) -> int:
        return resolve_node_index(cid, all_cids, self.cid_map, self.topology.num_nodes)

    # ── head election ─────────────────────────────────────────────────────────

    def _head_for_round(self, server_round: int) -> int:
        active = [n for n in self.topology.all_nodes() if self.topology.is_up(n)]
        if not active:
            raise RuntimeError("GlowStrategy: nenhum nó ativo na topologia.")
        return active[(server_round - 1) % len(active)]

    # ── Strategy interface ────────────────────────────────────────────────────

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self._initial_parameters

    def configure_fit(
        self,
        server_round:   int,
        parameters:     Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        head_cid = self._head_for_round(server_round)
        neighbours = self.topology.up_neighbors(head_cid)

        print(f"[GLow] round={server_round}  head={head_cid}  neighbours={neighbours}")

        all_clients = list(client_manager.all().values())
        all_cids    = [p.cid for p in all_clients]
        instructions = []

        for proxy in all_clients:
            cid = self._node_index(proxy.cid, all_cids)

            # Send each node its own current model (or global if not yet initialised)
            node_params = self.pool_parameters.get(cid, parameters)

            config = {
                "head_cid":    str(head_cid),
                "neighbours":  ",".join(str(n) for n in neighbours),
                "round":       str(server_round),
                "aggregation": self.aggregation,
            }
            instructions.append((proxy, FitIns(node_params, config)))

        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results:      list[tuple[ClientProxy, FitRes]],
        failures:     list[tuple[ClientProxy, Exception]],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        head_cid = self._head_for_round(server_round)

        # Learn the proxy->partition mapping from client-reported metrics,
        # then collect updated models from all nodes
        learn_cid_map(self.cid_map, results)
        all_cids = [proxy.cid for proxy, _ in results]
        for proxy, fit_res in results:
            cid = self._node_index(proxy.cid, all_cids)
            if self.aggregation == "xgb_union":
                # gossip COM MEMÓRIA: o nó acumula o que já sabe (recebido
                # em rounds anteriores) + o re-treino local — sem isso o
                # pool é sobrescrito e o conhecimento nunca difunde
                self.pool_parameters[cid] = dedup_union(
                    [self.pool_parameters.get(cid), fit_res.parameters])
            else:
                self.pool_parameters[cid] = fit_res.parameters
            acc = fit_res.metrics.get("accuracy", 0.0)
            self._last_accuracies[cid] = float(acc)

        # Aggregate neighbour models into the head's model
        neighbours = self.topology.up_neighbors(head_cid)
        participant_ids = [head_cid] + neighbours
        participant_params = [
            self.pool_parameters[i]
            for i in participant_ids
            if i in self.pool_parameters
        ]

        if not participant_params:
            return self.pool_parameters.get(head_cid), {}

        aggregated = self._aggregate(participant_params, participant_ids)

        # Head gets the aggregated model; others keep their own
        self.pool_parameters[head_cid] = aggregated

        metrics = {
            "head_cid": float(head_cid),
            "head_accuracy": self._last_accuracies.get(head_cid, 0.0),
        }
        return aggregated, metrics

    def configure_evaluate(
        self,
        server_round:   int,
        parameters:     Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        # Evaluate all nodes with their own local models
        all_clients = list(client_manager.all().values())
        all_cids    = [p.cid for p in all_clients]
        instructions = []
        for proxy in all_clients:
            cid = self._node_index(proxy.cid, all_cids)
            node_params = self.pool_parameters.get(cid, parameters)
            instructions.append((proxy, EvaluateIns(node_params, {"round": str(server_round)})))
        return instructions

    def aggregate_evaluate(
        self,
        server_round: int,
        results:      list[tuple[ClientProxy, EvaluateRes]],
        failures:     list[tuple[ClientProxy, Exception]],
    ) -> tuple[float | None, dict[str, Scalar]]:
        if not results:
            return None, {}
        total_loss    = sum(r.loss * r.num_examples for _, r in results)
        total_samples = sum(r.num_examples for _, r in results)
        avg_loss = total_loss / total_samples if total_samples else 0.0
        return avg_loss, {"round": float(server_round)}

    def evaluate(self, server_round: int, parameters: Parameters):
        return None

    # ── aggregation helpers ───────────────────────────────────────────────────

    def _aggregate(
        self,
        params_list: list[Parameters],
        node_ids:    list[int],
    ) -> Parameters:
        if self.aggregation == "xgb_union":
            # fusão OR: o agregado é o CONJUNTO (deduplicado) dos boosters
            # acumulados pela vizinhança — a união de decisões ocorre na
            # predição. É este agregado que vira o pool do head, fazendo o
            # conhecimento difundir de vizinhança em vizinhança.
            union = dedup_union(params_list)
            print(f"[GLow] união do head: {len(union.tensors)} boosters")
            return union

        if self.aggregation == "xgb":
            return _aggregate_xgb(params_list)

        arrays_list: list[NDArrays] = [parameters_to_ndarrays(p) for p in params_list]

        if self.aggregation == "score":
            weights = [self._last_accuracies.get(nid, 1.0) for nid in node_ids]
        else:
            weights = [1.0] * len(arrays_list)

        total = sum(weights)
        weights = [w / total for w in weights]

        aggregated: NDArrays = [
            sum(w * a[i] for w, a in zip(weights, arrays_list))
            for i in range(len(arrays_list[0]))
        ]
        return ndarrays_to_parameters(aggregated)

    # ── accessors ─────────────────────────────────────────────────────────────

    def get_node_model(self, node_id: int) -> Parameters | None:
        return self.pool_parameters.get(node_id)


# ── XGBoost aggregation helpers ───────────────────────────────────────────────

def _aggregate_xgb(params_list: list[Parameters]) -> Parameters:
    """Merge XGBoost boosters by concatenating their trees (gossip bagging).

    Each node contributes its full booster. The merged booster contains
    all trees from all nodes — equivalent to a bagging ensemble.
    """
    import json

    boosters: list[xgb.Booster] = []
    for params in params_list:
        arr = parameters_to_ndarrays(params)[0]
        b = xgb.Booster()
        b.load_model(bytearray(arr.tobytes()))
        boosters.append(b)

    merged = boosters[0]
    for other in boosters[1:]:
        merged = _merge_boosters(merged, other)

    raw = merged.save_raw("json")
    return ndarrays_to_parameters([np.frombuffer(raw, dtype=np.uint8)])


def _merge_boosters(base: xgb.Booster, other: xgb.Booster) -> xgb.Booster:
    """Concatenate trees of two XGBoost Boosters into a single Booster."""
    import json

    base_cfg  = json.loads(base.save_raw("json"))
    other_cfg = json.loads(other.save_raw("json"))

    base_model  = base_cfg["learner"]["gradient_booster"]["model"]
    other_model = other_cfg["learner"]["gradient_booster"]["model"]

    base_trees  = base_model.get("trees", [])
    other_trees = other_model.get("trees", [])

    # Re-index other trees to avoid id collisions
    offset = len(base_trees)
    for i, t in enumerate(other_trees):
        t["id"] = offset + i

    merged_trees = base_trees + other_trees
    base_model["trees"]     = merged_trees
    base_model["tree_info"] = base_model.get("tree_info", []) + other_model.get("tree_info", [])

    # Update num_trees in gbtree_model_param
    base_model["gbtree_model_param"]["num_trees"] = str(len(merged_trees))

    # Extend iteration_indptr: append other's indptr shifted by offset
    base_indptr  = base_model.get("iteration_indptr", list(range(offset + 1)))
    other_indptr = other_model.get("iteration_indptr", list(range(len(other_trees) + 1)))
    # other_indptr[0] == 0; shift subsequent entries by offset
    extra = [offset + v for v in other_indptr[1:]]
    base_model["iteration_indptr"] = base_indptr + extra

    merged = xgb.Booster()
    merged.load_model(bytearray(json.dumps(base_cfg).encode("utf-8")))
    return merged
