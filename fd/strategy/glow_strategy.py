"""GlowStrategy — Gossip Learning strategy for Flower 1.31.

Each round one node is elected as HEAD (round-robin over active nodes).
The head receives its neighbours' current models, aggregates them,
and returns an updated model. Non-head nodes validate for the head.

Aggregation modes:
  inplace  — weighted average of neighbour models (weight = num_samples)
  score    — weight proportional to neighbour's reported local accuracy
"""

from __future__ import annotations

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


class GlowStrategy(Strategy):
    """Gossip Learning strategy with round-robin head election.

    Parameters
    ----------
    topology        : Topology object with adjacency information
    aggregation     : 'inplace' (weighted avg) | 'score' (accuracy-weighted)
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
        instructions = []

        for proxy in all_clients:
            cid = int(proxy.cid) if proxy.cid.isdigit() else hash(proxy.cid) % self.topology.num_nodes

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

        # Collect updated models from all nodes
        for proxy, fit_res in results:
            cid = int(proxy.cid) if proxy.cid.isdigit() else hash(proxy.cid) % self.topology.num_nodes
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
        instructions = []
        for proxy in all_clients:
            cid = int(proxy.cid) if proxy.cid.isdigit() else hash(proxy.cid) % self.topology.num_nodes
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
