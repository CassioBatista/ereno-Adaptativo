"""HybridStrategy — meta-strategy that delegates to federated or gossip per round.

The ArchitectureManager decides which mode is active for each round.
On mode transitions, _transfer_model() adapts the global model state
so training continuity is preserved across paradigm switches.
"""

from __future__ import annotations

from flwr.common import Parameters, Scalar, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import Strategy

from fd.arch_manager import ArchitectureManager
from fd.strategy.glow_strategy import resolve_node_index, learn_cid_map


class _AllowedCidsCriterion(Criterion):
    """Selects only clients whose (real) proxy cid is in the allowed set."""

    def __init__(self, allowed_cids: set[str]) -> None:
        self._allowed = allowed_cids

    def select(self, client: ClientProxy) -> bool:
        return client.cid in self._allowed


FitResults     = list[tuple[ClientProxy, any]]
FitFailures    = list[tuple[ClientProxy, Exception]]
EvalResults    = list[tuple[ClientProxy, any]]
EvalFailures   = list[tuple[ClientProxy, Exception]]


class HybridStrategy(Strategy):
    """Delegates configure_fit / aggregate_fit to the active sub-strategy.

    Parameters
    ----------
    fed_strategy  : Strategy to use when mode == 'federated'
    glow_strategy : Strategy to use when mode == 'gossip'
    arch_manager  : ArchitectureManager that returns the mode for each round
    round_eval_fn : optional callback (round, mode, parameters) -> (loss, metrics) | None,
                    called server-side after each round with the aggregated
                    parameters (per-round convergence metrics)
    """

    def __init__(
        self,
        fed_strategy:  Strategy,
        glow_strategy: Strategy,
        arch_manager:  ArchitectureManager,
        round_eval_fn=None,
    ) -> None:
        self._strategies  = {"federated": fed_strategy, "gossip": glow_strategy}
        self.arch_manager = arch_manager
        self.round_eval_fn = round_eval_fn
        self._prev_mode:  str | None = None
        self._last_params: Parameters | None = None
        self.final_parameters: Parameters | None = None
        # proxy cid -> partition id, learned from client metrics and shared
        # with the gossip strategy (proxy cids are opaque under flwr sim)
        self.cid_map: dict[str, int] = {}
        if hasattr(glow_strategy, "cid_map"):
            glow_strategy.cid_map = self.cid_map

    # ── internal ──────────────────────────────────────────────────────────────

    def _active(self, round: int) -> tuple[str, Strategy]:
        mode = self.arch_manager.get_mode(round)
        return mode, self._strategies[mode]

    def _filter_manager(
        self, client_manager: ClientManager, round: int
    ) -> ClientManager:
        """Return a view of client_manager restricted to active clients, if defined."""
        active = self.arch_manager.get_active_clients(round)
        if active is None:
            return client_manager
        return _FilteredClientManager(client_manager, active, self.cid_map)

    def _sync_participation(self, strategy: Strategy, active: list[int] | None) -> None:
        """Align the sub-strategy's expectations with the round's active clients.

        FedAvg-style strategies: min_*_clients must not exceed the number of
        active clients, or sampling fails and the round becomes a no-op.
        Originals are captured once and restored when active is None.

        GlowStrategy: node up/down status in the topology must mirror the
        active set, or head election may pick an inactive node.
        """
        if hasattr(strategy, "min_fit_clients"):
            if not hasattr(strategy, "_orig_min_clients"):
                strategy._orig_min_clients = (
                    strategy.min_fit_clients,
                    strategy.min_evaluate_clients,
                    strategy.min_available_clients,
                )
            orig_fit, orig_eval, orig_avail = strategy._orig_min_clients
            if active is None:
                strategy.min_fit_clients       = orig_fit
                strategy.min_evaluate_clients  = orig_eval
                strategy.min_available_clients = orig_avail
            else:
                n = len(active)
                strategy.min_fit_clients       = min(orig_fit, n)
                strategy.min_evaluate_clients  = min(orig_eval, n)
                strategy.min_available_clients = min(orig_avail, n)

        topology = getattr(strategy, "topology", None)
        if topology is not None:
            for node in topology.all_nodes():
                topology.set_status(node, active is None or node in active)

    def _transfer_model(
        self,
        from_mode: str,
        to_mode:   str,
        parameters: Parameters,
    ) -> Parameters:
        """Adapt global model state when switching paradigms.

        federated → gossip:
            The single global model becomes each node's local starting point.
            GlowStrategy.pool_parameters is initialized uniformly.

        gossip → federated:
            pool_parameters are averaged into a new global model,
            which becomes the initial_parameters for the federated strategy.
        """
        if from_mode == "gossip" and to_mode == "federated":
            glow = self._strategies["gossip"]
            if hasattr(glow, "pool_parameters") and glow.pool_parameters:
                arrays_list = [
                    parameters_to_ndarrays(p)
                    for p in glow.pool_parameters.values()
                ]
                avg: NDArrays = [
                    sum(a[i] for a in arrays_list) / len(arrays_list)
                    for i in range(len(arrays_list[0]))
                ]
                parameters = ndarrays_to_parameters(avg)
                print(f"[Hybrid] gossip→federated: averaged {len(arrays_list)} node models")

        elif from_mode == "federated" and to_mode == "gossip":
            glow = self._strategies["gossip"]
            if hasattr(glow, "pool_parameters"):
                for node_id in glow.pool_parameters:
                    glow.pool_parameters[node_id] = parameters
                print(f"[Hybrid] federated→gossip: distributed global model to all nodes")

        return parameters

    # ── Strategy interface ────────────────────────────────────────────────────

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        fed = self._strategies["federated"]
        return fed.initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters:   Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, any]]:
        mode, strategy = self._active(server_round)
        self.arch_manager.notify_round_start(server_round, mode)

        if self._prev_mode is not None and mode != self._prev_mode:
            parameters = self._transfer_model(self._prev_mode, mode, parameters)
            print(f"[Hybrid] mode switch: {self._prev_mode} → {mode} at round {server_round}")

        self._prev_mode   = mode
        self._last_params = parameters
        self._sync_participation(strategy, self.arch_manager.get_active_clients(server_round))
        filtered_manager  = self._filter_manager(client_manager, server_round)
        return strategy.configure_fit(server_round, parameters, filtered_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results:      FitResults,
        failures:     FitFailures,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        learn_cid_map(self.cid_map, results)
        _, strategy = self._active(server_round)
        params, metrics = strategy.aggregate_fit(server_round, results, failures)
        if params is not None:
            self.final_parameters = params
        return params, metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters:   Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, any]]:
        _, strategy = self._active(server_round)
        self._sync_participation(strategy, self.arch_manager.get_active_clients(server_round))
        filtered_manager = self._filter_manager(client_manager, server_round)
        return strategy.configure_evaluate(server_round, parameters, filtered_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results:      EvalResults,
        failures:     EvalFailures,
    ) -> tuple[float | None, dict[str, Scalar]]:
        learn_cid_map(self.cid_map, results)
        _, strategy = self._active(server_round)
        return strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self,
        server_round: int,
        parameters:   Parameters,
    ) -> tuple[float, dict[str, Scalar]] | None:
        mode, strategy = self._active(server_round)
        if self.round_eval_fn is not None and server_round > 0:
            return self.round_eval_fn(server_round, mode, parameters)
        return strategy.evaluate(server_round, parameters)

    # ── convenience accessors ─────────────────────────────────────────────────

    def get_fed_strategy(self) -> Strategy:
        return self._strategies["federated"]

    def get_glow_strategy(self) -> Strategy:
        return self._strategies["gossip"]


# ── FilteredClientManager ─────────────────────────────────────────────────────

class _FilteredClientManager(ClientManager):
    """Read-only view of a ClientManager restricted to a subset of node indices.

    The active ids refer to logical partition/node indices (0..N-1); each
    proxy cid is resolved to its index via the shared cid_map (learned from
    client metrics), so filtering works under flwr's simulation where proxy
    cids are opaque node ids. All mutating operations are forwarded to the
    base manager.
    """

    def __init__(
        self,
        base:       ClientManager,
        active_ids: list[int],
        cid_map:    dict[str, int] | None = None,
    ) -> None:
        self._base    = base
        self._active  = set(active_ids)
        self._cid_map = cid_map if cid_map is not None else {}

    def _allowed_cids(self) -> set[str]:
        all_cids = list(self._base.all().keys())
        n = len(all_cids)
        return {
            cid for cid in all_cids
            if resolve_node_index(cid, all_cids, self._cid_map, n) in self._active
        }

    def num_available(self) -> int:
        return len(self._allowed_cids())

    def register(self, client: ClientProxy) -> bool:
        return self._base.register(client)

    def unregister(self, client: ClientProxy) -> None:
        self._base.unregister(client)

    def all(self) -> dict[str, ClientProxy]:
        allowed = self._allowed_cids()
        return {cid: proxy for cid, proxy in self._base.all().items()
                if cid in allowed}

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        return self._base.wait_for(num_clients, timeout)

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        active_criterion = _AllowedCidsCriterion(self._allowed_cids())
        if criterion is not None:
            # combine both criteria: client must pass both
            class _Combined(Criterion):
                def __init__(self, a: Criterion, b: Criterion) -> None:
                    self._a, self._b = a, b
                def select(self, client: ClientProxy) -> bool:
                    return self._a.select(client) and self._b.select(client)
            combined = _Combined(active_criterion, criterion)
        else:
            combined = active_criterion
        return self._base.sample(num_clients, min_num_clients, combined)
