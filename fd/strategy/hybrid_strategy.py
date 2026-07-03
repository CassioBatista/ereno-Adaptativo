"""HybridStrategy — meta-strategy that delegates to federated or gossip per round.

The ArchitectureManager decides which mode is active for each round.
On mode transitions, _transfer_model() adapts the global model state
so training continuity is preserved across paradigm switches.
"""

from __future__ import annotations

from flwr.common import Parameters, Scalar, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from fd.arch_manager import ArchitectureManager


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
    """

    def __init__(
        self,
        fed_strategy:  Strategy,
        glow_strategy: Strategy,
        arch_manager:  ArchitectureManager,
    ) -> None:
        self._strategies  = {"federated": fed_strategy, "gossip": glow_strategy}
        self.arch_manager = arch_manager
        self._prev_mode:  str | None = None
        self._last_params: Parameters | None = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _active(self, round: int) -> tuple[str, Strategy]:
        mode = self.arch_manager.get_mode(round)
        return mode, self._strategies[mode]

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
        return strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results:      FitResults,
        failures:     FitFailures,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        _, strategy = self._active(server_round)
        return strategy.aggregate_fit(server_round, results, failures)

    def configure_evaluate(
        self,
        server_round: int,
        parameters:   Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, any]]:
        _, strategy = self._active(server_round)
        return strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results:      EvalResults,
        failures:     EvalFailures,
    ) -> tuple[float | None, dict[str, Scalar]]:
        _, strategy = self._active(server_round)
        return strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self,
        server_round: int,
        parameters:   Parameters,
    ) -> tuple[float, dict[str, Scalar]] | None:
        _, strategy = self._active(server_round)
        return strategy.evaluate(server_round, parameters)

    # ── convenience accessors ─────────────────────────────────────────────────

    def get_fed_strategy(self) -> Strategy:
        return self._strategies["federated"]

    def get_glow_strategy(self) -> Strategy:
        return self._strategies["gossip"]
