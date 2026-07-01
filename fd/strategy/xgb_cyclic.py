"""Federated XGBoost — Cyclic strategy.

Each round a single client receives the current global model and adds
new trees on top (sequential boosting across clients).

Converges closer to centralised XGBoost because each client trains on
residuals from the globally accumulated model.

Flower is configured with num_rounds=num_clients; each round selects
client (round-1) % num_clients so every client trains exactly once.
"""

from typing import Optional

import numpy as np
import xgboost as xgb
from flwr.common import (
    FitIns, FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class XgbCyclicStrategy(FedAvg):
    """Sequential XGBoost federation: one client per round adds trees."""

    def __init__(self, num_clients: int, **kwargs):
        super().__init__(**kwargs)
        self.num_clients  = num_clients
        self.global_model: Optional[xgb.Booster] = None
        self._current_params: Parameters = ndarrays_to_parameters(
            [np.array([], dtype=np.uint8)]
        )

    # ------------------------------------------------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Select exactly one client per round (cyclic)."""
        all_clients = list(client_manager.all().values())
        selected    = all_clients[(server_round - 1) % self.num_clients]
        fit_ins     = FitIns(parameters, {"mode": "xgb_cyclic"})
        return [(selected, fit_ins)]

    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        if not results:
            return None, {}

        _, fit_res = results[0]
        arr = parameters_to_ndarrays(fit_res.parameters)[0]
        model_bytes = arr.tobytes()

        self.global_model = xgb.Booster()
        self.global_model.load_model(bytearray(model_bytes))

        model_bytes_out = self.global_model.save_raw("json")
        self._current_params = ndarrays_to_parameters(
            [np.frombuffer(model_bytes_out, dtype=np.uint8)]
        )
        return self._current_params, {}

    # ------------------------------------------------------------------
    def get_global_model(self) -> Optional[xgb.Booster]:
        return self.global_model
