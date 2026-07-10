"""Federated XGBoost — Bagging strategy.

Each client trains a local XGBoost model and sends its trees.
The server pools all trees into a single global Booster (bagging).

Global model has N_clients × n_estimators trees total.
"""

from logging import WARNING
from typing import Optional

import numpy as np
import xgboost as xgb
from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fd.strategy.glow_strategy import _merge_boosters


class XgbBaggingStrategy(FedAvg):
    """Aggregates XGBoost models by pooling all client trees (bagging).

    fusion:
      'sum' — merge all client trees into ONE booster (margens somadas);
      'or'  — keep each client's booster separate (um tensor por booster);
              a predição global dispara se QUALQUER booster disparar
              (união de decisões — adequada a sensores especialistas,
              onde a soma de margens é vetada pela maioria "normal").
    """

    def __init__(self, num_features: int, fusion: str = "sum", **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.fusion = fusion
        self.global_model: Optional[xgb.Booster] = None

    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        if not results:
            return None, {}

        if self.fusion == "or":
            # um tensor por booster de cliente — fusão acontece na predição
            arrays = [parameters_to_ndarrays(fit_res.parameters)[0]
                      for _, fit_res in results]
            return ndarrays_to_parameters(arrays), {}

        # fusion == 'sum': merge all client trees into one booster
        _, first_res = results[0]
        first_bytes = parameters_to_ndarrays(first_res.parameters)[0].tobytes()
        self.global_model = xgb.Booster()
        self.global_model.load_model(bytearray(first_bytes))

        # Append trees from remaining clients via JSON tree pool merge
        for _, fit_res in results[1:]:
            arr = parameters_to_ndarrays(fit_res.parameters)[0]
            other = xgb.Booster()
            other.load_model(bytearray(arr.tobytes()))
            self.global_model = _merge_boosters(self.global_model, other)

        # Return serialised global model as parameters
        model_bytes = self.global_model.save_raw("json")
        params = ndarrays_to_parameters([np.frombuffer(model_bytes, dtype=np.uint8)])
        return params, {}

    # ------------------------------------------------------------------
    def get_global_model(self) -> Optional[xgb.Booster]:
        return self.global_model
