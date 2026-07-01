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


class XgbBaggingStrategy(FedAvg):
    """Aggregates XGBoost models by pooling all client trees (bagging)."""

    def __init__(self, num_features: int, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
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

        # Collect serialised model bytes from each client
        tree_dicts = []
        for _, fit_res in results:
            arr = parameters_to_ndarrays(fit_res.parameters)[0]
            model_bytes = arr.tobytes()
            local_booster = xgb.Booster()
            local_booster.load_model(bytearray(model_bytes))
            tree_dicts.append(local_booster.get_dump())

        # Build global model by loading first client's model then appending others
        _, first_res = results[0]
        first_bytes = parameters_to_ndarrays(first_res.parameters)[0].tobytes()
        self.global_model = xgb.Booster()
        self.global_model.load_model(bytearray(first_bytes))

        # Append trees from remaining clients via JSON tree pool merge
        for _, fit_res in results[1:]:
            arr = parameters_to_ndarrays(fit_res.parameters)[0]
            other = xgb.Booster()
            other.load_model(bytearray(arr.tobytes()))
            # Merge by loading combined model from JSON config
            self.global_model = _merge_boosters(self.global_model, other)

        # Return serialised global model as parameters
        model_bytes = self.global_model.save_raw("json")
        params = ndarrays_to_parameters([np.frombuffer(model_bytes, dtype=np.uint8)])
        return params, {}

    # ------------------------------------------------------------------
    def get_global_model(self) -> Optional[xgb.Booster]:
        return self.global_model


def _merge_boosters(base: xgb.Booster, other: xgb.Booster) -> xgb.Booster:
    """Merge two Boosters by concatenating their trees via DMatrix re-fit workaround.

    XGBoost does not expose a direct 'merge trees' API, so we use
    load_model on a combined JSON that stacks both model's trees.
    """
    import json

    base_cfg  = json.loads(base.save_model(to_buffer=True))
    other_cfg = json.loads(other.save_model(to_buffer=True))

    # Stack learner trees
    base_trees  = base_cfg.get("learner", {}).get("gradient_booster", {}).get("model", {}).get("trees", [])
    other_trees = other_cfg.get("learner", {}).get("gradient_booster", {}).get("model", {}).get("trees", [])

    # Re-index other trees to avoid id collisions
    offset = len(base_trees)
    for i, t in enumerate(other_trees):
        t["id"] = offset + i

    base_cfg["learner"]["gradient_booster"]["model"]["trees"] = base_trees + other_trees

    # Update tree_info list
    base_info  = base_cfg["learner"]["gradient_booster"]["model"].get("tree_info", [])
    other_info = other_cfg["learner"]["gradient_booster"]["model"].get("tree_info", [])
    base_cfg["learner"]["gradient_booster"]["model"]["tree_info"] = base_info + other_info

    merged = xgb.Booster()
    merged.load_model(bytearray(json.dumps(base_cfg).encode()))
    return merged
