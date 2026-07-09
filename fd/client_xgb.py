"""Flower client for XGBoost federation in ERENO-FD-SF.

Separate from ErenoClient (sklearn) to avoid modifying tested code.
Supports two XGBoost federation modes:
  - mode="xgb_bagging" : train local model, send full Booster
  - mode="xgb_cyclic"  : receive global model, add trees, send back

Follows Flower 1.31 guidelines: XgbClient implements NumPyClient;
make_xgb_client_app() wraps it in a ClientApp for use with run_simulation().
"""

from typing import Any

import numpy as np
import xgboost as xgb
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, parameters_to_ndarrays

import python.util as util


_DEFAULT_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "eta":              0.1,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "seed":             42,
    "nthread":          1,
}

_N_LOCAL_ROUNDS = 10   # trees added per client per federation round


class XgbClient(NumPyClient):

    def __init__(
        self,
        cid: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        xgb_params: dict | None = None,
    ):
        self.cid     = cid
        self.dtrain  = xgb.DMatrix(X_train, label=y_train)
        self.dtest   = xgb.DMatrix(X_test,  label=y_test)
        self.y_test  = y_test
        self.params  = xgb_params or _DEFAULT_PARAMS
        self.booster: xgb.Booster | None = None

    # ------------------------------------------------------------------
    def get_parameters(self, config: dict[str, Any]) -> NDArrays:
        if self.booster is None:
            return [np.array([], dtype=np.uint8)]
        raw = self.booster.save_raw("json")
        return [np.frombuffer(raw, dtype=np.uint8)]

    # ------------------------------------------------------------------
    def fit(self, parameters: NDArrays, config: dict[str, Any]) -> tuple[NDArrays, int, dict]:
        mode: str = config.get("mode", "xgb_bagging")

        # Reequilibra a perda pelo desbalanceamento LOCAL do cliente
        # (binary:logistic): classes passam a pesar igual sem duplicar dados.
        label = self.dtrain.get_label()
        n_pos = float(label.sum())
        n_neg = float(len(label)) - n_pos
        params = dict(self.params)
        if n_pos > 0 and n_neg > 0:
            params["scale_pos_weight"] = n_neg / n_pos
        self.params = params

        if mode == "xgb_cyclic" and parameters[0].size > 0:
            # Load global model received from server, then continue boosting
            self.booster = xgb.Booster()
            self.booster.load_model(bytearray(parameters[0].tobytes()))
            self.booster = xgb.train(
                self.params,
                self.dtrain,
                num_boost_round=_N_LOCAL_ROUNDS,
                xgb_model=self.booster,
                verbose_eval=False,
            )
        else:
            # bagging or first cyclic round: train from scratch
            self.booster = xgb.train(
                self.params,
                self.dtrain,
                num_boost_round=_N_LOCAL_ROUNDS,
                verbose_eval=False,
            )

        raw = self.booster.save_raw("json")
        return [np.frombuffer(raw, dtype=np.uint8)], len(self.dtrain.get_label()), {"cid": self.cid}

    # ------------------------------------------------------------------
    def evaluate(self, parameters: NDArrays, config: dict[str, Any]) -> tuple[float, int, dict]:
        if self.booster is None or parameters[0].size == 0:
            return 1.0, len(self.y_test), {"local_accuracy": 0.0, "cid": self.cid}

        y_prob = self.booster.predict(self.dtest)
        y_pred = (y_prob >= 0.5).astype(int)
        acc    = float(np.mean(y_pred == self.y_test))
        return 1.0 - acc, len(self.y_test), {"local_accuracy": acc, "cid": self.cid}


def make_xgb_client_app(
    client_splits: list[tuple],
    xgb_params: dict | None = None,
) -> ClientApp:
    """Build a ClientApp wrapping XgbClient.

    client_splits: list of (X_train, y_train, X_test, y_test) per client.
    xgb_params   : optional XGBoost hyperparameters (uses defaults if None).
    """
    def client_fn(context: Context) -> NumPyClient:
        cid = int(context.node_config["partition-id"])
        X_ctr, y_ctr, X_cte, y_cte = client_splits[cid]
        return XgbClient(cid, X_ctr, y_ctr, X_cte, y_cte, xgb_params)

    return ClientApp(client_fn=client_fn)
