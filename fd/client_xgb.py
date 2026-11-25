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
        arrival_s: float | None = None,
        specialists: list[list[int]] | None = None,
        normal_class: int | None = None,
    ):
        self.cid     = cid
        self.params  = xgb_params or _DEFAULT_PARAMS
        self.booster: xgb.Booster | None = None
        # partitioner 'specialists' (evaluation-v1): y_* keep the class codes and the node
        # trains one binary model per specialist (its codes against the benign rows)
        self.specialists = specialists
        self.normal_class = normal_class
        self.boosters: list[xgb.Booster] = []
        if specialists is not None:
            self.X_train, self.y_train = X_train, y_train
            y_test = (y_test != normal_class).astype(np.int64)
            y_train = (y_train != normal_class).astype(np.int64)
        self.dtrain  = xgb.DMatrix(X_train, label=y_train)
        self.dtest   = xgb.DMatrix(X_test,  label=y_test)
        self.y_test  = y_test
        # time (s since run_t0) at which this node's data -- its block of specialists --
        # exists (evaluation-v1 section 9); None = from the start. Only used when the
        # server sends the round time (config "t_s", node_holdings mode).
        self.arrival_s = arrival_s

    # ------------------------------------------------------------------
    def get_parameters(self, config: dict[str, Any]) -> NDArrays:
        if self.specialists is not None and self.boosters:
            return [np.frombuffer(b.save_raw("json"), dtype=np.uint8) for b in self.boosters]
        if self.booster is None:
            return [np.array([], dtype=np.uint8)]
        raw = self.booster.save_raw("json")
        return [np.frombuffer(raw, dtype=np.uint8)]

    # ------------------------------------------------------------------
    def fit(self, parameters: NDArrays, config: dict[str, Any]) -> tuple[NDArrays, int, dict]:
        mode: str = config.get("mode", "xgb_bagging")
        # Rounds gossip (GLow) carregam o contexto de vizinhança no config.
        is_gossip = "head_cid" in config
        # node_holdings mode (HybridStrategy): the server sends the round time, whether
        # this node is in local mode, and asks for the node's own specialist only.
        local = int(config.get("local", 0))
        t_s = config.get("t_s")
        if t_s is not None and self.arrival_s is not None and float(t_s) < self.arrival_s:
            # the node's block has not arrived yet: nothing of its own to contribute
            return ([np.array([], dtype=np.uint8)], len(self.dtrain.get_label()),
                    {"cid": self.cid, "local": local, "arrived": 0})

        if self.specialists is not None:
            # one binary model per specialist of this node, trained from scratch on its
            # own attack rows against the node's benign slice (scripts/ev1_step1.py)
            out = []
            self.boosters = []
            for codes in self.specialists:
                m = np.isin(self.y_train, codes) | (self.y_train == self.normal_class)
                yb = np.isin(self.y_train[m], codes).astype(np.int8)
                p = dict(self.params)
                n_pos, n_neg = int(yb.sum()), int((yb == 0).sum())
                if n_pos and n_neg:
                    p["scale_pos_weight"] = n_neg / n_pos
                b = xgb.train(p, xgb.DMatrix(self.X_train[m], label=yb),
                              num_boost_round=_N_LOCAL_ROUNDS, verbose_eval=False)
                self.boosters.append(b)
                out.append(np.frombuffer(b.save_raw("json"), dtype=np.uint8))
            self.booster = self.boosters[0] if self.boosters else None
            return out, len(self.y_train), {"cid": self.cid, "local": local, "arrived": 1}

        # Reequilibra a perda pelo desbalanceamento LOCAL do cliente
        # (binary:logistic): classes passam a pesar igual sem duplicar dados.
        label = self.dtrain.get_label()
        n_pos = float(label.sum())
        n_neg = float(len(label)) - n_pos
        params = dict(self.params)
        if n_pos > 0 and n_neg > 0:
            params["scale_pos_weight"] = n_neg / n_pos
        self.params = params

        adopt = not int(config.get("fresh_only", 0)) and (
            # cyclic: continua o boosting do modelo global recebido
            (mode == "xgb_cyclic" and parameters[0].size > 0)
            # gossip fiel ao GLow: ADOTA o modelo recebido (o equivalente
            # para árvores do set_parameters + train do GLow original).
            # Conjuntos multi-booster (fusão OR) não têm warm-start único —
            # nesse modo o nó treina do zero e a memória vive no pool.
            or (is_gossip and len(parameters) == 1 and parameters[0].size > 0)
        )

        if adopt:
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
            # bagging federado ou primeiro round: treina do zero
            self.booster = xgb.train(
                self.params,
                self.dtrain,
                num_boost_round=_N_LOCAL_ROUNDS,
                verbose_eval=False,
            )

        raw = self.booster.save_raw("json")
        return ([np.frombuffer(raw, dtype=np.uint8)], len(self.dtrain.get_label()),
                {"cid": self.cid, "local": local, "arrived": 1})

    # ------------------------------------------------------------------
    def evaluate(self, parameters: NDArrays, config: dict[str, Any]) -> tuple[float, int, dict]:
        if self.booster is None or not parameters or parameters[0].size == 0:
            return 1.0, len(self.y_test), {"local_accuracy": 0.0, "cid": self.cid}

        if self.specialists is not None:      # OR over the node's own specialists
            y_pred = np.zeros(len(self.y_test), dtype=int)
            for b in self.boosters:
                y_pred |= (b.predict(self.dtest) >= 0.5).astype(int)
        else:
            y_prob = self.booster.predict(self.dtest)
            y_pred = (y_prob >= 0.5).astype(int)
        acc    = float(np.mean(y_pred == self.y_test))
        return 1.0 - acc, len(self.y_test), {"local_accuracy": acc, "cid": self.cid}


def make_xgb_client_app(
    client_splits: list[tuple],
    xgb_params: dict | None = None,
    arrival: dict[int, float] | None = None,
    specialist_groups: list[list[list[int]]] | None = None,
    normal_class: int | None = None,
) -> ClientApp:
    """Build a ClientApp wrapping XgbClient.

    client_splits: list of (X_train, y_train, X_test, y_test) per client.
    xgb_params   : optional XGBoost hyperparameters (uses defaults if None).
    arrival      : optional {node index: arrival time in s since run_t0} of each node's
                   block (nodes absent: from the start).
    specialist_groups / normal_class: partitioner 'specialists' -- per node, the class
                   codes of each specialist it trains; y_* then carry class codes.
    """
    arrival = arrival or {}

    def client_fn(context: Context) -> NumPyClient:
        cid = int(context.node_config["partition-id"])
        X_ctr, y_ctr, X_cte, y_cte = client_splits[cid]
        return XgbClient(cid, X_ctr, y_ctr, X_cte, y_cte, xgb_params,
                         arrival_s=arrival.get(cid),
                         specialists=specialist_groups[cid] if specialist_groups else None,
                         normal_class=normal_class)

    return ClientApp(client_fn=client_fn)
