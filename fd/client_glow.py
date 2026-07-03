"""GlowClient — dual-role client for Gossip Learning.

Role is determined per-round by the config sent from GlowStrategy:
  HEAD      : trains locally and returns updated model
  VALIDATOR : evaluates the head's model on local data, returns metrics
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone, BaseEstimator
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, NDArrays

from fd.model import serialize_model, deserialize_model


class GlowClient(NumPyClient):
    """Gossip Learning client with HEAD / VALIDATOR dual role.

    Parameters
    ----------
    cid     : client / node index (0..N-1)
    X_tr    : local training features
    y_tr    : local training labels
    X_te    : local test features
    y_te    : local test labels
    base_clf: unfitted sklearn estimator used as local model
    """

    def __init__(
        self,
        cid:      int,
        X_tr:     np.ndarray,
        y_tr:     np.ndarray,
        X_te:     np.ndarray,
        y_te:     np.ndarray,
        base_clf: BaseEstimator,
    ) -> None:
        self.cid      = cid
        self.X_tr     = X_tr
        self.y_tr     = y_tr
        self.X_te     = X_te
        self.y_te     = y_te
        self.base_clf = base_clf
        self._model:  BaseEstimator | None = None

    # ── NumPyClient interface ─────────────────────────────────────────────────

    def get_parameters(self, config: dict) -> NDArrays:
        if self._model is None:
            self._model = clone(self.base_clf)
            self._model.fit(self.X_tr, self.y_tr)
        return serialize_model(self._model)

    def fit(self, parameters: NDArrays, config: dict) -> tuple[NDArrays, int, dict]:
        head_cid = int(config.get("head_cid", -1))
        is_head  = (self.cid == head_cid)

        if is_head:
            # HEAD role: train locally with received parameters as warm start
            self._model = deserialize_model(parameters)
            if self._model is None:
                self._model = clone(self.base_clf)
            self._model.fit(self.X_tr, self.y_tr)
            updated_params = serialize_model(self._model)
            acc = float(np.mean(self._model.predict(self.X_te) == self.y_te))
            return updated_params, len(self.X_tr), {"accuracy": acc, "role": 1.0}

        else:
            # VALIDATOR role: load model, evaluate on local data, return unchanged
            self._model = deserialize_model(parameters)
            if self._model is not None:
                acc = float(np.mean(self._model.predict(self.X_te) == self.y_te))
            else:
                acc = 0.0
            return parameters, len(self.X_tr), {"accuracy": acc, "role": 0.0}

    def evaluate(self, parameters: NDArrays, config: dict) -> tuple[float, int, dict]:
        self._model = deserialize_model(parameters)
        if self._model is None:
            return 1.0, len(self.X_te), {"accuracy": 0.0}
        y_pred = self._model.predict(self.X_te)
        acc    = float(np.mean(y_pred == self.y_te))
        loss   = 1.0 - acc
        return loss, len(self.X_te), {"accuracy": acc}


# ── factory ───────────────────────────────────────────────────────────────────

def make_glow_client_app(
    client_splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    base_clf:      BaseEstimator,
) -> ClientApp:
    """Create a ClientApp where each supernode maps to a GlowClient.

    Parameters
    ----------
    client_splits : list of (X_tr, y_tr, X_te, y_te) per client
    base_clf      : unfitted sklearn estimator (will be cloned per client)
    """
    def client_fn(context: Context) -> NumPyClient:
        cid = int(context.node_config["partition-id"])
        X_tr, y_tr, X_te, y_te = client_splits[cid]
        return GlowClient(cid, X_tr, y_tr, X_te, y_te, base_clf)

    return ClientApp(client_fn=client_fn)
