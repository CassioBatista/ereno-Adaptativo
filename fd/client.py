"""Flower client for ERENO-FD.

Supports two federation modes (set via the `config` dict received in fit()):
  - mode="ensemble"      : train any sklearn classifier, send full model
  - mode="federated_nb"  : train GaussianNB, send sufficient statistics

Follows Flower 1.31 guidelines: ErenoClient implements NumPyClient;
make_client_app() wraps it in a ClientApp for use with run_simulation().
"""

from typing import Any

import numpy as np
from sklearn.base import clone
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar

import python.util as util
from fd.model import serialize_model, nb_to_params


class ErenoClient(NumPyClient):

    def __init__(
        self,
        cid: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        classifier,           # sklearn estimator instance
    ):
        self.cid = cid
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.classifier = clone(classifier)

    # ------------------------------------------------------------------
    def get_parameters(self, config: dict[str, Any]) -> NDArrays:
        return serialize_model(self.classifier)

    # ------------------------------------------------------------------
    def fit(self, parameters: NDArrays, config: dict[str, Any]) -> tuple[NDArrays, int, dict]:
        """Train on local data and return parameters according to mode."""
        self.classifier.fit(self.X_train, self.y_train)

        mode: str = config.get("mode", "ensemble")

        match mode:
            case "federated_nb":
                # Send GaussianNB sufficient statistics
                params = nb_to_params(self.classifier)
            case _:  # "ensemble" — send full serialised model
                params = serialize_model(self.classifier)

        return params, len(self.X_train), {"cid": self.cid, "mode": mode}

    # ------------------------------------------------------------------
    def evaluate(self, parameters: NDArrays, config: dict[str, Any]) -> tuple[float, int, dict]:
        """Evaluate local model on local test split."""
        from sklearn.utils.validation import check_is_fitted
        from sklearn.exceptions import NotFittedError
        try:
            check_is_fitted(self.classifier)
        except NotFittedError:
            # Called before fit() with initial parameters — skip gracefully.
            return 1.0, len(self.X_test), {"local_accuracy": 0.0, "cid": self.cid}

        y_pred = self.classifier.predict(self.X_test)
        acc    = float(np.mean(y_pred == self.y_test))
        return 1.0 - acc, len(self.X_test), {"local_accuracy": acc, "cid": self.cid}


def make_client_app(
    client_splits: list[tuple],
    base_clf,
) -> ClientApp:
    """Build a ClientApp wrapping ErenoClient.

    client_splits: list of (X_train, y_train, X_test, y_test) per client.
    base_clf     : unfitted sklearn estimator (will be cloned per client).
    """
    def client_fn(context: Context) -> NumPyClient:
        cid = int(context.node_id % len(client_splits))
        X_ctr, y_ctr, X_cte, y_cte = client_splits[cid]
        return ErenoClient(cid, X_ctr, y_ctr, X_cte, y_cte, base_clf)

    return ClientApp(client_fn=client_fn)
