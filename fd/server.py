"""Flower ServerApp factory for ERENO-FD-SF.

Follows Flower 1.31 simulation guidelines:
  ServerApp encapsulates server-side logic (strategy + round config).
  Called by main_fd.py and handed to run_simulation().
"""

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import Strategy


def make_server_app(strategy: Strategy, num_rounds: int = 1) -> ServerApp:
    """Build a ServerApp with the given aggregation strategy."""
    return ServerApp(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )
