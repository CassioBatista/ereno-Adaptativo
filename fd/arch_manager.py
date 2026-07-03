"""Architecture manager — decides whether each round runs as 'federated' or 'gossip'.

Phase 1: FixedArchManager reads a static schedule from conf/base.yaml.
Phase 2: ApiArchManager (stub) will receive commands from an external REST API.
"""

from __future__ import annotations


class ArchitectureManager:
    """Abstract base — answers get_mode(round) -> 'federated' | 'gossip'."""

    def get_mode(self, round: int) -> str:
        raise NotImplementedError

    def notify_round_start(self, round: int, mode: str) -> None:
        """Optional hook called by HybridStrategy before each round."""


# ── Phase 1 ───────────────────────────────────────────────────────────────────

class FixedArchManager(ArchitectureManager):
    """Decides mode per round from a static schedule list.

    Schedule format (list of dicts):
        [
            {"from": 1,  "to": 10, "mode": "federated"},
            {"from": 11, "to": 30, "mode": "gossip"},
            {"from": 31, "to": 50, "mode": "federated"},
        ]

    Rounds not covered by any entry fall back to `default_mode`.
    """

    def __init__(
        self,
        schedule: list[dict],
        default_mode: str = "federated",
    ) -> None:
        self.schedule     = schedule
        self.default_mode = default_mode
        self._validate()

    def _validate(self) -> None:
        valid = {"federated", "gossip"}
        for entry in self.schedule:
            if entry.get("mode") not in valid:
                raise ValueError(
                    f"mode inválido: '{entry.get('mode')}'. Use 'federated' ou 'gossip'."
                )
            if entry["from"] > entry["to"]:
                raise ValueError(
                    f"Entrada inválida no schedule: from={entry['from']} > to={entry['to']}"
                )

    def get_mode(self, round: int) -> str:
        for entry in self.schedule:
            if entry["from"] <= round <= entry["to"]:
                return entry["mode"]
        return self.default_mode

    def notify_round_start(self, round: int, mode: str) -> None:
        print(f"[ArchManager] round={round}  mode={mode}")


def load_arch_manager(conf: dict) -> ArchitectureManager:
    """Factory — instantiates the manager specified in conf/base.yaml.

    conf expected structure:
        architecture:
          manager: fixed          # or 'api' (future)
          default_mode: federated
          schedule:
            - {from: 1,  to: 10, mode: federated}
            - {from: 11, to: 30, mode: gossip}
    """
    arch_conf = conf.get("architecture", {})
    kind      = arch_conf.get("manager", "fixed")

    if kind == "fixed":
        return FixedArchManager(
            schedule     = arch_conf.get("schedule", [{"from": 1, "to": 999_999, "mode": "federated"}]),
            default_mode = arch_conf.get("default_mode", "federated"),
        )

    if kind == "api":
        raise NotImplementedError(
            "ApiArchManager ainda não implementado. "
            "Use manager: fixed no conf/base.yaml por enquanto."
        )

    raise ValueError(f"manager desconhecido: '{kind}'. Use 'fixed' ou 'api'.")
