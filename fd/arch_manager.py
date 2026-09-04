"""Architecture manager — decides whether each round runs as 'federated' or 'gossip'.

Two managers:
  * FixedArchManager      — static schedule from conf/base.yaml (phase 1).
  * DistributedArchManager — v2: the switch is DISTRIBUTED and AUTOMATIC. There
    is no central schedule and no server-side policy. The control plane is
    replicated across nodes: each node detects failures LOCALLY, self-protects
    to gossip immediately (fail-fast, no quorum), and a global mode commit
    happens only by QUORUM of nodes. Recovery (gossip->federated) is careful:
    full participation restored, stable for a dwell window, past a cooldown.

The centralized Flower simulation runs a single data-plane mode per round, so
DistributedArchManager models the N per-node controllers internally and exposes
the *committed* mode. For crash / node-loss (v2) all nodes observe the same
silence and converge unanimously; the per-node/quorum structure is what makes
the control plane resilient and is the basis for the Byzantine case (v3).
"""

from __future__ import annotations


class ArchitectureManager:
    """Abstract base — answers get_mode(round) and get_active_clients(round)."""

    def get_mode(self, round: int) -> str:
        raise NotImplementedError

    def get_active_clients(self, round: int) -> list[int] | None:
        """Return list of active client IDs for this round, or None for all clients."""
        return None

    def notify_round_start(self, round: int, mode: str) -> None:
        """Optional hook called by HybridStrategy before each round."""

    def observe(self, round: int, reported_nodes: set[int]) -> None:
        """Feedback hook: which node indices reported results this round.

        No-op for static managers; DistributedArchManager uses it as the local
        failure-detection signal (nodes expected but silent are treated as down).
        """

    def consume_switch_reason(self) -> dict | None:
        """Return and clear the reason for the last committed switch, if any.

        Used by HybridStrategy to annotate the monitor's architecture_change
        event with the real cause (node_failure / recovery) instead of inferring
        it. Returns None when there is no pending reason.
        """
        return None


# ── Phase 1 — static schedule ───────────────────────────────────────────────────

class FixedArchManager(ArchitectureManager):
    """Decides mode and active clients per round from a static schedule list.

    Schedule format (list of dicts):
        [
            {"from": 1,  "to": 10, "mode": "federated"},
            {"from": 11, "to": 30, "mode": "gossip",    "active_clients": [0, 1, 2]},
            {"from": 31, "to": 50, "mode": "federated", "active_clients": [0, 1, 2, 3, 4]},
        ]

    active_clients is optional — omit it to use all available clients.
    Rounds not covered by any entry fall back to `default_mode` and all clients.
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
            if "active_clients" in entry:
                ac = entry["active_clients"]
                if not isinstance(ac, list) or not all(isinstance(i, int) for i in ac):
                    raise ValueError(
                        f"active_clients deve ser uma lista de inteiros: {ac}"
                    )

    def _entry_for(self, round: int) -> dict | None:
        for entry in self.schedule:
            if entry["from"] <= round <= entry["to"]:
                return entry
        return None

    def get_mode(self, round: int) -> str:
        entry = self._entry_for(round)
        return entry["mode"] if entry else self.default_mode

    def get_active_clients(self, round: int) -> list[int] | None:
        entry = self._entry_for(round)
        if entry and "active_clients" in entry:
            return entry["active_clients"]
        return None

    def notify_round_start(self, round: int, mode: str) -> None:
        active = self.get_active_clients(round)
        suffix = f"  active_clients={active}" if active is not None else ""
        print(f"[ArchManager] round={round}  mode={mode}{suffix}")


# ── Phase 2 — distributed, automatic switch ─────────────────────────────────────

class _NodeController:
    """One node's replicated control-plane state (local view + local decision).

    Each node believes a set of peers is alive, holds a local mode, and self-
    protects to gossip the moment it detects a believed peer went silent. Its
    vote for the *global* committed mode is derived from its local view:
      * wants gossip while any believed peer is missing (self-protection);
      * wants federated once its full peer set has reported again.
    """

    def __init__(self, node_id: int, all_nodes: set[int], mode: str) -> None:
        self.id       = node_id
        self.believed = set(all_nodes)   # peers this node believes are up
        self.mode     = mode

    def step(self, reported: set[int]) -> None:
        # local detection from this node's own observation of who answered
        if not (self.believed <= reported):          # some believed peer silent
            self.mode = "gossip"                     # fail-fast self-protection
        self.believed = set(reported)                # refresh local view

    def is_up(self, reported: set[int]) -> bool:
        return self.id in reported

    def wants(self, reported: set[int], n_total: int) -> str:
        """This node's vote: 'gossip' while peers are missing, else 'federated'."""
        return "federated" if len(reported) == n_total else "gossip"


class DistributedArchManager(ArchitectureManager):
    """Distributed, automatic FL<->GL switch driven by local failure detection.

    Parameters
    ----------
    n_nodes        : number of federation nodes.
    initial_mode   : starting mode (default 'federated', the efficient base).
    quorum         : nodes needed to commit a global change; default majority
                     of the CURRENTLY-active nodes, q = floor(n_active/2)+1.
    dwell_rounds   : consecutive stable (full-participation) rounds required
                     before a gossip->federated recovery.
    cooldown_rounds: minimum rounds between committed switches (anti-flapping).
    faults         : OPTIONAL environment ground truth (not read by the control
                     logic) — a list of {round, nodes:[...], up:bool} the sim
                     uses to actually drop/restore nodes, so the controllers
                     have something to detect. Detection is from observation,
                     never from this schedule.
    """

    def __init__(
        self,
        n_nodes: int,
        initial_mode: str = "federated",
        quorum: int | None = None,
        dwell_rounds: int = 3,
        cooldown_rounds: int = 2,
        faults: list[dict] | None = None,
    ) -> None:
        if initial_mode not in ("federated", "gossip"):
            raise ValueError(f"initial_mode inválido: '{initial_mode}'.")
        self.n_nodes         = n_nodes
        self.default_mode    = initial_mode
        self._committed_mode = initial_mode
        self._fixed_quorum   = quorum
        self._dwell          = max(1, int(dwell_rounds))
        self._cooldown       = max(0, int(cooldown_rounds))
        self._faults         = sorted((faults or []), key=lambda f: f["from"] if "from" in f else f["round"])
        alln                 = set(range(n_nodes))
        self._nodes          = {i: _NodeController(i, alln, initial_mode) for i in range(n_nodes)}
        self._believed_alive = set(alln)         # union view for logging/detection
        self._last_switch    = -10 ** 9
        self._stable         = 0
        self._pending_reason: dict | None = None

    # -- environment ground truth (participation), NOT used by control logic --

    def _env_alive(self, round: int) -> set[int]:
        alive = set(range(self.n_nodes))
        for f in self._faults:
            r = f.get("round", f.get("from"))
            if r > round:
                break
            nodes = set(f.get("nodes", []))
            if f.get("up", False):
                alive |= nodes
            else:
                alive -= nodes
        return alive

    def get_active_clients(self, round: int) -> list[int] | None:
        alive = self._env_alive(round)
        return None if alive == set(range(self.n_nodes)) else sorted(alive)

    def get_mode(self, round: int) -> str:
        return self._committed_mode

    def notify_round_start(self, round: int, mode: str) -> None:
        active = self.get_active_clients(round)
        suffix = f"  active={active}" if active is not None else ""
        print(f"[DistArch] round={round}  mode={mode}  q={self._quorum(active)}{suffix}")

    # -- distributed control plane --------------------------------------------

    def _quorum(self, active: list[int] | None) -> int:
        if self._fixed_quorum is not None:
            return self._fixed_quorum
        n_active = self.n_nodes if active is None else len(active)
        return n_active // 2 + 1

    def _commit(self, mode: str, round: int, reason: str, nodes: list[int]) -> None:
        self._committed_mode = mode
        self._last_switch    = round
        self._stable         = 0
        self._pending_reason = {"reason": reason, "failed_nodes": sorted(nodes)}
        print(f"[DistArch] COMMIT {mode} @round={round}  reason={reason}  nodes={sorted(nodes)}")

    def observe(self, round: int, reported_nodes: set[int]) -> None:
        reported = set(reported_nodes)
        newly_down = self._believed_alive - reported
        self._believed_alive = set(reported)

        # each replicated controller updates its local view / self-protects
        for nc in self._nodes.values():
            nc.step(reported)

        # tally quorum votes among UP nodes (a down node casts no vote)
        up = [nc for nc in self._nodes.values() if nc.is_up(reported)]
        votes_gl = sum(1 for nc in up if nc.wants(reported, self.n_nodes) == "gossip")
        votes_fl = sum(1 for nc in up if nc.wants(reported, self.n_nodes) == "federated")
        active = self.get_active_clients(round)
        q = self._quorum(active)
        cooldown_ok = (round - self._last_switch) >= self._cooldown

        if self._committed_mode == "federated":
            # fail-fast: quorum of nodes self-protected to gossip on detection
            if votes_gl >= q and cooldown_ok:
                self._commit("gossip", round, "node_failure", sorted(newly_down))
        else:  # gossip -> federated, recover carefully
            if len(reported) == self.n_nodes and not newly_down:
                self._stable += 1
            else:
                self._stable = 0
            if self._stable >= self._dwell and votes_fl >= q and cooldown_ok:
                self._commit("federated", round, "recovery", [])

    def consume_switch_reason(self) -> dict | None:
        reason, self._pending_reason = self._pending_reason, None
        return reason


def load_arch_manager(conf: dict, num_nodes: int | None = None) -> ArchitectureManager:
    """Factory — instantiates the manager specified in conf/base.yaml.

    conf expected structure:
        architecture:
          manager: fixed            # fixed | distributed
          default_mode: federated
          # fixed:
          schedule:
            - {from: 1,  to: 10, mode: federated}
            - {from: 11, to: 30, mode: gossip}
          # distributed:
          initial_mode: federated
          quorum: null              # null -> majority of active nodes
          dwell_rounds: 3
          cooldown_rounds: 2
          faults:                   # environment ground truth (optional)
            - {round: 6,  nodes: [3]}
            - {round: 12, nodes: [3], up: true}
    """
    arch_conf = conf.get("architecture", {})
    kind      = arch_conf.get("manager", "fixed")

    if kind == "fixed":
        return FixedArchManager(
            schedule     = arch_conf.get("schedule", [{"from": 1, "to": 999_999, "mode": "federated"}]),
            default_mode = arch_conf.get("default_mode", "federated"),
        )

    if kind == "distributed":
        n = num_nodes or arch_conf.get("n_nodes")
        if not n:
            raise ValueError("DistributedArchManager requer num_nodes (passe ao factory "
                             "ou defina architecture.n_nodes no conf).")
        return DistributedArchManager(
            n_nodes         = int(n),
            initial_mode    = arch_conf.get("initial_mode", arch_conf.get("default_mode", "federated")),
            quorum          = arch_conf.get("quorum"),
            dwell_rounds    = int(arch_conf.get("dwell_rounds", 3)),
            cooldown_rounds = int(arch_conf.get("cooldown_rounds", 2)),
            faults          = arch_conf.get("faults"),
        )

    if kind == "api":
        raise NotImplementedError(
            "ApiArchManager foi substituído pelo monitor de saída (fd/monitor.py) "
            "e pelo DistributedArchManager. Use manager: fixed | distributed."
        )

    raise ValueError(f"manager desconhecido: '{kind}'. Use 'fixed' ou 'distributed'.")
