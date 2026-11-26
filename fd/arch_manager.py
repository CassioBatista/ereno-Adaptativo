"""Architecture manager — decides whether each round runs as 'federated' or 'gossip'.

Three managers:
  * TrustLevelArchManager — each node's mode from the trust level of its Disaster-FD
    monitor (API 1.9.0, docs/TRUST_LEVEL.md), on the telemetry of a run.
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

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fd.topology import Topology


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
    topology       : OPTIONAL agent graph (ring/chain/star/...). When given, the
                     manager runs the DECENTRALIZED detection path (Gap 1): each
                     node times out its silent neighbours locally and suspicions
                     + mode votes DIFFUSE over the GLow substrate, so the switch
                     commit reflects the real diffusion LATENCY instead of an
                     instantaneous central tally. Absent -> legacy central path.
    peer_timeout   : rounds of silence before a neighbour is locally suspected
                     (decentralized path only).
    min_witnesses  : independent suspectors required to accept a peer down
                     (decentralized path only; 1 = crash-trusted, v2).
    """

    def __init__(
        self,
        n_nodes: int,
        initial_mode: str = "federated",
        quorum: int | None = None,
        dwell_rounds: int = 3,
        cooldown_rounds: int = 2,
        faults: list[dict] | None = None,
        topology: "Topology | None" = None,
        peer_timeout: int = 2,
        min_witnesses: int = 1,
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

        # -- decentralized control plane (Gap 1) --------------------------------
        self._topo          = topology
        self._peer_timeout  = max(1, int(peer_timeout))
        self._min_witnesses = max(1, int(min_witnesses))
        self._detector = self._vote = None
        self.ctrl_bytes     = 0                   # cumulative control-plane bytes (Gap 2)
        self._node_stable   = {i: 0 for i in range(n_nodes)}  # per-node local dwell (recovery)
        if topology is not None:
            from fd.peer_failure import PeerFailureDetector
            from fd.vote_diffusion import VoteDiffusion
            self._detector = PeerFailureDetector(
                topology, timeout=self._peer_timeout, min_witnesses=self._min_witnesses)
            self._vote = VoteDiffusion(topology)

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
        if self._detector is not None:                 # Gap 1: decentralized path
            self._observe_decentralized(round, set(reported_nodes))
        else:
            self._observe_central(round, set(reported_nodes))

    # -- legacy central-oracle path (topology is None) -------------------------

    def _observe_central(self, round: int, reported: set[int]) -> None:
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

    # -- decentralized path (Gap 1): local timeout + GLow diffusion ------------

    def _observe_decentralized(self, round: int, alive: set[int]) -> None:
        """Detection/decision emerge from local timeouts + diffusion, not a
        global tally. `alive` is the environment's participation ground truth
        (who answered the aggregator); the CONTROL decision, however, only fires
        once suspicions/votes have DIFFUSED to a quorum -> realistic latency."""
        newly_down = self._believed_alive - alive
        self._believed_alive = set(alive)

        # topology up/down status mirrors participation (heads/diffusion skip down)
        for n in self._topo.all_nodes():
            self._topo.set_status(n, n in alive)

        # (1) local per-neighbour timeout + GLow diffusion of suspicions
        self._detector.observe_and_step(round, alive)

        # (2) control-plane digest rides EVERY GLow message this round (Gap 2):
        #     vote bitmap + suspicion bitmap = 2*ceil(N/8) B per neighbour->head
        active = [n for n in self._topo.all_nodes() if n in alive]
        if active:
            head = active[(round - 1) % len(active)]
            self.ctrl_bytes += len(self._topo.up_neighbors(head)) * self._digest_size()

        q = self._quorum(sorted(alive))
        cooldown_ok = (round - self._last_switch) >= self._cooldown
        down = set(range(self.n_nodes)) - alive

        if self._committed_mode == "federated":
            # FAIL-FAST (no network consensus): commit to gossip as soon as a
            # down peer is LOCALLY corroborated by min_witnesses independent
            # observers (<= degree(j)) — knows_down() already encodes that gate.
            # A trusted crash needs no quorum: self-protection is per-node and the
            # global switch fires on first corroborated detection (latency ~= T).
            # The NETWORK quorum is reserved for careful GL->FL recovery and for
            # the Byzantine regime (v3), where detections can be false.
            if cooldown_ok and any(self._detector.knows_down(i, j)
                                   for j in down for i in alive):
                # report the peers currently down (the failure may have occurred a
                # couple rounds earlier, so newly_down at commit can be empty)
                self._commit("gossip", round, "node_failure", sorted(down))
                self._reset_vote()
        else:  # gossip -> federated recovery — FULLY DECENTRALIZED (sound)
            # No global "len(alive)==N" check. Each node votes on its OWN DIRECT
            # observation ("my neighbourhood is fully alive"), and recovery commits
            # only on UNANIMITY (q = N): every node must assert local health, which
            # holds iff the whole overlay is intact (full membership restored).
            #   Why unanimity, not a majority: a crashed node PARTITIONS the ring
            #   (dead nodes don't relay gossip), so "nobody is down" can't be learned
            #   by diffusion — a far node's ignorance is indistinguishable from
            #   health. Unanimity of LOCAL health sidesteps this: a down node's
            #   neighbours have an incomplete neighbourhood and withhold their vote,
            #   so a partial membership can NEVER reach N votes -> no false recovery.
            # Per-node LOCAL dwell: each node counts its own neighbourhood stable.
            if newly_down:
                self._reset_vote()                       # a fresh drop cancels recovery
            for i in self._topo.all_nodes():
                healthy = i in alive and all(nb in alive
                                             for nb in self._topo.neighbors(i))
                self._node_stable[i] = self._node_stable[i] + 1 if healthy else 0
            for i in alive:
                if self._node_stable[i] >= self._dwell:
                    self._vote.cast(i)                   # i asserts: my neighbourhood is whole
            self._vote.step(round)
            # recover only when ALL N nodes have asserted local health (unanimity)
            if cooldown_ok and self._vote_quorum(alive, self.n_nodes):
                self._commit("federated", round, "recovery", [])
                self._reset_vote()

    def _vote_quorum(self, alive: set[int], q: int) -> bool:
        """Decentralized quorum: some up node has heard >= q distinct votes."""
        return any(self._vote.count(i) >= q for i in alive)

    def _reset_vote(self) -> None:
        from fd.vote_diffusion import VoteDiffusion
        self._vote = VoteDiffusion(self._topo)

    def _digest_size(self) -> int:
        """Control-plane digest per GLow message: vote bitmap + suspicion bitmap."""
        return 2 * math.ceil(self.n_nodes / 8)

    def _bitmap(self, ids: set[int]) -> bytes:
        b = bytearray(math.ceil(self.n_nodes / 8))
        for i in ids:
            if 0 <= i < self.n_nodes:
                b[i // 8] |= 1 << (i % 8)
        return bytes(b)

    def digest_hex(self) -> str:
        """Current control-plane digest (vote bitmap ++ suspicion bitmap) as hex,
        for piggybacking on the real GLow FitIns messages (Gap 2)."""
        if self._detector is None or self._vote is None:
            return ""
        voters = {i for i in range(self.n_nodes) if i in self._vote.heard.get(i, ())}
        suspected = {j for j in range(self.n_nodes)
                     if any(self._detector.knows_down(i, j)
                            for i in self._believed_alive)}
        return (self._bitmap(voters) + self._bitmap(suspected)).hex()

    def consume_switch_reason(self) -> dict | None:
        reason, self._pending_reason = self._pending_reason, None
        return reason


# ── Trust level (API 1.9.0) — each node decides its own mode from its monitor's TL ──

class TrustLevelArchManager(ArchitectureManager):
    """Mode of every node from the trust level of its colocated Disaster-FD monitor.

    Each node applies the agent policy of docs/TRUST_LEVEL.md to the telemetry of a
    Disaster-FD run (fd/trust_level.py): local below T_GL, gossip from T_GL, federated from
    T_FL, a new band committed after the settle time S, local after F seconds without a TL.
    Round r is the learning round at r * delta seconds after run_t0.

    The Flower simulation runs one data-plane mode per round. A round therefore runs the
    non-local mode that the nodes hold at that time, with those nodes as the active clients;
    the nodes in local mode take no part and keep what they hold (HybridStrategy with
    node_holdings). A round in which every node is local runs nothing ("local"). If, during
    a switch, some nodes are already federated while others are still in gossip, the round
    runs the mode held by more nodes (ties: federated) and the other nodes skip it; such
    rounds are listed in `conflicts`.
    """

    uses_gossip = True
    POLICIES = ("switching", "fixed_local", "fixed_gossip", "fixed_federated", "failover_federated")

    def __init__(self, run: str, t_end: float, delta_s: float = 60.0,
                 settle: float = 5.0, stale: float = 15.0, gl: float = 30.0,
                 fl: float = 50.0, n_nodes: int | None = None,
                 policy: str = "switching", links: str = "monitor",
                 round_offset_s: float = 0.0, gl_peers: str = "any",
                 election_rounds: int = 1, quorum: str = "majority") -> None:
        """policy: 'switching' (ReSIDS: each node's mode from its trust level) or a fixed
        mode for every node, which ignores the trust level: the counterfactuals of
        evaluation-v1 section 9. Under a fixed policy the failures still happen: a round
        takes the nodes that can reach its counterpart (federated: the aggregator; gossip:
        at least one ring neighbour), read from the probes of the same run (LinkStates);
        with no such node the round exchanges nothing.
        links: 'monitor' (a ring link is up for a node when its own monitor gets answers) or
        'scenario' (up only when both ends answer each other: the network of the scenario,
        the same constraint for every policy; see LinkStates).
        round_offset_s: round r runs at r * delta + offset. The scenario events fall on
        multiples of 15 min (with about 10 s of jitter); an offset of delta / 2 keeps every
        round away from them, so a round never samples a network in transition and an event
        at 30 min reaches the first round of the new phase.
        gl_peers: which nodes take part in a round of fixed GL. 'any' (default): a node with
        at least one ring link up. 'both': a node with both ring links up, the condition of GL
        in the scenario documents and of the gossip band of ReSIDS (trust level 30 = both
        neighbours), so that fixed GL and ReSIDS gossip with the same nodes.
        failover_federated (baseline of aggregator re-election, added 2026-11-26): federated
        learning that elects a new aggregator among the nodes when the original one is lost,
        as RAFT-based failover does. While some node reaches the original aggregator, the round
        federates those nodes, as fixed FL. Otherwise the largest set of nodes connected by up
        ring links elects one of them as aggregator, if it holds at least `quorum` nodes
        ('majority': more than half of all nodes, as RAFT requires; 'none': any two), once the
        original aggregator has been unreachable for more than `election_rounds` rounds; the
        round then federates that set, its members reaching the elected node over the ring
        links (multi-hop relay assumed). Below the quorum, or during the election, no node
        exchanges. One federated group per round; the cost of consensus is not modelled."""
        from fd.trust_level import LinkStates, NodeModes
        if quorum not in ("majority", "none"):
            raise ValueError(f"quorum inválido: '{quorum}'. Use 'majority' | 'none'.")
        self.election_rounds = max(0, int(election_rounds))
        self.quorum = quorum
        if policy not in self.POLICIES:
            raise ValueError(f"policy inválida: '{policy}'. Use {self.POLICIES}.")
        if links not in ("monitor", "scenario"):
            raise ValueError(f"links inválido: '{links}'. Use 'monitor' | 'scenario'.")
        if gl_peers not in ("any", "both"):
            raise ValueError(f"gl_peers inválido: '{gl_peers}'. Use 'any' | 'both'.")
        self.policy = policy
        self.link_model = links
        self.gl_peers = gl_peers
        self.modes = NodeModes(run, t_end, settle=settle, stale=stale, gl=gl, fl=fl)
        self.links = LinkStates(self.modes.trace, stale=stale, symmetric=(links == "scenario"))
        if n_nodes is not None and int(n_nodes) != self.modes.n_nodes:
            raise ValueError(f"the trace has {self.modes.n_nodes} monitors, the run "
                             f"{n_nodes} nodes")
        self.delta = float(delta_s)
        self.t_end = float(t_end)
        self.offset = float(round_offset_s)
        # rounds at r * delta + offset < t_end, r >= 1
        self.n_rounds = int(np.ceil((self.t_end - self.offset) / self.delta)) - 1
        self.default_mode = "local"
        self.conflicts: list[dict] = []
        self._cache: dict[int, tuple[str, list[int], list[str]]] = {}
        print(f"[TLArch] trace {run}: {self.modes.n_nodes} monitors, run_t0 "
              f"{self.modes.t0:.2f} ({self.modes.t0_source}); policy '{self.policy}', "
              f"links '{self.link_model}', gl_peers '{self.gl_peers}', "
              + (f"election_rounds {self.election_rounds}, quorum '{self.quorum}', "
                 if self.policy == "failover_federated" else "") +
              f"TL policy {self.modes.params}; delta {self.delta:g} s, offset {self.offset:g} s, "
              f"{self.n_rounds} rounds")

    def time_of(self, round: int) -> float:
        return round * self.delta + self.offset

    def reachable(self, round: int) -> list[bool]:
        """Environment: whether each node reaches the aggregator or a ring neighbour."""
        t = self.time_of(round)
        return [self.links.reachable(i, t) for i in range(self.modes.n_nodes)]

    def _decide_fixed(self, round: int) -> tuple[str, list[int], list[str]]:
        t = self.time_of(round)
        n = self.modes.n_nodes
        if self.policy == "fixed_local":
            mode, nodes = "local", []
        elif self.policy == "fixed_federated":
            nodes = [i for i in range(n) if self.links.aggregator_up(i, t)]
            mode = "federated" if nodes else "local"
        elif self.policy == "failover_federated":
            nodes = [i for i in range(n) if self.links.aggregator_up(i, t)]
            if not nodes:
                comp = self._largest_component(t)
                if len(comp) >= self._quorum_size() and self._election_wait(round) > self.election_rounds:
                    nodes = comp
            mode = "federated" if nodes else "local"
        else:                                                    # fixed_gossip
            ok = self.links.both_peers_up if self.gl_peers == "both" else self.links.peer_up
            nodes = [i for i in range(n) if ok(i, t)]
            mode = "gossip" if nodes else "local"
        m = [mode if i in nodes else "local" for i in range(n)]
        return mode, nodes, m

    def _largest_component(self, t: float) -> list[int]:
        """Largest set of nodes connected by ring links that are up at t (failover baseline)."""
        n = self.modes.n_nodes
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i in range(n):
            j = (i + 1) % n
            if self.links.link_up(i, j + 1, t):
                parent[find(i)] = find(j)
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return sorted(max(groups.values(), key=lambda g: (len(g), -min(g))))

    def _quorum_size(self) -> int:
        n = self.modes.n_nodes
        return n // 2 + 1 if self.quorum == "majority" else 2

    def _election_wait(self, round: int) -> int:
        """Consecutive rounds, up to and including `round`, in which no node reaches the
        original aggregator and a set of connected nodes holds the quorum (failover baseline:
        the election starts in the first of them and takes `election_rounds` rounds)."""
        k, n = 0, self.modes.n_nodes
        while round - k >= 1:
            t = self.time_of(round - k)
            if any(self.links.aggregator_up(i, t) for i in range(n)) \
                    or len(self._largest_component(t)) < self._quorum_size():
                break
            k += 1
        return k

    def _decide(self, round: int) -> tuple[str, list[int], list[str]]:
        if round not in self._cache and self.policy != "switching":
            self._cache[round] = self._decide_fixed(round)
        if round not in self._cache:
            t = self.time_of(round)
            m = self.modes.modes(t)
            fl = [i for i, x in enumerate(m) if x == "federated"]
            gl = [i for i, x in enumerate(m) if x == "gossip"]
            if fl and gl:
                mode, nodes = ("federated", fl) if len(fl) >= len(gl) else ("gossip", gl)
                self.conflicts.append({"round": round, "t": t, "federated": fl,
                                       "gossip": gl, "run": mode})
            elif fl:
                mode, nodes = "federated", fl
            elif gl:
                mode, nodes = "gossip", gl
            else:
                mode, nodes = "local", []
            self._cache[round] = (mode, nodes, m)
        return self._cache[round]

    def get_mode(self, round: int) -> str:
        return self._decide(round)[0]

    def get_active_clients(self, round: int) -> list[int] | None:
        return list(self._decide(round)[1])

    def node_modes(self, round: int) -> list[str]:
        return list(self._decide(round)[2])

    def notify_round_start(self, round: int, mode: str) -> None:
        _, nodes, m = self._decide(round)
        conflict = any(c["round"] == round for c in self.conflicts)
        print(f"[TLArch] round={round}  t={self.time_of(round):.0f}s  mode={mode}  "
              f"nodes={''.join(x[0].upper() for x in m)}  active={nodes}"
              + ("  CONFLICT (FL and GL at once; minority skips)" if conflict else ""))


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
          # decentralized detection (Gap 1) — realistic diffusion latency:
          decentralized: true       # off -> legacy central-oracle tally
          topology: ring            # ring | chain | star (graph substrate)
          peer_timeout: 2
          min_witnesses: 1
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
        topology = None
        if arch_conf.get("decentralized"):
            from fd.topology import build_topology
            topology = build_topology(arch_conf.get("topology", "ring"), int(n))
        return DistributedArchManager(
            n_nodes         = int(n),
            initial_mode    = arch_conf.get("initial_mode", arch_conf.get("default_mode", "federated")),
            quorum          = arch_conf.get("quorum"),
            dwell_rounds    = int(arch_conf.get("dwell_rounds", 3)),
            cooldown_rounds = int(arch_conf.get("cooldown_rounds", 2)),
            faults          = arch_conf.get("faults"),
            topology        = topology,
            peer_timeout    = int(arch_conf.get("peer_timeout", 2)),
            min_witnesses   = int(arch_conf.get("min_witnesses", 1)),
        )

    if kind == "trust_level":
        # architecture:
        #   manager: trust_level
        #   trace: ~/datasets/disaster_fd_scenario2    # Disaster-FD run (CSV telemetry)
        #   t_end: 5880                                 # trace length (s)
        #   delta_s: 60                                 # learning-round period
        #   settle_s: 5, stale_s: 15, t_gl: 30, t_fl: 50   # docs/TRUST_LEVEL.md
        #   policy: switching   # | fixed_local | fixed_gossip | fixed_federated
        #   links: monitor      # | scenario (a link is up only when both ends answer)
        #   round_offset_s: 0   # rounds at r * delta + offset (delta/2 avoids event instants)
        #   gl_peers: any       # | both (fixed GL: only nodes with both ring links up)
        #   node_holdings: retain                       # required (HybridStrategy)
        if arch_conf.get("node_holdings") not in ("retain", "reformed"):
            raise ValueError("manager: trust_level requires architecture.node_holdings: "
                             "retain | reformed (per-node knowledge in both modes)")
        return TrustLevelArchManager(
            run     = arch_conf["trace"],
            t_end   = float(arch_conf["t_end"]),
            delta_s = float(arch_conf.get("delta_s", 60.0)),
            settle  = float(arch_conf.get("settle_s", 5.0)),
            stale   = float(arch_conf.get("stale_s", 15.0)),
            gl      = float(arch_conf.get("t_gl", 30.0)),
            fl      = float(arch_conf.get("t_fl", 50.0)),
            n_nodes = num_nodes,
            policy  = arch_conf.get("policy", "switching"),
            links   = arch_conf.get("links", "monitor"),
            round_offset_s = float(arch_conf.get("round_offset_s", 0.0)),
            gl_peers = arch_conf.get("gl_peers", "any"),
            election_rounds = int(arch_conf.get("election_rounds", 1)),
            quorum   = arch_conf.get("quorum", "majority"),
        )

    if kind == "api":
        raise NotImplementedError(
            "ApiArchManager foi substituído pelo monitor de saída (fd/monitor.py) "
            "e pelo DistributedArchManager. Use manager: fixed | distributed."
        )

    raise ValueError(f"manager desconhecido: '{kind}'. Use 'fixed' ou 'distributed'.")
