"""HybridStrategy — meta-strategy that delegates to federated or gossip per round.

The ArchitectureManager decides which mode is active for each round.
On mode transitions, _transfer_model() adapts the global model state
so training continuity is preserved across paradigm switches.
"""

from __future__ import annotations

import numpy as np
from flwr.common import (FitIns, Parameters, Scalar, NDArrays, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import Strategy

from fd.arch_manager import ArchitectureManager
from fd.monitor import NullMonitor
from fd.strategy.glow_strategy import resolve_node_index, learn_cid_map, dedup_union


class _AllowedCidsCriterion(Criterion):
    """Selects only clients whose (real) proxy cid is in the allowed set."""

    def __init__(self, allowed_cids: set[str]) -> None:
        self._allowed = allowed_cids

    def select(self, client: ClientProxy) -> bool:
        return client.cid in self._allowed


FitResults     = list[tuple[ClientProxy, any]]
FitFailures    = list[tuple[ClientProxy, Exception]]
EvalResults    = list[tuple[ClientProxy, any]]
EvalFailures   = list[tuple[ClientProxy, Exception]]


class HybridStrategy(Strategy):
    """Delegates configure_fit / aggregate_fit to the active sub-strategy.

    Parameters
    ----------
    fed_strategy  : Strategy to use when mode == 'federated'
    glow_strategy : Strategy to use when mode == 'gossip'
    arch_manager  : ArchitectureManager that returns the mode for each round
    round_eval_fn : optional callback (round, mode, parameters) -> (loss, metrics) | None,
                    called server-side after each round with the aggregated
                    parameters (per-round convergence metrics)
    monitor       : optional MonitorRecorder (fd.monitor) notified on every
                    architecture change and node failure; defaults to NullMonitor
                    (no overhead). The monitor only observes (read-only).
    node_holdings : None (default, legacy behaviour) | 'retain' | 'reformed' (xgb with
                    fusion 'or' only). Per-node knowledge in BOTH modes: the gossip
                    strategy's pool_parameters hold what each node holds.
                    - gossip round: unchanged (each node keeps its pool plus what it
                      receives; the head takes the union of its neighbourhood);
                    - federated round, 'retain': the aggregate is the content-deduplicated
                      union of the participants' fresh boosters and of everything the
                      participants hold, and every participant then holds it;
                    - federated round, 'reformed': the aggregate is the participants'
                      fresh boosters only, and every participant then holds it only;
                    - nodes outside the round (inactive or in local mode) keep what they
                      hold; no state is kept by the server between rounds (a returning
                      aggregator starts from what the participants hold);
                    - nodes in local mode (outside the round's exchange, or every node
                      in a round whose mode is 'local') train their own specialist and
                      keep it, exchanging nothing;
                    - with an arrival schedule (clients) and a round time (manager
                      time_of), a node contributes nothing before its block arrives;
                    - clients always train their own specialist from scratch
                      (fresh_only): no warm-started variants.
    holdings_log  : optional fd.holdings_log.HoldingsLog; with node_holdings, what every
                    node holds is logged after every round.
                    The mode-switch transfer (_transfer_model) is not used: the per-node
                    pools already are the state carried across switches.
    """

    def __init__(
        self,
        fed_strategy:  Strategy,
        glow_strategy: Strategy,
        arch_manager:  ArchitectureManager,
        round_eval_fn=None,
        monitor=None,
        node_holdings: str | None = None,
        holdings_log=None,
    ) -> None:
        if node_holdings not in (None, "retain", "reformed"):
            raise ValueError(f"node_holdings inválido: '{node_holdings}'")
        if holdings_log is not None and node_holdings is None:
            raise ValueError("holdings_log requires node_holdings")
        self.holdings_log = holdings_log
        self._strategies  = {"federated": fed_strategy, "gossip": glow_strategy}
        self.arch_manager = arch_manager
        self.round_eval_fn = round_eval_fn
        self.node_holdings = node_holdings
        self._round_mode: dict[int, str] = {}
        self._monitor = monitor if monitor is not None else NullMonitor()
        self._prev_mode:  str | None = None
        self._last_params: Parameters | None = None
        self.final_parameters: Parameters | None = None
        # running content-deduped union of every booster ever aggregated. Because
        # specialists are deterministic (fixed seed/data) it converges to one
        # booster per node and does NOT grow. Used to seed GL at an FL->GL switch
        # so a node that failed just before the switch still has its booster in
        # the pool (models "survivors carry the network"; doc P4/§2). xgb only.
        self._retained_union: Parameters | None = None
        # proxy cid -> partition id, learned from client metrics and shared
        # with the gossip strategy (proxy cids are opaque under flwr sim)
        self.cid_map: dict[str, int] = {}
        if hasattr(glow_strategy, "cid_map"):
            glow_strategy.cid_map = self.cid_map

    # ── internal ──────────────────────────────────────────────────────────────

    def _active(self, round: int) -> tuple[str, Strategy | None]:
        mode = self.arch_manager.get_mode(round)
        if mode == "local":                      # every node local: no exchange this round
            return mode, None
        return mode, self._strategies[mode]

    def _filter_manager(
        self, client_manager: ClientManager, round: int
    ) -> ClientManager:
        """Return a view of client_manager restricted to active clients, if defined."""
        active = self.arch_manager.get_active_clients(round)
        if active is None:
            return client_manager
        return _FilteredClientManager(client_manager, active, self.cid_map)

    def _sync_participation(self, strategy: Strategy, active: list[int] | None) -> None:
        """Align the sub-strategy's expectations with the round's active clients.

        FedAvg-style strategies: min_*_clients must not exceed the number of
        active clients, or sampling fails and the round becomes a no-op.
        Originals are captured once and restored when active is None.

        GlowStrategy: node up/down status in the topology must mirror the
        active set, or head election may pick an inactive node.
        """
        if hasattr(strategy, "min_fit_clients"):
            if not hasattr(strategy, "_orig_min_clients"):
                strategy._orig_min_clients = (
                    strategy.min_fit_clients,
                    strategy.min_evaluate_clients,
                    strategy.min_available_clients,
                )
            orig_fit, orig_eval, orig_avail = strategy._orig_min_clients
            if active is None:
                strategy.min_fit_clients       = orig_fit
                strategy.min_evaluate_clients  = orig_eval
                strategy.min_available_clients = orig_avail
            else:
                n = len(active)
                strategy.min_fit_clients       = min(orig_fit, n)
                strategy.min_evaluate_clients  = min(orig_eval, n)
                strategy.min_available_clients = min(orig_avail, n)

        topology = getattr(strategy, "topology", None)
        if topology is not None:
            for node in topology.all_nodes():
                topology.set_status(node, active is None or node in active)

    def _transfer_model(
        self,
        from_mode: str,
        to_mode:   str,
        parameters: Parameters,
    ) -> Parameters:
        """Adapt global model state when switching paradigms.

        federated → gossip:
            The single global model becomes each node's local starting point.
            GlowStrategy.pool_parameters is initialized uniformly.

        gossip → federated:
            pool_parameters are averaged into a new global model,
            which becomes the initial_parameters for the federated strategy.
        """
        if from_mode == "gossip" and to_mode == "federated":
            glow = self._strategies["gossip"]
            if hasattr(glow, "pool_parameters") and glow.pool_parameters:
                agg = str(getattr(glow, "aggregation", ""))
                if agg.startswith("xgb"):
                    # XGBoost: boosters não são promediáveis — a transferência
                    # é a UNIÃO (deduplicada) do conhecimento acumulado nos nós
                    parameters = dedup_union(list(glow.pool_parameters.values()))
                    print(f"[Hybrid] gossip→federated (xgb): união do pool de "
                          f"{len(glow.pool_parameters)} nós → "
                          f"{len(parameters.tensors)} boosters")
                else:
                    arrays_list = [
                        parameters_to_ndarrays(p)
                        for p in glow.pool_parameters.values()
                    ]
                    avg: NDArrays = [
                        sum(a[i] for a in arrays_list) / len(arrays_list)
                        for i in range(len(arrays_list[0]))
                    ]
                    parameters = ndarrays_to_parameters(avg)
                    print(f"[Hybrid] gossip→federated: averaged "
                          f"{len(arrays_list)} node models")

        elif from_mode == "federated" and to_mode == "gossip":
            glow = self._strategies["gossip"]
            if hasattr(glow, "pool_parameters"):
                # semeia TODOS os nós da topologia com o modelo global —
                # iterar o pool (vazio na primeira comutação) era um no-op.
                # Para xgb, semeia da UNIÃO RETIDA (não do agregado do round, que
                # pode já ter perdido o booster do nó que acabou de cair): assim
                # um nó que falha logo antes da troca ainda tem seu especialista
                # no pool difundido (doc P4/§2, "sobreviventes carregam a rede").
                seed = parameters
                if str(getattr(glow, "aggregation", "")).startswith("xgb") \
                        and self._retained_union is not None:
                    seed = self._retained_union
                    print(f"[Hybrid] federated→gossip: seeding from RETAINED union "
                          f"({len(seed.tensors)} boosters) instead of round aggregate "
                          f"({len(parameters.tensors)})")
                nodes = (glow.topology.all_nodes()
                         if hasattr(glow, "topology") else list(glow.pool_parameters))
                for node_id in nodes:
                    glow.pool_parameters[node_id] = seed
                print(f"[Hybrid] federated→gossip: global model seeded to "
                      f"{len(nodes)} nodes")
                parameters = seed

        return parameters

    # ── Strategy interface ────────────────────────────────────────────────────

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        fed = self._strategies["federated"]
        return fed.initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters:   Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, any]]:
        mode, strategy = self._active(server_round)
        self.arch_manager.notify_round_start(server_round, mode)
        active = self.arch_manager.get_active_clients(server_round)
        self._round_mode[server_round] = mode
        if strategy is None:                     # 'local' round: no exchange; with
            return self._with_local_fits(        # node_holdings every node trains alone
                server_round, [], active, client_manager)
        self._monitor.round_start(server_round, mode, active)

        if self._prev_mode is not None and mode != self._prev_mode:
            if self.node_holdings is None:       # legacy: carry the state across the switch
                parameters = self._transfer_model(self._prev_mode, mode, parameters)
            print(f"[Hybrid] mode switch: {self._prev_mode} → {mode} at round {server_round}")
            reason_info = self.arch_manager.consume_switch_reason()
            # Nodes dropped silently (removed from participation) raise no Flower
            # failure, so the decentralized detector's knowledge of WHICH nodes
            # failed must be forwarded to the monitor explicitly — otherwise the
            # architecture_change event reports failed_nodes=[] (v2 requires "qual
            # nó apresenta falha"). node_failure() populates the monitor's window.
            failed = (reason_info or {}).get("failed_nodes") or []
            if failed:
                self._monitor.node_failure(
                    server_round, failed,
                    detail="decentralized detection quorum at FL→GL switch")
            self._monitor.architecture_change(
                server_round, self._prev_mode, mode,
                reason=(reason_info or {}).get("reason"),
                active_nodes=active)

        self._prev_mode   = mode
        self._last_params = parameters
        self._sync_participation(strategy, self.arch_manager.get_active_clients(server_round))
        # Gap 2: piggyback the decentralized control-plane digest (vote+suspicion
        # bitmaps) on the real GLow FitIns messages, when the manager exposes one.
        glow = self._strategies["gossip"]
        if hasattr(glow, "ctrl_digest_hex") and hasattr(self.arch_manager, "digest_hex"):
            glow.ctrl_digest_hex = self.arch_manager.digest_hex()
        filtered_manager  = self._filter_manager(client_manager, server_round)
        instructions = strategy.configure_fit(server_round, parameters, filtered_manager)
        return self._with_local_fits(server_round, instructions, active, client_manager)

    def _round_time(self, server_round: int) -> float | None:
        f = getattr(self.arch_manager, "time_of", None)
        return float(f(server_round)) if f is not None else None

    def _with_local_fits(self, server_round: int, instructions, active,
                         client_manager: ClientManager):
        """node_holdings: tag the exchange instructions (round time, own specialist only)
        and add a local fit for every node outside the exchange (local mode): it trains its
        own specialist and keeps it, exchanging nothing. Legacy mode: unchanged."""
        if self.node_holdings is None:
            return instructions
        t_s = self._round_time(server_round)
        extra = {"fresh_only": 1}
        if t_s is not None:
            extra["t_s"] = t_s
        out = [(proxy, FitIns(ins.parameters, {**ins.config, **extra, "local": 0}))
               for proxy, ins in instructions]
        in_exchange = {proxy.cid for proxy, _ in instructions}
        pools = self._strategies["gossip"].pool_parameters
        all_cids = list(client_manager.all().keys())
        n = max(len(all_cids), len(self.cid_map)) or 1
        empty = ndarrays_to_parameters([np.array([], dtype=np.uint8)])
        for cid, proxy in client_manager.all().items():
            if cid in in_exchange:
                continue
            node = resolve_node_index(cid, all_cids, self.cid_map, n)
            out.append((proxy, FitIns(pools.get(node) or empty,
                                      {"mode": "local", **extra, "local": 1})))
        return out

    def _resolve_failed_nodes(self, results: FitResults, failures: FitFailures) -> list[int]:
        """Identify which logical node indices failed this round.

        Two sources: (i) explicit Flower failures (proxy raised/returned an
        error), and (ii) active nodes that did not report a result (silent
        drop). Proxy cids are resolved to node indices via the shared cid_map.
        """
        all_cids = [p.cid for p, _ in results]
        all_cids += [f[0].cid for f in failures if isinstance(f, tuple)]
        n = max(len(all_cids), len(self.cid_map)) or 1

        failed: set[int] = set()
        for f in failures:
            if isinstance(f, tuple):          # (ClientProxy, Exception)
                failed.add(resolve_node_index(f[0].cid, all_cids, self.cid_map, n))
        return sorted(failed)

    def _reported_nodes(self, results: FitResults) -> set[int]:
        """Logical node indices that reported a result this round (who is up)."""
        all_cids = [p.cid for p, _ in results]
        n = max(len(all_cids), len(self.cid_map)) or 1
        return {resolve_node_index(p.cid, all_cids, self.cid_map, n) for p, _ in results}

    def aggregate_fit(
        self,
        server_round: int,
        results:      FitResults,
        failures:     FitFailures,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        learn_cid_map(self.cid_map, results)
        if failures:
            failed = self._resolve_failed_nodes(results, failures)
            if failed:
                self._monitor.node_failure(
                    server_round, failed,
                    detail=f"{len(failures)} client failure(s) in aggregate_fit")
        # aggregate THIS round with the mode it was configured/run in...
        mode, strategy = self._active(server_round)
        if self.node_holdings is not None:
            # local fits: each node keeps its own fresh specialist, nothing is exchanged
            pools = self._strategies["gossip"].pool_parameters
            all_cids = [p.cid for p, _ in results]
            n = max(len(all_cids), len(self.cid_map)) or 1
            exchange = []
            for proxy, res in results:
                if int((res.metrics or {}).get("local", 0)):
                    node = resolve_node_index(proxy.cid, all_cids, self.cid_map, n)
                    pools[node] = dedup_union([pools.get(node), res.parameters])
                else:
                    exchange.append((proxy, res))
            results = exchange
            if strategy is None or not results:
                return None, {}
        if strategy is None:
            return None, {}
        params, metrics = strategy.aggregate_fit(server_round, results, failures)
        if params is not None and mode == "federated" and self.node_holdings is not None:
            params = self._federated_holdings(params, results)
        if params is not None:
            self.final_parameters = params
            # accumulate the retained union (xgb only; deterministic -> bounded)
            glow = self._strategies["gossip"]
            if str(getattr(glow, "aggregation", "")).startswith("xgb"):
                self._retained_union = dedup_union([self._retained_union, params])
        # ...THEN feed the control plane its local detection signal, so any
        # commit takes effect on the NEXT round (1-round detection latency).
        self.arch_manager.observe(server_round, self._reported_nodes(results))
        return params, metrics

    def _federated_holdings(self, fresh: Parameters, results: FitResults) -> Parameters:
        """Federated round with per-node knowledge (node_holdings): see the class doc."""
        glow = self._strategies["gossip"]
        pools = glow.pool_parameters
        participants = sorted(self._reported_nodes(results))
        if self.node_holdings == "retain":
            agg = dedup_union([fresh] + [pools.get(i) for i in participants])
        else:
            agg = dedup_union([fresh])
        for i in participants:
            pools[i] = agg
        print(f"[Hybrid] federated ({self.node_holdings}): {len(participants)} participants "
              f"{participants} -> aggregate of {len(agg.tensors)} boosters")
        return agg

    def configure_evaluate(
        self,
        server_round: int,
        parameters:   Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, any]]:
        _, strategy = self._active(server_round)
        if strategy is None:
            return []
        self._sync_participation(strategy, self.arch_manager.get_active_clients(server_round))
        filtered_manager = self._filter_manager(client_manager, server_round)
        return strategy.configure_evaluate(server_round, parameters, filtered_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results:      EvalResults,
        failures:     EvalFailures,
    ) -> tuple[float | None, dict[str, Scalar]]:
        learn_cid_map(self.cid_map, results)
        _, strategy = self._active(server_round)
        if strategy is None:
            return None, {}
        return strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self,
        server_round: int,
        parameters:   Parameters,
    ) -> tuple[float, dict[str, Scalar]] | None:
        mode, strategy = self._active(server_round)
        if self.holdings_log is not None and server_round > 0:
            glow = self._strategies["gossip"]
            nm = getattr(self.arch_manager, "node_modes", None)
            rc = getattr(self.arch_manager, "reachable", None)
            self.holdings_log.snapshot(
                server_round, self._round_time(server_round),
                self._round_mode.get(server_round, mode),
                self.arch_manager.get_active_clients(server_round),
                nm(server_round) if nm is not None else None,
                glow.pool_parameters, glow.topology.num_nodes,
                reachable=rc(server_round) if rc is not None else None)
        if self.round_eval_fn is not None and server_round > 0:
            # label with the mode the round was actually run in (configure_fit),
            # not any post-aggregation commit that only takes effect next round
            run_mode = self._round_mode.get(server_round) or self._prev_mode or mode
            return self.round_eval_fn(server_round, run_mode, parameters)
        if strategy is None:
            return None
        return strategy.evaluate(server_round, parameters)

    # ── convenience accessors ─────────────────────────────────────────────────

    def get_fed_strategy(self) -> Strategy:
        return self._strategies["federated"]

    def get_glow_strategy(self) -> Strategy:
        return self._strategies["gossip"]


# ── FilteredClientManager ─────────────────────────────────────────────────────

class _FilteredClientManager(ClientManager):
    """Read-only view of a ClientManager restricted to a subset of node indices.

    The active ids refer to logical partition/node indices (0..N-1); each
    proxy cid is resolved to its index via the shared cid_map (learned from
    client metrics), so filtering works under flwr's simulation where proxy
    cids are opaque node ids. All mutating operations are forwarded to the
    base manager.
    """

    def __init__(
        self,
        base:       ClientManager,
        active_ids: list[int],
        cid_map:    dict[str, int] | None = None,
    ) -> None:
        self._base    = base
        self._active  = set(active_ids)
        self._cid_map = cid_map if cid_map is not None else {}

    def _allowed_cids(self) -> set[str]:
        all_cids = list(self._base.all().keys())
        n = len(all_cids)
        return {
            cid for cid in all_cids
            if resolve_node_index(cid, all_cids, self._cid_map, n) in self._active
        }

    def num_available(self) -> int:
        return len(self._allowed_cids())

    def register(self, client: ClientProxy) -> bool:
        return self._base.register(client)

    def unregister(self, client: ClientProxy) -> None:
        self._base.unregister(client)

    def all(self) -> dict[str, ClientProxy]:
        allowed = self._allowed_cids()
        return {cid: proxy for cid, proxy in self._base.all().items()
                if cid in allowed}

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        return self._base.wait_for(num_clients, timeout)

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        active_criterion = _AllowedCidsCriterion(self._allowed_cids())
        if criterion is not None:
            # combine both criteria: client must pass both
            class _Combined(Criterion):
                def __init__(self, a: Criterion, b: Criterion) -> None:
                    self._a, self._b = a, b
                def select(self, client: ClientProxy) -> bool:
                    return self._a.select(client) and self._b.select(client)
            combined = _Combined(active_criterion, criterion)
        else:
            combined = active_criterion
        return self._base.sample(num_clients, min_num_clients, combined)
