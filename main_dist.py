"""Pipeline adaptativo — federado e/ou gossip com comutação por round.

Fluxo:
  ① run_grasp()           → seleciona features via GRASP
  ② load_topology()       → carrega grafo (star implícita p/ federated puro)
  ③ load_and_partition()  → particiona dataset com num_clients=topology.num_nodes
  ④ ArchitectureManager   → define modo (federated|gossip) por round
  ⑤ HybridStrategy        → delega para fed_strategy ou glow_strategy
  ⑥ run_simulation()      → executa todos os rounds

Uso via ereno.py:
  python ereno.py distributed <grasp_method> <clf_idx> <dataset_name>
                              [--conf conf/base.yaml]
                              [--strategy ensemble|federated_nb|xgb_bagging|xgb_cyclic]
                              [--topology ring|chain|star|<path.yaml>]
                              [--clients N]
                              [--rounds N]
                              [--partitioner iid|dirichlet|shard|exponential|linear]
                              [--partitioner-arg FLOAT]
"""

from __future__ import annotations

import sys
import os
import yaml
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from flwr.common import ndarrays_to_parameters
from flwr.simulation import run_simulation

import python.config as config
from python.feature_subsets.wsn    import WsnFeatures
from python.feature_subsets.kdd    import KddFeatures
from python.feature_subsets.cicids import CicidsFeatures
from python.feature_subsets.swat   import SWATFeatures

from fd.dataset           import load_and_partition
from fd.server            import make_server_app
from fd.client            import make_client_app
from fd.client_xgb        import make_xgb_client_app
from fd.client_glow       import make_glow_client_app
from fd.model             import serialize_model
from fd.evaluate          import evaluate_predictions
from fd.topology          import resolve_topology, Topology
from fd.arch_manager      import load_arch_manager
from fd.strategy.ensemble     import EnsembleStrategy
from fd.strategy.federated_nb import FederatedNBStrategy
from fd.strategy.xgb_bagging  import XgbBaggingStrategy
from fd.strategy.xgb_cyclic   import XgbCyclicStrategy
from fd.strategy.glow_strategy   import GlowStrategy
from fd.strategy.hybrid_strategy import HybridStrategy


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_feature_subsets(dataset: str):
    match dataset.lower():
        case s if "wsn"   in s: return WsnFeatures()
        case s if "kdd"   in s: return KddFeatures()
        case s if "cicid" in s: return CicidsFeatures()
        case s if "swat"  in s: return SWATFeatures()
        case _:                 return WsnFeatures()


def _print_result(label: str, r) -> None:
    print(f"\n  [{label}]")
    print(f"    F1       : {r.f1score:.4f}%")
    print(f"    Accuracy : {r.accuracy:.4f}%")
    print(f"    Precision: {r.precision:.4f}%")
    print(f"    Recall   : {r.recall:.4f}%")
    print(f"    VP={r.VP}  VN={r.VN}  FP={r.FP}  FN={r.FN}")


def _load_conf(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_grasp(
    grasp_method: str,
    clf_idx:      int,
    dataset:      str,
    max_iterations: int = 10,
) -> list[int]:
    from python.grasp.vnd  import GraspVND
    from python.grasp.rvnd import GraspRVND
    from python.grasp.simple import GraspSimple

    config.DATASET = f"{dataset}.csv"
    config.FOLDS   = 5

    feature_subsets = _get_feature_subsets(dataset)

    match grasp_method.upper():
        case "GR-G-VND" | "F-G-VND" | "I-G-VND":
            grasp = GraspVND()
        case "GR-G-RVND" | "F-G-RVND":
            grasp = GraspRVND()
        case _:
            grasp = GraspSimple()

    grasp.setup_grasp_microservice(clf_idx)
    grasp.max_iterations = max_iterations
    best = grasp.run(feature_subsets.RCL_GR, grasp_method, dataset)
    return best.get_array_features()


def _build_base_clf(strategy_name: str, seed: int = 42):
    match strategy_name:
        case "federated_nb":
            return GaussianNB()
        case "xgb_bagging" | "xgb_cyclic":
            return None   # XgbClient handles its own model
        case _:
            return DecisionTreeClassifier(criterion="entropy", random_state=seed)


def _build_fed_strategy(
    strategy_name: str,
    num_clients:   int,
    num_features:  int,
    base_clf,
    X_tr:          np.ndarray,
    y_tr:          np.ndarray,
    seed:          int = 42,
):
    fit_config   = {"mode": strategy_name}
    config_fn    = lambda _: fit_config
    common_kw    = dict(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=config_fn,
    )

    if strategy_name == "federated_nb":
        dummy = GaussianNB()
        dummy.fit(X_tr[:20], y_tr[:20])
        return FederatedNBStrategy(
            initial_parameters=ndarrays_to_parameters(serialize_model(dummy)),
            **common_kw,
        )

    if strategy_name in ("xgb_bagging", "xgb_cyclic"):
        if strategy_name == "xgb_cyclic":
            return XgbCyclicStrategy(num_clients=num_clients, **common_kw)
        return XgbBaggingStrategy(num_features=num_features, **common_kw)

    # ensemble (default)
    dummy = clone(base_clf)
    dummy.fit(X_tr[:20], y_tr[:20])
    return EnsembleStrategy(
        initial_parameters=ndarrays_to_parameters(serialize_model(dummy)),
        **common_kw,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: list[str] | None = None) -> None:
    argv = args if args is not None else sys.argv[1:]

    # ── parse positional args ─────────────────────────────────────────────────
    if len(argv) < 3:
        print("Uso: ereno.py distributed <grasp_method> <clf_idx> <dataset_name> [opções]")
        sys.exit(1)

    grasp_method = argv[0]
    clf_idx      = int(argv[1]) - 1   # 1-based → 0-based
    dataset      = argv[2]

    # ── parse optional named args ─────────────────────────────────────────────
    def _get(flag, default):
        try:
            idx = argv.index(flag)
            return argv[idx + 1]
        except (ValueError, IndexError):
            return default

    conf_path      = _get("--conf",           "conf/base.yaml")
    strategy_name  = _get("--strategy",       None)
    topology_arg   = _get("--topology",       None)
    num_clients    = _get("--clients",        None)
    num_rounds     = _get("--rounds",         None)
    partitioner    = _get("--partitioner",    None)
    partitioner_arg= _get("--partitioner-arg",None)

    # ── load conf/base.yaml (defaults, overridden by CLI) ─────────────────────
    conf = _load_conf(conf_path) if os.path.exists(conf_path) else {}
    sim_conf  = conf.get("simulation", {})
    topo_conf = conf.get("topology",   {})
    data_conf = conf.get("dataset",    {})
    grasp_conf= conf.get("grasp",      {})

    strategy_name   = strategy_name  or sim_conf.get("strategy",    "ensemble")
    topology_arg    = topology_arg   or topo_conf.get("type",        "star")
    num_clients     = int(num_clients  or topo_conf.get("num_nodes", 3))
    num_rounds      = int(num_rounds   or sim_conf.get("num_rounds", 1))
    partitioner     = partitioner    or data_conf.get("partitioner", "iid")
    partitioner_arg = partitioner_arg or data_conf.get("partitioner_arg")
    max_iterations  = int(grasp_conf.get("max_iterations", 10))
    seed            = int(conf.get("seed", 42))

    # ── ① GRASP ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ① GRASP — {grasp_method}  clf={clf_idx+1}  dataset={dataset}")
    features = run_grasp(grasp_method, clf_idx, dataset, max_iterations)
    print(f"  Features selecionadas ({len(features)}): {sorted(features)}")

    # ── ② Topologia ───────────────────────────────────────────────────────────
    # Determine initial mode to resolve topology correctly
    arch_manager = load_arch_manager(conf)
    initial_mode = arch_manager.get_mode(1)
    topology: Topology = resolve_topology(initial_mode, topology_arg, num_clients)
    num_clients = topology.num_nodes
    print(f"\n  ② Topologia: {topology}")

    # ── ③ Particionamento ─────────────────────────────────────────────────────
    kwargs = {}
    if partitioner_arg is not None:
        if partitioner == "dirichlet":
            kwargs["alpha"] = float(partitioner_arg)
        elif partitioner == "shard":
            kwargs["shards_per_client"] = int(float(partitioner_arg))

    partitions, X_f, y_all = load_and_partition(
        f"{dataset}.csv", features, num_clients, seed=seed,
        partitioner=partitioner, **kwargs,
    )
    print(f"\n  ③ Particionamento: {partitioner}  num_clients={num_clients}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_f, y_all, test_size=0.2, random_state=seed, stratify=y_all
    )
    client_splits = []
    for X_c, y_c in partitions:
        Xct, Xce, yct, yce = train_test_split(X_c, y_c, test_size=0.2, random_state=seed)
        client_splits.append((Xct, yct, Xce, yce))

    # ── ④ Estratégias ─────────────────────────────────────────────────────────
    base_clf    = _build_base_clf(strategy_name, seed)
    fed_strategy = _build_fed_strategy(
        strategy_name, num_clients, len(features), base_clf, X_tr, y_tr, seed
    )
    glow_strategy = GlowStrategy(
        topology    = topology,
        aggregation = conf.get("architecture", {}).get("aggregation", "inplace"),
        initial_parameters = fed_strategy.initialize_parameters(None),
    )
    hybrid = HybridStrategy(fed_strategy, glow_strategy, arch_manager)

    print(f"\n  ④ Strategy: {strategy_name}  |  ArchManager: {arch_manager.__class__.__name__}")
    print(f"     num_rounds={num_rounds}")

    # ── ⑤ Client app ──────────────────────────────────────────────────────────
    if strategy_name in ("xgb_bagging", "xgb_cyclic"):
        xgb_params = {
            "max_depth": 4, "eta": 0.1,
            "objective": "binary:logistic", "seed": seed,
        }
        client_app = make_xgb_client_app(client_splits, xgb_params)
    else:
        client_app = make_glow_client_app(client_splits, base_clf) \
            if initial_mode == "gossip" \
            else make_client_app(client_splits, base_clf)

    # ── ⑥ Simulação ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ⑥ run_simulation — {num_rounds} rounds  |  {num_clients} supernodes")
    print(f"{'='*60}")

    run_simulation(
        server_app    = make_server_app(hybrid, num_rounds=num_rounds),
        client_app    = client_app,
        num_supernodes= num_clients,
    )

    # ── resultado ─────────────────────────────────────────────────────────────
    if hasattr(fed_strategy, "majority_vote_predict"):
        y_pred = fed_strategy.majority_vote_predict(X_te)
        result = evaluate_predictions("Hybrid (fed_strategy)", y_te, y_pred)
        _print_result(f"Pipeline Adaptativo [{strategy_name}]", result)


if __name__ == "__main__":
    main()
