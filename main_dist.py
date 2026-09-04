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
                              [--partitioner iid|dirichlet|shard|exponential|linear|attack]
                              [--partitioner-arg FLOAT]
"""

from __future__ import annotations

import sys
import os
import yaml
import numpy as np

try:
    import xgboost as xgb  # exigido pelas estratégias xgb_* e pela GlowStrategy
except ImportError:
    sys.exit("Dependência ausente: xgboost. Instale com: pip install xgboost")
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.simulation import run_simulation

import python.config as config
import python.util   as util
from python.feature_subsets.wsn    import WsnFeatures
from python.feature_subsets.kdd    import KddFeatures
from python.feature_subsets.cicids import CicidsFeatures
from python.feature_subsets.swat   import SWATFeatures
from python.feature_subsets.ereno  import ErenoFeatures

from fd.dataset           import load_and_partition
from fd.server            import make_server_app
from fd.client            import make_client_app
from fd.client_xgb        import make_xgb_client_app
from fd.client_glow       import make_glow_client_app
from fd.model             import serialize_model
from fd.evaluate          import evaluate_predictions, evaluate_model
from fd.topology          import resolve_topology, Topology
from fd.arch_manager      import load_arch_manager
from fd.monitor           import build_monitor
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
        case s if "ereno" in s: return ErenoFeatures()
        case _:
            sys.exit(f"Dataset '{dataset}' sem subconjuntos de features "
                     f"definidos. Use: wsn, kdd, cicids, swat ou ereno.")


def _print_result(label: str, r) -> None:
    fpr = 100.0 * r.FP / (r.FP + r.VN) if (r.FP + r.VN) else 0.0
    print(f"\n  [{label}]")
    print(f"    F1       : {r.f1score:.4f}%")
    print(f"    Accuracy : {r.accuracy:.4f}%")
    print(f"    Precision: {r.precision:.4f}%")
    print(f"    Recall   : {r.recall:.4f}%")
    print(f"    FPR      : {fpr:.4f}%   (falsos alarmes / benignos)")
    print(f"    VP={r.VP}  VN={r.VN}  FP={r.FP}  FN={r.FN}")


def _load_conf(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_grasp(
    grasp_method: str,
    clf_idx:      int,
    dataset:      str,
    max_iterations: int = 10,
    sample:       int | None = None,
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

    # RCL conforme o método (espelha main_fd.run_grasp):
    #   GR-* → ranking Gain Ratio | F-* → todas as features | I-* → IWSSR do classificador
    match grasp_method.upper():
        case "F-G-VND" | "F-G-RVND":
            rcl = feature_subsets.RCL_FULL
        case "I-G-VND":
            rcl = feature_subsets.RCL_I[clf_idx]
        case _:
            rcl = feature_subsets.RCL_GR

    grasp.setup_grasp_microservice(clf_idx)

    # GRASP em subamostra estratificada (datasets grandes): a seleção de
    # features roda sobre a amostra; o treinamento distribuído usa tudo.
    if sample and grasp._all_instances is not None:
        X_all, y_all = grasp._all_instances
        if sample < len(y_all):
            idx, _ = train_test_split(
                np.arange(len(y_all)), train_size=sample,
                random_state=config.GRASP_SEED, stratify=y_all,
            )
            grasp._all_instances = (X_all[idx], y_all[idx])
            print(f"[GRASP] subamostra estratificada: {sample:,} de {len(y_all):,} amostras")

    grasp.max_iterations = max_iterations
    best = grasp.run(rcl, grasp_method, dataset)
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
    xgb_fusion:    str = "sum",
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
        return XgbBaggingStrategy(num_features=num_features,
                                  fusion=xgb_fusion, **common_kw)

    # ensemble (default)
    dummy = clone(base_clf)
    dummy.fit(X_tr[:20], y_tr[:20])
    return EnsembleStrategy(
        initial_parameters=ndarrays_to_parameters(serialize_model(dummy)),
        **common_kw,
    )


def _boosters_from_parameters(parameters) -> list[xgb.Booster]:
    """Deserialize XGBoost Booster(s) from Flower Parameters.

    Um tensor = modelo único (fusão 'sum'); vários tensores = conjunto de
    boosters especialistas (fusão 'or' — um por cliente/nó).
    """
    if parameters is None:
        return []
    boosters = []
    for arr in parameters_to_ndarrays(parameters):
        if arr.size == 0:
            continue
        b = xgb.Booster()
        b.load_model(bytearray(arr.tobytes()))
        boosters.append(b)
    return boosters


def _predict_boosters(boosters: list[xgb.Booster], dtest: "xgb.DMatrix",
                      k: int = 1) -> np.ndarray:
    """Predição binária por k-de-n: alarme se >= k boosters derem prob >= 0.5.
    k=1 é a união OR (idempotente); k>=2 é corroboração (não-idempotente)."""
    votes = np.sum([(b.predict(dtest) >= 0.5) for b in boosters], axis=0)
    return (votes >= k).astype(int)


def _read_class_names(path: str) -> list[str] | None:
    """Lê só a linha @attribute @class@ do ARFF (sem carregar os dados)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                low = line.lower()
                if low.startswith("@attribute") and "@class@" in low and "{" in line:
                    return [v.strip() for v in
                            line[line.index("{") + 1:line.index("}")].split(",")]
                if low.startswith("@data"):
                    break
    except OSError:
        pass
    return None


def _print_per_client_table(client_models, specialists, X_te, y_te, arch: str) -> None:
    """Avalia o modelo LOCAL de cada cliente no teste global e imprime
    cliente | classe-especialista | F1 | Recall | Precision | FPR.

    client_models[cid] é uma lista de boosters (1 = modelo único; vários =
    conjunto acumulado do nó no gossip, avaliado por união OR)."""
    dtest = xgb.DMatrix(X_te)
    print(f"\n{'='*72}")
    print(f"  Métricas por cliente — {arch} (modelo local × teste global)")
    print(f"{'='*72}")
    print(f"  {'cli':>3} {'classe-especialista':<24} {'F1':>7} {'Recall':>7} "
          f"{'Prec':>7} {'FPR':>7} {'#bst':>5}")
    for cid, boosters in enumerate(client_models):
        if not boosters:
            print(f"  {cid:>3} {specialists[cid]:<24} {'—':>7} (sem modelo)")
            continue
        y_pred = _predict_boosters(boosters, dtest)
        r = evaluate_predictions(f"cli{cid}", y_te, y_pred)
        fpr = 100.0 * r.FP / (r.FP + r.VN) if (r.FP + r.VN) else 0.0
        print(f"  {cid:>3} {specialists[cid]:<24} {r.f1score:>7.2f} "
              f"{r.recall:>7.2f} {r.precision:>7.2f} {fpr:>7.2f} {len(boosters):>5}")


def _make_round_eval_fn(strategy_name: str, X_te: np.ndarray, y_te: np.ndarray,
                        eval_sample: int | None = None, k: int = 1):
    """Server-side per-round evaluation of the aggregated model (P5).

    Prints machine-readable lines for the convergence curve:
        ROUND;<round>;<mode>;f1=<...>;recall=<...>;fpr=<...>
    Only implemented for xgb strategies (the aggregated Parameters carry
    the serialized global booster); returns None for the others.

    eval_sample: avalia os rounds numa subamostra estratificada do teste
    (economia de memória/tempo em testes grandes); a avaliação FINAL
    continua usando o teste completo.
    """
    if strategy_name not in ("xgb_bagging", "xgb_cyclic"):
        return None

    if eval_sample and eval_sample < len(y_te):
        idx, _ = train_test_split(
            np.arange(len(y_te)), train_size=eval_sample,
            random_state=42, stratify=y_te,
        )
        X_te, y_te = X_te[idx], y_te[idx]
        print(f"[round-eval] subamostra estratificada do teste: {eval_sample:,}")

    dtest = xgb.DMatrix(X_te)

    def eval_fn(server_round: int, mode: str, parameters):
        boosters = _boosters_from_parameters(parameters)
        if not boosters:
            return None
        y_pred = _predict_boosters(boosters, dtest, k=k)
        r = evaluate_predictions(f"round-{server_round}", y_te, y_pred)
        fpr = 100.0 * r.FP / (r.FP + r.VN) if (r.FP + r.VN) else 0.0
        print(f"ROUND;{server_round};{mode};f1={r.f1score:.4f};"
              f"recall={r.recall:.4f};fpr={fpr:.4f}")
        return 1.0 - r.accuracy / 100.0, {"f1": r.f1score, "fpr": fpr, "mode": mode}

    return eval_fn


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
    grasp_sample    = grasp_conf.get("sample")
    grasp_sample    = int(grasp_sample) if grasp_sample else None
    test_file       = data_conf.get("test_file")   # test set pré-definido (opcional)
    seed            = int(conf.get("seed", 42))

    # ── ① GRASP ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    fixed_features = grasp_conf.get("features")
    features_file  = grasp_conf.get("features_file")
    if fixed_features:
        # features fixadas no YAML: pula o GRASP — garante o MESMO subconjunto
        # entre execuções comparadas (sanity check V5 do plano de validação)
        features = sorted(int(f) for f in fixed_features)
        print(f"  ① GRASP pulado — features fixadas pelo conf")
    elif features_file:
        # pipeline separado: as features vêm do JSON gerado uma única vez
        # pelo GRASP one-shot (python ereno.py grasp ...) — a simulação
        # NUNCA executa seleção de features
        import json as _json
        with open(features_file) as fh:
            payload = _json.load(fh)
        features = sorted(int(f) for f in payload["features"])
        print(f"  ① GRASP bypassado — {features_file} "
              f"({payload.get('grasp_method')}, f1_cv={payload.get('f1_grasp_cv')})")
    else:
        print(f"  ① GRASP — {grasp_method}  clf={clf_idx+1}  dataset={dataset}")
        features = run_grasp(grasp_method, clf_idx, dataset, max_iterations, grasp_sample)
    print(f"  Features selecionadas ({len(features)}): {sorted(features)}")

    # ── ② Topologia ───────────────────────────────────────────────────────────
    # A topologia só é consumida pelos rounds gossip; se QUALQUER round do
    # schedule for gossip, resolve a topologia pedida (senão o modo misto
    # cairia na star implícita do federated e a fase gossip rodaria na
    # topologia errada).
    arch_manager = load_arch_manager(conf)
    initial_mode = arch_manager.get_mode(1)
    uses_gossip = (
        getattr(arch_manager, "default_mode", None) == "gossip"
        or any(e.get("mode") == "gossip"
               for e in getattr(arch_manager, "schedule", []))
    )
    topo_mode = "gossip" if uses_gossip else "federated"
    topology: Topology = resolve_topology(topo_mode, topology_arg, num_clients)
    num_clients = topology.num_nodes
    print(f"\n  ② Topologia: {topology}")

    # ── ③ Particionamento ─────────────────────────────────────────────────────
    kwargs = {}
    benign_cap = data_conf.get("benign_cap")
    if benign_cap:
        kwargs["benign_cap"] = int(benign_cap)
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

    # classe-especialista de cada cliente (rótulos multi-classe, antes da
    # binarização) — para a tabela de métricas por cliente
    class_names = _read_class_names(f"{dataset}.csv")
    nc0 = util.normal_class
    client_specialist = []
    for _, y_c in partitions:
        atk = sorted(int(c) for c in np.unique(y_c) if int(c) != nc0)
        if class_names and all(a < len(class_names) for a in atk):
            client_specialist.append("+".join(class_names[a] for a in atk) or "—")
        else:
            client_specialist.append("+".join(map(str, atk)) or "—")

    if test_file:
        # Test set pré-definido (ex.: split do autor do dataset — evita o
        # vazamento do split aleatório e respeita separação por blocos).
        nc_train = util.normal_class
        X_te_raw, y_te, _ = util.load_arff(f"{test_file}.csv")
        util.normal_class = nc_train      # o global é redefinido a cada load
        X_te = util.filter_features(X_te_raw, features)
        del X_te_raw                      # libera as colunas não usadas (~1.4 GB no ERENO)
        X_tr, y_tr = X_f, y_all           # baseline treina no train inteiro
        print(f"  test set pré-definido: {test_file}.csv ({len(y_te):,} amostras)")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_f, y_all, test_size=0.2, random_state=seed, stratify=y_all
        )
    client_splits = []
    for X_c, y_c in partitions:
        Xct, Xce, yct, yce = train_test_split(X_c, y_c, test_size=0.2, random_state=seed)
        client_splits.append((Xct, yct, Xce, yce))

    # XGBoost treina binary:logistic — binariza os rótulos (normal=0, ataque=1)
    # DEPOIS do particionamento, para os particionadores por classe (dirichlet,
    # shard) continuarem enxergando as classes originais.
    _is_xgb = strategy_name in ("xgb_bagging", "xgb_cyclic")
    if _is_xgb:
        nc = util.normal_class
        _bin = lambda y: (y != nc).astype(np.int64)
        y_tr, y_te = _bin(y_tr), _bin(y_te)
        client_splits = [
            (Xct, _bin(yct), Xce, _bin(yce))
            for Xct, yct, Xce, yce in client_splits
        ]

    # ── ④ Estratégias ─────────────────────────────────────────────────────────
    xgb_fusion  = sim_conf.get("xgb_fusion", "sum")   # sum | or
    fusion_k    = int(sim_conf.get("xgb_fusion_k", 1))  # k-de-n: alarme se >=k boosters
    base_clf    = _build_base_clf(strategy_name, seed)
    fed_strategy = _build_fed_strategy(
        strategy_name, num_clients, len(features), base_clf, X_tr, y_tr, seed,
        xgb_fusion=xgb_fusion,
    )
    glow_strategy = GlowStrategy(
        topology    = topology,
        aggregation = conf.get("architecture", {}).get("aggregation", "inplace"),
        initial_parameters = fed_strategy.initialize_parameters(None),
    )
    eval_sample = sim_conf.get("eval_sample")
    monitor = build_monitor(conf)
    hybrid = HybridStrategy(
        fed_strategy, glow_strategy, arch_manager,
        round_eval_fn=_make_round_eval_fn(
            strategy_name, X_te, y_te,
            eval_sample=int(eval_sample) if eval_sample else None, k=fusion_k),
        monitor=monitor,
    )

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
    monitor.stop()

    # ── resultado final (P1) ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Avaliação final — teste global (20%, mesmas features)")
    print(f"{'='*60}")

    result = None
    match strategy_name:
        case "xgb_bagging" | "xgb_cyclic":
            # Preferir os parâmetros agregados do último round (no modo misto,
            # o modelo final pode ter saído do gossip, não da fed_strategy).
            boosters = _boosters_from_parameters(hybrid.final_parameters)
            if not boosters and fed_strategy.get_global_model() is not None:
                boosters = [fed_strategy.get_global_model()]
            if not boosters:
                print("  [AVISO] nenhum modelo global disponível — agregação falhou.")
            else:
                y_pred = _predict_boosters(boosters, xgb.DMatrix(X_te), k=fusion_k)
                result = evaluate_predictions(strategy_name, y_te, y_pred)
                n_trees = sum(len(b.get_dump()) for b in boosters)
                fusao = f"{len(boosters)} boosters/k≥{fusion_k}, " if len(boosters) > 1 else ""
                _print_result(
                    f"DISTRIBUÍDO [{strategy_name}] ({fusao}{n_trees} árvores)", result
                )
        case "federated_nb":
            model = fed_strategy.get_global_model()
            if model is None:
                print("  [AVISO] nenhum modelo global disponível — agregação falhou.")
            else:
                result = evaluate_model(model, X_te, y_te, "FederatedNB")
                _print_result(f"DISTRIBUÍDO [{strategy_name}]", result)
        case _:
            if hasattr(fed_strategy, "majority_vote_predict"):
                y_pred = fed_strategy.majority_vote_predict(X_te)
                result = evaluate_predictions("Ensemble", y_te, y_pred)
                _print_result(f"DISTRIBUÍDO [{strategy_name}]", result)

    # ── baseline monolítico (P2): mesmas features, mesmo split 80/20 ──────────
    if result is not None:
        if _is_xgb:
            n_pos = int(y_tr.sum())
            n_neg = len(y_tr) - n_pos
            booster_c = xgb.train(
                {"objective": "binary:logistic", "eval_metric": "logloss",
                 "max_depth": 4, "eta": 0.1, "seed": seed, "nthread": 1,
                 "scale_pos_weight": n_neg / max(n_pos, 1)},
                xgb.DMatrix(X_tr, label=y_tr),
                num_boost_round=num_clients * 10,
                verbose_eval=False,
            )
            y_pred_c = (booster_c.predict(xgb.DMatrix(X_te)) >= 0.5).astype(int)
            r_central = evaluate_predictions("XGBoost-central", y_te, y_pred_c)
            _print_result("MONOLÍTICO — XGBoost (mesmas features/split)", r_central)
        else:
            clf_central = clone(base_clf)
            clf_central.fit(X_tr, y_tr)
            r_central = evaluate_model(clf_central, X_te, y_te, "central")
            _print_result("MONOLÍTICO (mesmas features/split)", r_central)

        print(f"\n{'='*60}")
        print("  Diferença  (distribuído − monolítico)")
        print(f"{'='*60}")
        for attr, label in [("f1score", "F1-score "), ("accuracy", "Accuracy "),
                            ("recall", "Recall   "), ("precision", "Precision")]:
            delta = getattr(result, attr) - getattr(r_central, attr)
            print(f"  {label}: {delta:+.4f} pp")
        _fpr = lambda r: 100.0 * r.FP / (r.FP + r.VN) if (r.FP + r.VN) else 0.0
        print(f"  FPR      : {_fpr(result) - _fpr(r_central):+.4f} pp")

    # ── métricas por cliente (FL e GL) ────────────────────────────────────────
    if _is_xgb and result is not None:
        glow = hybrid.get_glow_strategy()
        pool = getattr(glow, "pool_parameters", None)
        if pool:
            # GOSSIP: modelo evoluído de cada nó (conjunto acumulado; união OR)
            client_models = [_boosters_from_parameters(glow.get_node_model(cid))
                             for cid in range(num_clients)]
            arch_label = "GOSSIP"
        else:
            # FEDERADO: reproduz o modelo local de cada cliente (treino do zero,
            # mesmos params do XgbClient) — determinístico
            client_models = []
            for cid in range(num_clients):
                Xct, yct, _, _ = client_splits[cid]
                npos = int(np.sum(yct)); nneg = len(yct) - npos
                p = dict(xgb_params, nthread=1)
                if npos > 0 and nneg > 0:
                    p["scale_pos_weight"] = nneg / npos
                b = xgb.train(p, xgb.DMatrix(Xct, label=yct),
                              num_boost_round=10, verbose_eval=False)
                client_models.append([b])
            arch_label = "FEDERADO"
        _print_per_client_table(client_models, client_specialist, X_te, y_te, arch_label)


if __name__ == "__main__":
    main()
