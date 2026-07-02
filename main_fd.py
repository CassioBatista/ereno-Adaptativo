"""ERENO-FD-SF — Federated IDS com Seleção de Features (GRASP antes da federação).

Fluxo:
  1. GRASP roda centralizado no dataset completo → seleciona subconjunto de features
  2. Dataset é filtrado pelas features selecionadas e particionado entre N clientes
  3. Clientes treinam localmente; servidor agrega (ensemble, federated_nb ou XGBoost)

Usage:
    python main_fd.py <strategy> <grasp_method> <classifier_idx> <dataset_name>
                      [<num_clients>] [<partitioner> [<partitioner_arg>]]

    strategy         : ensemble | federated_nb | xgb_bagging | xgb_cyclic
    grasp_method     : GR-G-BF | GR-G-VND | GR-G-RVND | F-G-VND | F-G-RVND | I-G-VND
    classifier_idx   : 1=RandomTree  2=J48  3=REPTree  4=NaiveBayes  5=RandomForest
                       (ignorado para xgb_bagging e xgb_cyclic — usa XGBoost nativo)
    dataset_name     : dataset sem extensão  (ex: all_in_one_wsn)
    num_clients      : clientes federados  (default: 3)
    partitioner      : iid | dirichlet | shard | exponential | linear  (default: iid)
    partitioner_arg  : alpha para dirichlet (default: 0.5)
                       shards_per_client para shard (default: 2)

Exemplos:
    python main_fd.py ensemble     GR-G-VND 2 all_in_one_wsn 3
    python main_fd.py ensemble     GR-G-VND 2 all_in_one_wsn 3 dirichlet 0.3
    python main_fd.py ensemble     GR-G-VND 2 all_in_one_wsn 3 shard 2
    python main_fd.py ensemble     GR-G-VND 2 all_in_one_wsn 3 exponential
    python main_fd.py federated_nb GR-G-VND 4 all_in_one_wsn 5
    python main_fd.py xgb_bagging  GR-G-VND 2 all_in_one_wsn 3
    python main_fd.py xgb_cyclic   GR-G-VND 2 all_in_one_wsn 3
"""

import sys
import os
import numpy as np
from flwr.common import ndarrays_to_parameters
from flwr.simulation import run_simulation
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import python.config as config
import python.util   as util
from python.classifiers import all_classifiers

from fd.dataset  import load_and_partition
from fd.client   import make_client_app
from fd.server   import make_server_app
from fd.evaluate import evaluate_model, evaluate_predictions
from fd.strategy.ensemble     import EnsembleStrategy
from fd.strategy.federated_nb import FederatedNBStrategy
from fd.strategy.xgb_bagging  import XgbBaggingStrategy
from fd.strategy.xgb_cyclic   import XgbCyclicStrategy
from fd.model import nb_to_params, serialize_model


# ── feature-subset helper (replicado de main.py) ──────────────────────────

def _get_feature_subsets(dataset_name: str):
    name = dataset_name.lower()
    if "wsn" in name:
        from python.feature_subsets.wsn import WsnFeatures
        return WsnFeatures()
    elif "kdd" in name:
        from python.feature_subsets.kdd import KddFeatures
        return KddFeatures()
    elif "cicids" in name:
        from python.feature_subsets.cicids import CicidsFeatures
        return CicidsFeatures()
    elif "swat" in name:
        from python.feature_subsets.swat import SWATFeatures
        return SWATFeatures()
    else:
        sys.exit(f"Dataset inválido '{dataset_name}'. Use: wsn, kdd, cicids ou swat.")


# ── GRASP centralizado ─────────────────────────────────────────────────────

def run_grasp(grasp_method: str, clf_idx: int, dataset_name: str) -> list[int]:
    """Executa GRASP no dataset completo e retorna os índices de features selecionados."""

    feature_subsets = _get_feature_subsets(dataset_name)
    config.DATASET  = f"{dataset_name}.csv"
    config.FOLDS    = 5

    match grasp_method:
        case "GR-G-BF":
            from python.grasp.simple import GraspSimple
            grasp = GraspSimple()
            grasp.setup_grasp_microservice(clf_idx)
            best = grasp.run(feature_subsets.RCL_GR, grasp_method, "BIT_FLIP", dataset_name)

        case "GR-G-VND":
            from python.grasp.vnd import GraspVND
            grasp = GraspVND()
            grasp.setup_grasp_microservice(clf_idx)
            best = grasp.run(feature_subsets.RCL_GR, grasp_method, dataset_name)

        case "GR-G-RVND":
            from python.grasp.rvnd import GraspRVND
            grasp = GraspRVND()
            grasp.setup_grasp_microservice(clf_idx)
            best = grasp.run(feature_subsets.RCL_GR, grasp_method, dataset_name)

        case "F-G-VND":
            from python.grasp.vnd import GraspVND
            grasp = GraspVND()
            grasp.setup_grasp_microservice(clf_idx)
            best = grasp.run(feature_subsets.RCL_FULL, grasp_method, dataset_name)

        case "F-G-RVND":
            from python.grasp.rvnd import GraspRVND
            grasp = GraspRVND()
            grasp.setup_grasp_microservice(clf_idx)
            best = grasp.run(feature_subsets.RCL_FULL, grasp_method, dataset_name)

        case "I-G-VND":
            from python.grasp.vnd import GraspVND
            grasp = GraspVND()
            grasp.setup_grasp_microservice(clf_idx)
            best = grasp.run(feature_subsets.RCL_I[clf_idx], grasp_method, dataset_name)

        case _:
            sys.exit(f"Método GRASP inválido: {grasp_method}. "
                     f"Opções: {config.GRASP_METHOD}")

    return best.get_array_features()


# ── main ───────────────────────────────────────────────────────────────────

def main(args: list[str] | None = None):
    argv = args if args is not None else sys.argv[1:]

    if len(argv) < 4:
        print(__doc__)
        sys.exit(1)

    strategy_name   = argv[0].lower()
    grasp_method    = argv[1]
    clf_idx         = int(argv[2]) - 1   # 0-based index into all_classifiers
    dataset_name    = argv[3]
    num_clients     = int(argv[4])        if len(argv) > 4 else 3
    partitioner     = argv[5].lower()     if len(argv) > 5 else "iid"
    partitioner_arg = argv[6]             if len(argv) > 6 else None

    dataset_path  = f"{dataset_name}.csv"
    if not os.path.exists(dataset_path):
        sys.exit(f"Dataset não encontrado: {dataset_path}")

    _xgb_strategy = strategy_name in ("xgb_bagging", "xgb_cyclic")

    clf_ext  = all_classifiers[clf_idx]
    base_clf = clf_ext.get_classifier()

    # ── validar federated_nb ──────────────────────────────────────────────
    if strategy_name == "federated_nb":
        from sklearn.naive_bayes import GaussianNB
        if not isinstance(base_clf, GaussianNB):
            print(
                f"[AVISO] federated_nb requer NaiveBayes (classifier 4). "
                f"Classificador {clf_idx + 1} ({clf_ext.get_classifier_name()}) "
                f"substituído por GaussianNB."
            )
            base_clf = GaussianNB()

    # ── PASSO 1: GRASP — seleção de features centralizada ─────────────────
    print(f"\n{'='*60}")
    print(f"  ERENO-FD-SF")
    print(f"  PASSO 1 — Seleção de Features (GRASP centralizado)")
    print(f"  método GRASP : {grasp_method}")
    print(f"  classificador: {clf_ext.get_classifier_name()}")
    print(f"  dataset      : {dataset_name}")
    print(f"{'='*60}\n")

    selected_features = run_grasp(grasp_method, clf_idx, dataset_name)

    print(f"\n  Features selecionadas ({len(selected_features)}): {sorted(selected_features)}")

    # ── PASSO 2: filtrar dataset e particionar ────────────────────────────
    print(f"\n{'='*60}")
    # montar kwargs do particionador
    partitioner_kwargs: dict = {}
    if partitioner_arg is not None:
        match partitioner:
            case "dirichlet":
                partitioner_kwargs["alpha"] = float(partitioner_arg)
            case "shard":
                partitioner_kwargs["shards_per_client"] = int(partitioner_arg)

    print(f"  PASSO 2 — Particionamento ({num_clients} clientes, método: {partitioner}"
          + (f", arg={partitioner_arg}" if partitioner_arg else "") + ")")
    print(f"{'='*60}\n")

    partitions, X_filtered, y_all = load_and_partition(
        dataset_path, selected_features, num_clients,
        seed=config.EVALUATION_SEED,
        partitioner=partitioner,
        **partitioner_kwargs,
    )

    # Split global 80/20 (mesmo dado visto nas duas comparações)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_filtered, y_all,
        test_size=0.2, random_state=config.EVALUATION_SEED, stratify=y_all,
    )

    # Splits por cliente
    client_splits: list[tuple] = []
    for X_c, y_c in partitions:
        X_ctr, X_cte, y_ctr, y_cte = train_test_split(
            X_c, y_c, test_size=0.2, random_state=config.EVALUATION_SEED,
        )
        client_splits.append((X_ctr, y_ctr, X_cte, y_cte))

    # ── PASSO 3: federated learning ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PASSO 3 — Aprendizado Federado (estratégia: {strategy_name})")
    print(f"{'='*60}\n")

    fit_config = {"mode": strategy_name}

    match strategy_name:
        case "federated_nb":
            from sklearn.naive_bayes import GaussianNB
            _dummy = GaussianNB()
            _dummy.fit(X_tr[:20], y_tr[:20])
            strategy = FederatedNBStrategy(
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
                on_fit_config_fn=lambda _: fit_config,
                initial_parameters=ndarrays_to_parameters(nb_to_params(_dummy)),
            )
        case "xgb_bagging":
            strategy = XgbBaggingStrategy(
                num_features=X_filtered.shape[1],
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
                on_fit_config_fn=lambda _: fit_config,
                initial_parameters=ndarrays_to_parameters([np.array([], dtype=np.uint8)]),
            )
        case "xgb_cyclic":
            strategy = XgbCyclicStrategy(
                num_clients=num_clients,
                min_fit_clients=1,
                min_evaluate_clients=1,
                min_available_clients=num_clients,
                on_fit_config_fn=lambda _: fit_config,
                initial_parameters=ndarrays_to_parameters([np.array([], dtype=np.uint8)]),
            )
        case _:  # ensemble
            _dummy_clf = clone(base_clf)
            _dummy_clf.fit(X_tr[:20], y_tr[:20])
            strategy = EnsembleStrategy(
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
                on_fit_config_fn=lambda _: fit_config,
                initial_parameters=ndarrays_to_parameters(serialize_model(_dummy_clf)),
            )

    # xgb_cyclic: N rounds sequenciais (um cliente diferente por round)
    # xgb_bagging / ensemble / federated_nb: 1 round paralelo
    _num_rounds = num_clients if strategy_name == "xgb_cyclic" else 1

    if _xgb_strategy:
        from fd.client_xgb import make_xgb_client_app
        client_app = make_xgb_client_app(client_splits)
    else:
        client_app = make_client_app(client_splits, base_clf)

    server_app = make_server_app(strategy, num_rounds=_num_rounds)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
    )

    # ── avaliação ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Avaliação no conjunto de teste global")
    print(f"  features usadas: {len(selected_features)} de {X_filtered.shape[1] + len(selected_features) - X_filtered.shape[1]}")
    print(f"{'='*60}")

    match strategy_name:
        case "federated_nb":
            global_model = strategy.get_global_model()
            if global_model is None:
                sys.exit("Agregação falhou — nenhum modelo global disponível.")
            r_fed = evaluate_model(global_model, X_te, y_te, "FederatedNB")
            fed_label = "FEDERADO-SF (NaiveBayes exato)"
        case "xgb_bagging" | "xgb_cyclic":
            import xgboost as xgb
            global_booster = strategy.get_global_model()
            if global_booster is None:
                sys.exit("Agregação XGBoost falhou — nenhum modelo global disponível.")
            dtest_global = xgb.DMatrix(X_te)
            y_prob = global_booster.predict(dtest_global)
            y_pred_xgb = (y_prob >= 0.5).astype(int)
            r_fed = evaluate_predictions(strategy_name, y_te, y_pred_xgb)
            fed_label = f"FEDERADO-SF (XGBoost {strategy_name}, {num_clients} clientes)"
        case _:
            y_pred = strategy.majority_vote_predict(X_te)
            r_fed  = evaluate_predictions("Ensemble", y_te, y_pred)
            fed_label = f"FEDERADO-SF (Ensemble, {num_clients} clientes)"

    _print_result(fed_label, r_fed)

    # baseline centralizado com as mesmas features selecionadas
    if _xgb_strategy:
        import xgboost as xgb
        dtrain_c = xgb.DMatrix(X_tr, label=y_tr)
        dtest_c  = xgb.DMatrix(X_te)
        booster_c = xgb.train(
            {"objective": "binary:logistic", "eval_metric": "logloss",
             "eta": 0.1, "max_depth": 6, "seed": 42, "nthread": 1},
            dtrain_c, num_boost_round=num_clients * 10, verbose_eval=False,
        )
        y_prob_c = booster_c.predict(dtest_c)
        y_pred_c = (y_prob_c >= 0.5).astype(int)
        r_central = evaluate_predictions("XGBoost-central", y_te, y_pred_c)
        _print_result("CENTRALIZADO-SF — XGBoost", r_central)
    else:
        clf_central = clone(base_clf)
        clf_central.fit(X_tr, y_tr)
        r_central = evaluate_model(clf_central, X_te, y_te, clf_ext.get_classifier_name())
        _print_result(f"CENTRALIZADO-SF — {clf_ext.get_classifier_name()}", r_central)

    # ── delta ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Diferença  (federado − centralizado)")
    print(f"{'='*60}")
    for attr, label in [("f1score", "F1-score"), ("accuracy", "Accuracy"),
                        ("recall", "Recall   "), ("precision", "Precision")]:
        delta = getattr(r_fed, attr) - getattr(r_central, attr)
        print(f"  {label}: {delta:+.4f} pp")


# ── helpers ────────────────────────────────────────────────────────────────

def _print_result(label: str, r) -> None:
    print(f"\n  [{label}]")
    print(f"    Accuracy : {r.accuracy:.4f}%")
    print(f"    Precision: {r.precision:.4f}%")
    print(f"    Recall   : {r.recall:.4f}%")
    print(f"    F1-score : {r.f1score:.4f}%")
    print(f"    VP={r.VP}  VN={r.VN}  FP={r.FP}  FN={r.FN}")


if __name__ == "__main__":
    main()
