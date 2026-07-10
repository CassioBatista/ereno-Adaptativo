"""Pipeline dedicado de seleção de features — GRASP one-shot até a convergência.

Separação de pipelines: o GRASP roda UMA vez por dataset, com todas as
iterações necessárias (para quando acumular --no-improvement iterações sem
melhora global), e persiste o subconjunto vencedor em JSON. Todas as
simulações passam a carregar esse arquivo (grasp.features_file no YAML)
e NUNCA mais executam o GRASP.

Uso (via ereno.py):
    python ereno.py grasp <grasp_method> <clf_idx> <dataset_name>
                    [--sample N]           subamostra estratificada p/ avaliação
                    [--no-improvement K]   parada: K iterações sem melhora (default 15)
                    [--max-iterations M]   teto de segurança (default 1000)
                    [--out PATH]           default: features/<dataset>.json

Exemplos:
    python ereno.py grasp GR-G-VND 2 all_in_one_ereno_train --sample 150000
    python ereno.py grasp GR-G-VND 2 all_in_one_cicids_v2  --sample 150000

Este pipeline NÃO depende do Flower — é independente da simulação.
"""

import json
import os
import random
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

import python.config as config


def _get_feature_subsets(dataset: str):
    match dataset.lower():
        case s if "wsn" in s:
            from python.feature_subsets.wsn import WsnFeatures
            return WsnFeatures()
        case s if "kdd" in s:
            from python.feature_subsets.kdd import KddFeatures
            return KddFeatures()
        case s if "cicid" in s:
            from python.feature_subsets.cicids import CicidsFeatures
            return CicidsFeatures()
        case s if "swat" in s:
            from python.feature_subsets.swat import SWATFeatures
            return SWATFeatures()
        case s if "ereno" in s:
            from python.feature_subsets.ereno import ErenoFeatures
            return ErenoFeatures()
        case _:
            sys.exit(f"Dataset '{dataset}' sem subconjuntos de features definidos.")


def main(args: list[str] | None = None) -> None:
    argv = args if args is not None else sys.argv[1:]
    if len(argv) < 3:
        sys.exit(__doc__)

    grasp_method = argv[0]
    clf_idx      = int(argv[1]) - 1
    dataset      = argv[2]

    def _get(flag: str, default):
        try:
            return argv[argv.index(flag) + 1]
        except (ValueError, IndexError):
            return default

    sample     = _get("--sample", None)
    sample     = int(sample) if sample else None
    no_improve = int(_get("--no-improvement", 15))
    max_iter   = int(_get("--max-iterations", 1000))
    out_path   = _get("--out", os.path.join("features", f"{dataset}.json"))

    config.DATASET = f"{dataset}.csv"
    config.FOLDS   = 5
    random.seed(config.GRASP_SEED)   # construção GRASP reprodutível

    feature_subsets = _get_feature_subsets(dataset)

    from python.grasp.vnd import GraspVND
    from python.grasp.rvnd import GraspRVND
    from python.grasp.simple import GraspSimple

    match grasp_method.upper():
        case "GR-G-VND" | "F-G-VND" | "I-G-VND":
            grasp = GraspVND()
        case "GR-G-RVND" | "F-G-RVND":
            grasp = GraspRVND()
        case _:
            grasp = GraspSimple()

    match grasp_method.upper():
        case "F-G-VND" | "F-G-RVND":
            rcl = feature_subsets.RCL_FULL
        case "I-G-VND":
            rcl = feature_subsets.RCL_I[clf_idx]
        case _:
            rcl = feature_subsets.RCL_GR

    grasp.setup_grasp_microservice(clf_idx)

    if sample and grasp._all_instances is not None:
        X_all, y_all = grasp._all_instances
        if sample < len(y_all):
            idx, _ = train_test_split(
                np.arange(len(y_all)), train_size=sample,
                random_state=config.GRASP_SEED, stratify=y_all,
            )
            grasp._all_instances = (X_all[idx], y_all[idx])
            print(f"[GRASP] subamostra estratificada: {sample:,} de {len(y_all):,}")

    # convergência: para após K iterações sem melhora do melhor global
    grasp.max_no_improvement = no_improve
    grasp.max_iterations     = max_iter

    print(f"[GRASP] {grasp_method} clf={clf_idx + 1} dataset={dataset} "
          f"| parada: {no_improve} iterações sem melhora (teto {max_iter})")
    t0   = time.time()
    best = grasp.run(rcl, grasp_method, dataset)
    dt   = time.time() - t0

    features = sorted(best.get_array_features())
    f1       = float(best.evaluation.f1score) if best.evaluation else None

    payload = {
        "dataset":        dataset,
        "grasp_method":   grasp_method,
        "classifier_idx": clf_idx + 1,
        "features":       features,
        "f1_grasp_cv":    f1,
        "iterations":     grasp.iteration_number,
        "evaluations":    grasp.number_evaluation,
        "no_improvement": no_improve,
        "sample":         sample,
        "seed":           config.GRASP_SEED,
        "elapsed_s":      round(dt, 1),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n[GRASP] convergiu em {grasp.iteration_number} iterações "
          f"({grasp.number_evaluation} avaliações, {dt / 60:.1f} min)")
    print(f"[GRASP] features ({len(features)}): {features}  f1_cv={f1}")
    print(f"[GRASP] persistido em: {out_path}")
    print(f"\nPara usar em TODAS as simulações, no YAML:")
    print(f"  grasp:\n    features_file: {out_path}")


if __name__ == "__main__":
    main()
