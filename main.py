"""Main entry point — mirrors Grasp.main() in Java.

Usage:
  python main.py <grasp_algorithm> <classifier_index> <dataset_name>
  python main.py GR-G-BF 2 all_in_one_wsn

grasp_algorithm : GR-G-BF | GR-G-VND | GR-G-RVND | F-G-VND | F-G-RVND | I-G-VND
classifier_index: 1=RandomTree 2=J48 3=REPTree 4=NaiveBayes 5=RandomForest 6=XGBoost
dataset_name    : filename without .csv (must be an ARFF file in working directory)
"""
from __future__ import annotations

import sys
import python.config as config
from python.classifiers import all_classifiers


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
        print(f"Invalid dataset name '{dataset_name}'. Must contain wsn, kdd, cicids, or swat.")
        sys.exit(1)


def show_options(grasp_algorithm: str, classifier_index: int, dataset_name: str) -> None:
    from python.feature_subsets.base import FeatureSubsets

    feature_subsets = _get_feature_subsets(dataset_name)
    config.DATASET = dataset_name + ".csv"
    config.SINGLE_CLASSIFIER_MODE = all_classifiers[classifier_index]
    config.CLASSIFIERS_FOREACH = all_classifiers

    if grasp_algorithm == config.GRASP_METHOD[0]:  # GR-G-BF
        FeatureSubsets.RCL = feature_subsets.RCL_GR
        from python.grasp.simple import GraspSimple
        grasp = GraspSimple()
        grasp.setup_grasp_microservice(classifier_index)
        grasp.run(FeatureSubsets.RCL_GR, grasp_algorithm, "BIT_FLIP", dataset_name)

    elif grasp_algorithm == config.GRASP_METHOD[1]:  # GR-G-VND
        from python.grasp.vnd import GraspVND
        grasp = GraspVND()
        grasp.setup_grasp_microservice(classifier_index)
        grasp.run(feature_subsets.RCL_GR, grasp_algorithm, dataset_name)

    elif grasp_algorithm == config.GRASP_METHOD[2]:  # GR-G-RVND
        from python.grasp.rvnd import GraspRVND
        grasp = GraspRVND()
        grasp.setup_grasp_microservice(classifier_index)
        grasp.run(feature_subsets.RCL_GR, grasp_algorithm, dataset_name)

    elif grasp_algorithm == config.GRASP_METHOD[3]:  # F-G-VND
        from python.grasp.vnd import GraspVND
        grasp = GraspVND()
        grasp.setup_grasp_microservice(classifier_index)
        grasp.run(feature_subsets.RCL_FULL, grasp_algorithm, dataset_name)

    elif grasp_algorithm == config.GRASP_METHOD[4]:  # F-G-RVND
        from python.grasp.rvnd import GraspRVND
        grasp = GraspRVND()
        grasp.setup_grasp_microservice(classifier_index)
        grasp.run(feature_subsets.RCL_FULL, grasp_algorithm, dataset_name)

    elif grasp_algorithm == config.GRASP_METHOD[5]:  # I-G-VND
        from python.grasp.vnd import GraspVND
        grasp = GraspVND()
        grasp.setup_grasp_microservice(classifier_index)
        grasp.run(feature_subsets.RCL_I[classifier_index], grasp_algorithm, dataset_name)

    else:
        print(f"Opção inválida: {grasp_algorithm}")
        print(f"Valid options: {config.GRASP_METHOD}")


def main(args: list[str]) -> None:
    if len(args) == 3:
        grasp_algorithm = args[0]
        classifier_index = int(args[1]) - 1
        dataset_name = args[2]
        config.FOLDS = 5
        show_options(grasp_algorithm, classifier_index, dataset_name)

    elif len(args) == 4:
        grasp_algorithm = args[0]
        classifier_index = int(args[1]) - 1
        dataset_name = args[2]
        config.FOLDS = int(args[3])
        show_options(grasp_algorithm, classifier_index, dataset_name)

    elif len(args) == 0:
        print("Please select the experimentation type:")
        for m in config.GRASP_METHOD:
            print(f"  {m}")
        grasp_algorithm = input("Algorithm: ").strip()
        dataset_name = input("Dataset name (without .csv): ").strip()
        classifier_index = -1
        show_options(grasp_algorithm, classifier_index, dataset_name)

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
