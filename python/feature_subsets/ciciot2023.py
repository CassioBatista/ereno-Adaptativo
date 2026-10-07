"""Ciciot2023Features — CICIoT2023 feature subsets.

45 numeric features F1..F45 (names in features/ciciot2023_feature_names.json): the CSV
release's 46 features minus IAT, which behaves as a capture clock (dataset audit).
Built by scripts/build_ciciot2023_dataset.py. GRASP starts from all 45.
"""
from python.feature_subsets.base import FeatureSubsets

CICIOT2023_ALL = list(range(1, 46))


class Ciciot2023Features(FeatureSubsets):
    def __init__(self):
        super().__init__(CICIOT2023_ALL, CICIOT2023_ALL, [CICIOT2023_ALL] * 5)
