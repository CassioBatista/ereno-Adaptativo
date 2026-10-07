"""Cicids2017cFeatures — corrected CICIDS2017 (Engelen et al.) feature subsets.

80 numeric features F1..F80 (names in features/cicids2017c_feature_names.json), left after
the audit exclusions: identifiers (id, Flow ID, IPs, ports, Timestamp) and the initial TCP
window sizes (a host fingerprint) are not features. Built by
scripts/build_cicids2017c_dataset.py. No pre-optimized subsets: GRASP starts from all 80.
"""
from python.feature_subsets.base import FeatureSubsets

CICIDS2017C_ALL = list(range(1, 81))


class Cicids2017cFeatures(FeatureSubsets):
    def __init__(self):
        super().__init__(CICIDS2017C_ALL, CICIDS2017C_ALL, [CICIDS2017C_ALL] * 5)
