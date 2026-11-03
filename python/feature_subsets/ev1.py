"""Ev1Features — feature subsets of the evaluation-v1 datasets (docs/EVALUATION_PROTOCOL.md).

Built by scripts/build_ev1_datasets.py (ev1_cicids, ev1_morris3, ev1_ciciot): numeric
features F1..Fn, names in features/<ds>_feature_names.json. GRASP starts from all n
(no pre-optimized subsets).
"""
import json
import os
import re

from python.feature_subsets.base import FeatureSubsets


class Ev1Features(FeatureSubsets):
    def __init__(self, dataset: str):
        ds = re.search(r"ev1_[a-z0-9]+", dataset.lower()).group(0)
        path = os.path.join("features", f"{ds}_feature_names.json")
        n = len(json.load(open(path))["names"])
        every = list(range(1, n + 1))
        super().__init__(every, every, [every] * 5)
