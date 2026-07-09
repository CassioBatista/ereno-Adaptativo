"""ErenoFeatures — ERENO IEC-61850 dataset feature subset definitions.

58 features numericas (F1..F58) — os atributos nominais do dataset original
(MACs, IDs de protocolo, flags) sao descartados na conversao; ver
scripts/build_ereno_dataset.py e docs/DATASETS.md.

Ainda nao ha subconjuntos pre-otimizados (IWSSR/GR) para este dataset:
todas as RCLs partem do conjunto completo — o GRASP descobre.
"""
from python.feature_subsets.base import FeatureSubsets

ERENO_FULL = list(range(1, 59))


class ErenoFeatures(FeatureSubsets):
    def __init__(self):
        super().__init__(ERENO_FULL, ERENO_FULL, [ERENO_FULL] * 5)
