"""ErenoFeatures — ERENO IEC-61850 dataset feature subset definitions.

58 features numericas (F1..F58) — os atributos nominais do dataset original
(MACs, IDs de protocolo, flags) sao descartados na conversao; ver
scripts/build_ereno_dataset.py e docs/DATASETS.md.

As RCLs excluem ainda os marcadores de POSICAO TEMPORAL ABSOLUTA
(F1=Time, F38=t, F39=GooseTimestamp): mesma justificativa dos nominais —
posicao na linha do tempo da simulacao e artefato do ambiente, nao
comportamento (nao generaliza; e explora o vies de blocos da CV interna).
As features temporais RELATIVAS (timestampDiff, tDiff, timeFromLastChange,
delay) permanecem — essas sao comportamentais.

Ainda nao ha subconjuntos pre-otimizados (IWSSR/GR) para este dataset:
as RCLs partem do conjunto elegivel completo — o GRASP descobre.
"""
from python.feature_subsets.base import FeatureSubsets

ERENO_ALL             = list(range(1, 59))
ERENO_TEMPO_ABSOLUTO  = [1, 38, 39]          # Time, t, GooseTimestamp
ERENO_RCL             = [f for f in ERENO_ALL if f not in ERENO_TEMPO_ABSOLUTO]


class ErenoFeatures(FeatureSubsets):
    def __init__(self):
        super().__init__(ERENO_RCL, ERENO_RCL, [ERENO_RCL] * 5)
