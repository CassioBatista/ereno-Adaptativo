"""GRASP global (multiclasse) COM penalidade de cardinalidade.

Reusa o pipeline main_grasp (mesma RCL-55, VND, 5-fold CV, XGB clf 6, sample
150k, no-improvement 15) — a UNICA mudanca vs o global-15 original e
config.FEATURE_PENALTY = lambda, tornando a selecao global parcimoniosa.
Serve para encolher o NUCLEO do conjunto combinado (o piso global-15).

Uso: python scripts/grasp_global_penalizado.py --lambda 0.2 [--sample 150000]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import python.config as config
import main_grasp


def _get(flag, default):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def main():
    lam    = float(_get("--lambda", 0.2))
    sample = _get("--sample", "150000")
    noimp  = _get("--no-improvement", "15")
    config.FEATURE_PENALTY = lam
    out = f"features/global_penalizado_L{lam}.json"
    print(f"[global-pen] FEATURE_PENALTY (lambda) = {lam} -> {out}")
    main_grasp.main(["GR-G-VND", "6", "all_in_one_ereno_train",
                     "--sample", str(sample), "--no-improvement", str(noimp),
                     "--out", out])


if __name__ == "__main__":
    main()
