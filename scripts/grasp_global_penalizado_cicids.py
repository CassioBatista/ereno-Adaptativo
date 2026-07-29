"""GRASP global (multiclasse) COM penalidade de cardinalidade — CICIDS2017.

Replica scripts/grasp_global_penalizado.py trocando o dataset para
all_in_one_cicids_v2 (main_grasp resolve CicidsFeatures automaticamente).
Produz o NUCLEO global parcimonioso que, unido ao superconjunto por-ataque,
forma o conjunto combinado do CICIDS (espelha o combinado-24 do ERENO).

Uso: python scripts/grasp_global_penalizado_cicids.py --lambda 0.05 [--sample 150000]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import python.config as config
import main_grasp

DATASET = "all_in_one_cicids_v2"


def _get(flag, default):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def main():
    lam    = float(_get("--lambda", 0.05))
    sample = _get("--sample", "150000")
    noimp  = _get("--no-improvement", "15")
    config.FEATURE_PENALTY = lam
    out = f"features/global_penalizado_cicids_L{lam}.json"
    print(f"[global-pen-cicids] FEATURE_PENALTY (lambda) = {lam} -> {out}")
    main_grasp.main(["GR-G-VND", "6", DATASET,
                     "--sample", str(sample), "--no-improvement", str(noimp),
                     "--out", out])


if __name__ == "__main__":
    main()
