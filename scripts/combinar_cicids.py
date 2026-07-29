"""Combina o superconjunto por-ataque + o nucleo global penalizado do CICIDS.

Espelha a montagem do combinado-24 do ERENO:
    combinado = uniao(por_ataque_superset) U global_penalizado

Le:
    features/por_ataque/cicids_SUPERSET.json  (superset_grasp)
    features/global_penalizado_cicids_L0.05.json  (nucleo global penalizado)
Grava:
    features/all_in_one_cicids_combined.json  (formato do pipeline distribuido)

Uso: python scripts/combinar_cicids.py [--penal features/global_penalizado_cicids_L0.05.json]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERSET = os.path.join(HERE, "features", "por_ataque", "cicids_SUPERSET.json")
DEF_PENAL = os.path.join(HERE, "features", "global_penalizado_cicids_L0.05.json")
OUT = os.path.join(HERE, "features", "all_in_one_cicids_combined.json")


def _get(flag, default):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def main():
    penal_path = _get("--penal", DEF_PENAL)
    with open(SUPERSET) as fh:
        sup = json.load(fh)
    with open(penal_path) as fh:
        pen = json.load(fh)

    superset = set(sup.get("superset_grasp", []))
    per_attack = sup.get("per_attack", {})
    skipped = sup.get("skipped", {})
    core = set(pen.get("features", []))

    combined = sorted(superset | core)
    payload = {
        "dataset": "all_in_one_cicids_v2",
        "grasp_method": "por-ataque(binario) U global-penalizado",
        "classifier_idx": 6,
        "features": combined,
        "n_features": len(combined),
        "superset_por_ataque": sorted(superset),
        "n_superset": len(superset),
        "global_penalizado": sorted(core),
        "n_global_penalizado": len(core),
        "penal_lambda": pen.get("feature_penalty", 0.05),
        "per_attack": per_attack,
        "skipped_por_ataque": skipped,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[combinar-cicids] por-ataque superset ({len(superset)}): {sorted(superset)}")
    print(f"[combinar-cicids] global penalizado ({len(core)}): {sorted(core)}")
    print(f"[combinar-cicids] COMBINADO ({len(combined)}): {combined}")
    print(f"[combinar-cicids] puladas: {skipped}")
    print(f"[combinar-cicids] -> {OUT}")


if __name__ == "__main__":
    main()
