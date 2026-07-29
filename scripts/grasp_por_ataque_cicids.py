"""GRASP rigoroso por-ataque para o CICIDS2017 (replica grasp_por_ataque.py).

Identico ao GRASP por-ataque do ERENO, trocando apenas o dataset e o conjunto
de features (CicidsFeatures, RCL = 78 features). Cada execucao otimiza um
problema BINARIO (ataque A vs BENIGN) com GraspVND + VND + 5-fold CV + XGBoost.
A uniao das selecoes por-ataque forma o superconjunto GRASP-CICIDS.

Classes minusculas (poucos positivos p/ CV 5-fold, ex.: sql=21, Heartbleed=11)
sao PULADAS via --min-pos e registradas em 'skipped' — suas features ficam
cobertas pelo nucleo global. Isso evita CV degenerado e horas desperdicadas.

Uso:
    python scripts/grasp_por_ataque_cicids.py [--sample N] [--normal-cap M]
        [--no-improvement K] [--min-pos P] [--attacks a,b,c]
        [--out-dir DIR] [--tag TAG]

Cada ataque persiste features/por_ataque/<tag>_<attack>.json; ao final grava
features/por_ataque/<tag>_SUPERSET.json (uniao + combinado com o global-xgb).
"""
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split

import python.config as config
import python.util as util
from python.classifiers import all_classifiers
from python.feature_subsets.cicids import CicidsFeatures
from python.grasp.vnd import GraspVND

DATASET = "all_in_one_cicids_v2"
# nucleo global de referencia (GRASP global-XGB ja computado, features/all_in_one_cicids_v2_xgb.json)
GLOBAL_CORE = [1, 3, 4, 7, 15, 25, 40, 66, 67, 68, 70, 77]
XGB_CLF_IDX = 5   # all_classifiers[5] == XGBoost


def _get(flag, default):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def main():
    sample      = int(_get("--sample", 60000))
    normal_cap  = int(_get("--normal-cap", 60000))
    no_improve  = int(_get("--no-improvement", 15))
    min_pos     = int(_get("--min-pos", 200))
    lam         = float(_get("--lambda", 0.0))
    out_dir     = _get("--out-dir", os.path.join("features", "por_ataque"))
    tag         = _get("--tag", "cicids")
    only        = _get("--attacks", None)
    os.makedirs(out_dir, exist_ok=True)
    config.FEATURE_PENALTY = lam
    print(f"[grasp-atk-cicids] FEATURE_PENALTY (lambda) = {lam}; min_pos = {min_pos}")

    config.DATASET = f"{DATASET}.csv"
    config.FOLDS = 5
    config.NUM_CLASSES = 2
    config.CROSS_VALIDATION = True
    config.SINGLE_CLASSIFIER_MODE = all_classifiers[XGB_CLF_IDX]
    rcl = CicidsFeatures().RCL_GR

    print(f"[grasp-atk-cicids] carregando {config.DATASET} uma vez...", flush=True)
    X, y, classes = util.load_arff(config.DATASET)
    nc = int(util.normal_class)   # classe normal (BENIGN) no rotulo original
    atk_ids = sorted(int(c) for c in np.unique(y) if int(c) != nc)
    if only:
        want = set(only.split(","))
        atk_ids = [a for a in atk_ids if classes[a] in want or str(a) in want]
    print(f"[grasp-atk-cicids] normal={classes[nc]}({nc}); "
          f"ataques={[classes[a] for a in atk_ids]}", flush=True)

    norm_idx = np.where(y == nc)[0]
    rng = np.random.default_rng(config.GRASP_SEED)
    per_attack = {}
    skipped = {}

    for a in atk_ids:
        aname = classes[a]
        # RESUMIVEL: se o JSON deste ataque ja existe, reaproveita e pula.
        done_path = os.path.join(out_dir, f"{tag}_{aname}.json")
        if os.path.exists(done_path):
            with open(done_path) as fh:
                prev = json.load(fh)
            per_attack[aname] = prev["features"]
            print(f"[grasp-atk-cicids] JA FEITO {aname}: {prev['features']} "
                  f"(pula, resume)", flush=True)
            continue
        a_idx = np.where(y == a)[0]
        npos = len(a_idx)
        if npos < min_pos:
            skipped[aname] = npos
            print(f"[grasp-atk-cicids] PULA {aname} ({a}): {npos} pos < {min_pos} "
                  f"(CV 5-fold inviavel)", flush=True)
            continue
        n_sub = min(normal_cap, len(norm_idx))
        nsel = rng.choice(norm_idx, size=n_sub, replace=False)
        idx = np.concatenate([a_idx, nsel])
        Xb = X[idx]
        yb = (y[idx] == a).astype(np.int64)   # ataque->1, normal->0
        if sample and sample < len(yb):
            keep, _ = train_test_split(np.arange(len(yb)), train_size=sample,
                                       random_state=config.GRASP_SEED, stratify=yb)
            Xb, yb = Xb[keep], yb[keep]
        util.normal_class = 0   # metrica VP/FP: normal=0, positivo=ataque(1)

        grasp = GraspVND()
        grasp._all_instances = (Xb, yb)
        grasp.max_no_improvement = no_improve
        grasp.max_iterations = 1000

        print(f"\n{'='*64}\n[grasp-atk-cicids] ATAQUE {aname} ({a}) | "
              f"{int((yb==1).sum())} pos / {int((yb==0).sum())} neg | "
              f"parada {no_improve} sem melhora\n{'='*64}", flush=True)
        t0 = time.time()
        best = grasp.run(rcl, "GR-G-VND", DATASET)
        dt = time.time() - t0
        feats = sorted(best.get_array_features())
        f1 = float(best.evaluation.f1score) if best.evaluation else None
        per_attack[aname] = feats
        payload = {"dataset": DATASET, "attack": aname, "attack_id": a,
                   "grasp_method": "GR-G-VND(binario)", "classifier_idx": XGB_CLF_IDX + 1,
                   "features": feats, "n_features": len(feats), "f1_grasp_cv": f1,
                   "iterations": grasp.iteration_number, "evaluations": grasp.number_evaluation,
                   "no_improvement": no_improve, "sample": sample, "normal_cap": normal_cap,
                   "feature_penalty": lam,
                   "seed": config.GRASP_SEED, "elapsed_s": round(dt, 1)}
        with open(os.path.join(out_dir, f"{tag}_{aname}.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[grasp-atk-cicids] {aname}: {len(feats)} feats {feats} f1_cv={f1} "
              f"({dt/60:.1f} min, {grasp.number_evaluation} avals)", flush=True)

    superset = sorted(set().union(*per_attack.values())) if per_attack else []
    combined = sorted(set(GLOBAL_CORE) | set(superset))
    summary = {"dataset": DATASET, "per_attack": per_attack,
               "skipped": skipped,
               "superset_grasp": superset, "n_superset": len(superset),
               "global_core": GLOBAL_CORE, "combined_grasp": combined,
               "n_combined": len(combined), "tag": tag}
    with open(os.path.join(out_dir, f"{tag}_SUPERSET.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[grasp-atk-cicids] PULADAS (poucos pos): {skipped}")
    print(f"[grasp-atk-cicids] SUPERSET GRASP ({len(superset)}): {superset}")
    print(f"[grasp-atk-cicids] COMBINADO c/ global-core ({len(combined)}): {combined}")
    print(f"[grasp-atk-cicids] resumo: {os.path.join(out_dir, tag + '_SUPERSET.json')}")


if __name__ == "__main__":
    main()
