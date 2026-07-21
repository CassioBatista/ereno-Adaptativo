"""GRASP rigoroso por-ataque (substitui o proxy de importancia por gain).

Reusa EXATAMENTE o GRASP do repo (GraspVND, RCL = ERENO_RCL 55 features, VND,
5-fold CV, XGBoost) mudando apenas o ALVO: em vez do problema multiclasse,
cada execucao otimiza um problema BINARIO (ataque A vs normal). A uniao das
selecoes por-ataque forma o superconjunto GRASP; combinado com o global-15
gera o conjunto final de features.

Uso:
    python scripts/grasp_por_ataque.py [--sample N] [--normal-cap M]
        [--no-improvement K] [--attacks a,b,c] [--out-dir DIR] [--tag TAG]

Cada ataque persiste features/por_ataque/<tag>_<attack>.json; ao final grava
features/por_ataque/<tag>_SUPERSET.json (uniao + combinado com global-15).
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
from python.feature_subsets.ereno import ErenoFeatures
from python.grasp.vnd import GraspVND

DATASET = "all_in_one_ereno_train"
GLOBAL15 = [5, 7, 8, 11, 19, 31, 33, 40, 41, 42, 44, 45, 50, 57, 58]
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
    out_dir     = _get("--out-dir", os.path.join("features", "por_ataque"))
    tag         = _get("--tag", "grasp")
    only        = _get("--attacks", None)
    os.makedirs(out_dir, exist_ok=True)

    config.DATASET = f"{DATASET}.csv"
    config.FOLDS = 5
    config.NUM_CLASSES = 2
    config.CROSS_VALIDATION = True
    config.SINGLE_CLASSIFIER_MODE = all_classifiers[XGB_CLF_IDX]
    rcl = ErenoFeatures().RCL_GR

    print(f"[grasp-atk] carregando {config.DATASET} uma vez...", flush=True)
    X, y, classes = util.load_arff(config.DATASET)
    nc = int(util.normal_class)   # classe normal no rotulo original
    atk_ids = sorted(int(c) for c in np.unique(y) if int(c) != nc)
    if only:
        want = set(only.split(","))
        atk_ids = [a for a in atk_ids if classes[a] in want or str(a) in want]
    print(f"[grasp-atk] normal={classes[nc]}({nc}); ataques={[classes[a] for a in atk_ids]}",
          flush=True)

    norm_idx = np.where(y == nc)[0]
    rng = np.random.default_rng(config.GRASP_SEED)
    per_attack = {}

    for a in atk_ids:
        aname = classes[a]
        a_idx = np.where(y == a)[0]
        n_sub = min(normal_cap, len(norm_idx))
        nsel = rng.choice(norm_idx, size=n_sub, replace=False)
        idx = np.concatenate([a_idx, nsel])
        Xb = X[idx]
        yb = (y[idx] == a).astype(np.int64)   # ataque->1, normal->0
        # subamostra estratificada final (se ainda maior que --sample)
        if sample and sample < len(yb):
            keep, _ = train_test_split(np.arange(len(yb)), train_size=sample,
                                       random_state=config.GRASP_SEED, stratify=yb)
            Xb, yb = Xb[keep], yb[keep]
        util.normal_class = 0   # metrica VP/FP: normal=0, positivo=ataque(1)

        grasp = GraspVND()
        grasp._all_instances = (Xb, yb)
        grasp.max_no_improvement = no_improve
        grasp.max_iterations = 1000

        print(f"\n{'='*64}\n[grasp-atk] ATAQUE {aname} ({a}) | "
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
                   "seed": config.GRASP_SEED, "elapsed_s": round(dt, 1)}
        with open(os.path.join(out_dir, f"{tag}_{aname}.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[grasp-atk] {aname}: {len(feats)} feats {feats} f1_cv={f1} "
              f"({dt/60:.1f} min, {grasp.number_evaluation} avals)", flush=True)

    superset = sorted(set().union(*per_attack.values())) if per_attack else []
    combined = sorted(set(GLOBAL15) | set(superset))
    summary = {"dataset": DATASET, "per_attack": per_attack,
               "superset_grasp": superset, "n_superset": len(superset),
               "global15": GLOBAL15, "combined_grasp": combined,
               "n_combined": len(combined), "tag": tag}
    with open(os.path.join(out_dir, f"{tag}_SUPERSET.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[grasp-atk] SUPERSET GRASP ({len(superset)}): {superset}")
    print(f"[grasp-atk] COMBINADO c/ global-15 ({len(combined)}): {combined}")
    print(f"[grasp-atk] resumo: {os.path.join(out_dir, tag + '_SUPERSET.json')}")


if __name__ == "__main__":
    main()
