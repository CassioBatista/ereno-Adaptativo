"""GRASP rigoroso por-ataque (substitui o proxy de importancia por gain).

Reusa EXATAMENTE o GRASP do repo (GraspVND, RCL = ERENO_RCL 55 features, VND,
5-fold CV, XGBoost) mudando apenas o ALVO: em vez do problema multiclasse,
cada execucao otimiza um problema BINARIO (ataque A vs normal). A uniao das
selecoes por-ataque forma o superconjunto GRASP; combinado com o global-15
gera o conjunto final de features.

Uso:
    python scripts/grasp_por_ataque.py [--sample N] [--normal-cap M]
        [--no-improvement K] [--attacks a,b,c] [--out-dir DIR] [--tag TAG]
        [--max-time-h H]   (wall-clock cap per run; default 24 h, 0 = no cap)
        [--checkpoint-dir DIR]   (one evaluation checkpoint per type, DIR/<tag>_<attack>.jsonl)

Cada ataque persiste features/por_ataque/<tag>_<attack>.json; ao final grava
features/por_ataque/<tag>_SUPERSET.json (uniao + combinado com global-15).

Reproducibility (2026-11-14): each type's run is seeded on its own, so its result does not
depend on the process, on the order of the types or on which types an interrupted run had
already finished: Python's `random` (construction, neighbourhoods, BitFlip) is seeded with
GRASP_SEED at the start of each type, and the benign/attack row draws use a NumPy generator
seeded with (GRASP_SEED, attack id). Before this change `random` was never seeded here (the
state came from the OS at process start) and the row draws shared one generator across the
types, so per-type runs made before it cannot be reproduced or resumed.
"""
import hashlib
import json
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split

import python.config as config
import python.util as util
from python.classifiers import all_classifiers
from python.grasp.vnd import GraspVND
from python.grasp.base import max_time_ms
from main_grasp import _get_feature_subsets

DATASET = "all_in_one_ereno_train"          # override with --dataset
GLOBAL15 = [5, 7, 8, 11, 19, 31, 33, 40, 41, 42, 44, 45, 50, 57, 58]   # ERENO; --global-file
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
    lam         = float(_get("--lambda", 0.0))
    out_dir     = _get("--out-dir", os.path.join("features", "por_ataque"))
    tag         = _get("--tag", "grasp")
    only        = _get("--attacks", None)
    attack_cap  = _get("--attack-cap", None)          # None | "balanced" | integer
    max_time_h  = _get("--max-time-h", None)          # None = 24 h (default); 0 = no cap
    max_time_h  = float(max_time_h) if max_time_h is not None else None
    ckpt_dir    = _get("--checkpoint-dir", None)
    global DATASET, GLOBAL15
    DATASET = _get("--dataset", DATASET)
    gfile = _get("--global-file", None)
    if gfile == "none":
        GLOBAL15 = []                  # per-category run only; union done afterwards
    elif gfile:
        GLOBAL15 = json.load(open(gfile))["features"]
    os.makedirs(out_dir, exist_ok=True)
    config.FEATURE_PENALTY = lam
    print(f"[grasp-atk] dataset={DATASET}; FEATURE_PENALTY (lambda) = {lam}; "
          f"global set ({len(GLOBAL15)}) = {GLOBAL15}")

    config.DATASET = f"{DATASET}.csv"
    config.FOLDS = 5
    config.NUM_CLASSES = 2
    config.CROSS_VALIDATION = True
    config.SINGLE_CLASSIFIER_MODE = all_classifiers[XGB_CLF_IDX]
    rcl = _get_feature_subsets(DATASET).RCL_GR

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
    per_attack = {}

    for a in atk_ids:
        aname = classes[a]
        done = os.path.join(out_dir, f"{tag}_{aname}.json")
        if os.path.exists(done):        # resume: category finished in an earlier run
            per_attack[aname] = json.load(open(done))["features"]
            print(f"[grasp-atk] {aname}: retomado de {done}", flush=True)
            continue
        rng = np.random.default_rng([config.GRASP_SEED, a])   # this type's own row draws
        a_idx = np.where(y == a)[0]
        n_sub = min(normal_cap, len(norm_idx))
        nsel = rng.choice(norm_idx, size=n_sub, replace=False)
        # --attack-cap balanced: at most as many attack rows as normal rows, so the binary
        # problem never has an attack majority (else F1 rewards "always attack"). ERENO
        # never needed it (each attack <= 39k < normal cap); default keeps old behaviour.
        if attack_cap is not None:
            cap = n_sub if attack_cap == "balanced" else int(attack_cap)
            if len(a_idx) > cap:
                a_idx = rng.choice(a_idx, size=cap, replace=False)
        idx = np.concatenate([a_idx, nsel])
        Xb = X[idx]
        yb = (y[idx] == a).astype(np.int64)   # ataque->1, normal->0
        # subamostra estratificada final (se ainda maior que --sample)
        if sample and sample < len(yb):
            keep, _ = train_test_split(np.arange(len(yb)), train_size=sample,
                                       random_state=config.GRASP_SEED, stratify=yb)
            Xb, yb = Xb[keep], yb[keep]
        util.normal_class = 0   # metrica VP/FP: normal=0, positivo=ataque(1)
        # fingerprint of the rows this type is searched on (row ids and labels)
        sel = idx if not (sample and sample < len(idx)) else idx[keep]
        rows_sha = hashlib.sha256(np.ascontiguousarray(sel).tobytes()
                                  + np.ascontiguousarray(yb).tobytes()).hexdigest()[:16]

        grasp = GraspVND()
        grasp._all_instances = (Xb, yb)
        grasp.max_no_improvement = no_improve
        grasp.max_iterations = 1000
        grasp.max_time = max_time_ms(max_time_h)
        if ckpt_dir:
            from python.grasp.checkpoint import EvalCheckpoint
            grasp.checkpoint = EvalCheckpoint(os.path.join(ckpt_dir, f"{tag}_{aname}.jsonl"), {
                "dataset": DATASET, "attack": aname, "attack_id": int(a),
                "grasp_method": "GR-G-VND(binario)", "classifier_idx": XGB_CLF_IDX + 1,
                "sample": sample, "normal_cap": normal_cap, "attack_cap": attack_cap,
                "no_improvement": no_improve, "max_iterations": 1000, "seed": config.GRASP_SEED,
                "folds": config.FOLDS, "feature_penalty": lam, "rows_sha": rows_sha})
        random.seed(config.GRASP_SEED)   # this type's search, independent of the process

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
                   "attack_cap": attack_cap,
                   "feature_penalty": lam,
                   "seed": config.GRASP_SEED, "elapsed_s": round(dt, 1),
                   "max_time_h": None if grasp.max_time == float("inf")
                                 else grasp.max_time / 3.6e6}
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
