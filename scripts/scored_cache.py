#!/usr/bin/env python3
"""Cache of the scored ERENO test set, so analyses stop retraining 14 specialists.

Training + scoring takes ~8 min (train split 1.6 GB). Every analysis built on the
operating point (N=14, 2 specialists per attack, combined-24, seed 42) needs the same
per-sample, per-specialist firing matrix, so it is computed once and cached under
results/.cache/ (gitignored: it is derived, ~40 MB, and reproducible from this script).

The cache key is the scenario's data + model binding: change the features, the seed or
the number of specialists and a new file is written rather than a stale one read.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from scenario_events import load_scenario

CACHE_DIR = "results/.cache"


def _key(sc):
    bind = {"data": sc["data"], "model": sc["model"], "nodes": sc["nodes"]}
    return hashlib.sha256(json.dumps(bind, sort_keys=True, default=str).encode()).hexdigest()[:12]


def get_scored(scenario="conf/scenarios/availability.yaml", benign_cap=None):
    """-> ts, fired (N x n bool), y, nc, cv (class names), spec_attack, cache_path

    benign_cap overrides the scenario's training cap on benign samples; None keeps it.
    Use a value above the train benign count (e.g. 10**9) to train on ALL benign."""
    sc = load_scenario(scenario)
    if benign_cap is not None:
        sc["data"]["benign_cap"] = int(benign_cap)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = f"{CACHE_DIR}/ereno_scored_{_key(sc)}.npz"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        print(f"[cache] hit {path}")
        return (z["ts"], z["fired"], z["y"], int(z["nc"]), list(z["cv"]),
                list(z["spec_attack"]), path)
    print(f"[cache] miss -> training (writes {path})")
    from instance_streams import train_and_score
    ts, fired, y, nc, cv, spec_attack = train_and_score(sc)
    np.savez_compressed(path, ts=ts, fired=fired, y=y, nc=nc,
                        cv=np.array([str(c) for c in cv]),
                        spec_attack=np.array(spec_attack))
    return ts, fired, y, nc, [str(c) for c in cv], list(spec_attack), path
