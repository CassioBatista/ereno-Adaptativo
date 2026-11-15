#!/usr/bin/env python3
"""Fusion rule selection repeated on the evaluation-v1 ERENO selection, on the calibration
set only (docs/EVALUATION_PROTOCOL.md, change log of 2026-11-13, consequence (1), and the
entry of 2026-11-15 that specifies this run).

Same procedure as scripts/fusion_rule_selection.py (docs/FUSION_RULE.md: candidates R0-R4,
target fused FPR 0.1 %, scenarios Full / Absence / Silent fault, hard constraints H1-H3,
decision rule, tie of 0.5 pp, simplicity order), with two changes only:

  1. Features: the evaluation-v1 ERENO selection under the best-subset retention rule
     (features/all_in_one_ev1_ereno_train_combined.json, 15 features) instead of the 24 of
     the earlier work. Specialists as in docs/FUSION_RULE.md section 2: the 14 of the
     `attack` partitioner (seed 42, benign cap 500,000; the same partition as section 2.3 of
     the evaluation protocol), each trained on 80 % of its partition (train_test_split,
     test_size 0.2, seed 42); the other 20 % of every partition is the calibration set.
  2. No test set: the ERENO test file is not read. The calibration set is split into two
     folds (StratifiedKFold, 2, shuffled, seed 42, on the class); the thresholds of a fold
     are calibrated on the benign rows of the other fold and applied to it (cross-fitting),
     and every scenario metric is computed on the pooled out-of-fold predictions. H3 bounds
     this out-of-fold false-positive rate. R0 and R1 have no threshold and are applied
     directly.

Two processes (the second does not load the dataset):
  python scripts/fusion_rule_selection_ev1.py train    # specialists + calibration scores
  python scripts/fusion_rule_selection_ev1.py          # selection (corrected H1/H2)
  ORIGINAL=1 python scripts/fusion_rule_selection_ev1.py   # H1/H2 as first registered
Out: results/fusion_rule_selection_ev1.{txt,json} (the first run, under the original H1/H2,
is kept as results/fusion_rule_selection_ev1_h12original.{txt,json}); models and scores in
results/.cache/
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import k_fusion_audit as K
import fusion_rule_selection as F

SELECTION = "features/all_in_one_ev1_ereno_train_combined.json"
MODELS = "results/.cache/fusion_ev1_models"
CALIB = "results/.cache/ereno_probs_n14_calib_ev1.npz"
FOLDS = 2
ORIGINAL = os.environ.get("ORIGINAL") == "1"
OUT = "results/fusion_rule_selection_ev1" + ("_h12original" if ORIGINAL else "")


def features():
    j = json.load(open(SELECTION))
    assert "best-subset retention rule" in j.get("method", ""), j.get("method")
    return sorted(int(f) for f in j["features"])


def phase_train():
    import xgboost as xgb
    import sklearn
    from sklearn.model_selection import train_test_split
    import python.util as util
    from fd.dataset import load_and_partition
    feats = features()
    seen = {}

    def loader(path, *a, **k):
        X, y, classes = K.lean_load_arff(path, keep=feats)
        seen["classes"] = classes
        return X, y, classes

    util.load_arff = loader
    parts, _, _ = load_and_partition(f"{K.DATASET}.csv", feats, K.N, seed=K.SEED,
                                     partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    os.makedirs(MODELS, exist_ok=True)
    spec, Xs, ys = [], [], []
    boosters = []
    for i, (X_c, y_c) in enumerate(parts):
        assert X_c.shape[1] == len(feats), X_c.shape
        atk = np.unique(y_c[y_c != nc])
        assert len(atk) == 1, atk
        spec.append(int(atk[0]))
        Xt, Xc, yt, yc = train_test_split(X_c, y_c, test_size=0.2, random_state=K.SEED)
        yy = (yt != nc).astype(int)
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = dict(K.XGB_PARAMS)
        if p and q:
            pr["scale_pos_weight"] = q / p
        b = xgb.train(pr, xgb.DMatrix(Xt, label=yy), num_boost_round=10, verbose_eval=False)
        b.save_model(f"{MODELS}/spec{i:02d}.json")
        boosters.append(b)
        Xs.append(Xc)
        ys.append(yc)
        print(f"[train] specialist {i:02d}: attack class {spec[-1]} ({seen['classes'][spec[-1]]}), "
              f"train {len(yt):,} ({p:,} attack), held out {len(yc):,}", flush=True)
    del parts
    X, y = np.concatenate(Xs), np.concatenate(ys).astype(np.int16)
    d = xgb.DMatrix(X)
    P = np.column_stack([b.predict(d).astype(np.float32) for b in boosters])
    np.savez_compressed(CALIB, P=P, y=y, nc=nc, spec=np.array(spec), cv=np.array(seen["classes"]),
                        feats=np.array(feats))
    json.dump({"nc": int(nc), "spec": spec, "features": feats, "xgboost": xgb.__version__,
               "numpy": np.__version__, "sklearn": sklearn.__version__,
               "python": sys.version.split()[0]}, open(f"{MODELS}/meta.json", "w"), indent=1)
    print(f"[calib] {len(y):,} held-out rows, {int((y == nc).sum()):,} benign -> {CALIB}", flush=True)


def kept(scen, full):
    """H1/H2 for one attack: it keeps at least 90 % of its Full recall. Corrected criterion
    (default; docs/FUSION_RULE.md, change log of 2026-11-15): an attack with zero Full recall
    is not kept, since 0 >= 0.9 x 0 would let a rule pass by not detecting it at all.
    ORIGINAL=1 in the environment applies the criterion as first registered."""
    if not ORIGINAL and full == 0:
        return False
    return scen >= F.H_KEEP * full


def crossfit(rule, P, y, nc, spec, present, folds, silent=None):
    """Out-of-fold alarms: the thresholds of fold k come from the benign of the other folds."""
    pred = np.zeros(len(y), bool)
    Pe = P if silent is None else silent
    for k in range(FOLDS):
        cal, ev = folds != k, folds == k
        thr = F.calibrate(rule, P[cal & (y == nc)], spec, present)
        pred[ev] = F.alarm(rule, Pe[ev], spec, present, thr)
    return pred


def main():
    from sklearn.model_selection import StratifiedKFold
    z = np.load(CALIB, allow_pickle=False)
    P, y, nc, spec, cv = z["P"], z["y"], int(z["nc"]), z["spec"], list(z["cv"])
    meta = json.load(open(f"{MODELS}/meta.json"))
    folds = np.zeros(len(y), np.int8)
    for k, (_, ev) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=K.SEED).split(P, y)):
        folds[ev] = k
    attacks = [int(a) for a in np.unique(spec)]
    name = {a: cv[a] for a in attacks}
    first = np.zeros(K.N, bool)
    for a in attacks:
        first[np.where(spec == a)[0].min()] = True
    full = np.ones(K.N, bool)
    Ps = P.copy()
    Ps[:, ~first] = 0.0                          # silent second specialist of each pair
    print(f"features ({len(meta['features'])}): {meta['features']}")
    print(f"environment: python {meta['python']}, xgboost {meta['xgboost']}, numpy {meta['numpy']}, "
          f"scikit-learn {meta['sklearn']}")
    print(f"calibration set {len(y):,} held-out rows ({int((y != nc).sum()):,} attack, "
          f"{int((y == nc).sum()):,} benign), {FOLDS} folds cross-fitted; target fused FPR "
          f"{100 * F.TARGET:.2f}%; the test file is not read")
    print("per attack (calibration rows): " + "  ".join(f"{name[a]} {int((y == a).sum()):,}" for a in attacks))
    res = {}
    for rule in ["R0", "R1", "R2", "R3", "R4"]:
        r = {
            "Full": F.evaluate(crossfit(rule, P, y, nc, spec, full, folds), y, nc, attacks),
            "Absence": F.evaluate(crossfit(rule, P, y, nc, spec, first, folds), y, nc, attacks),
            "Silent": F.evaluate(crossfit(rule, P, y, nc, spec, full, folds, silent=Ps), y, nc, attacks),
        }
        h1 = all(kept(r["Absence"]["per"][a], r["Full"]["per"][a]) for a in attacks)
        h2 = all(kept(r["Silent"]["per"][a], r["Full"]["per"][a]) for a in attacks)
        h3 = r["Full"]["fpr"] <= F.H_FPR
        thr = [F.calibrate(rule, P[(folds != k) & (y == nc)], spec, full) for k in range(FOLDS)]
        res[rule] = {"scen": r, "H1": h1, "H2": h2, "H3": h3,
                     "thr_by_fold": [t if not isinstance(t, dict) else {int(i): v for i, v in t.items()}
                                     for t in thr]}
        print(f"\n=== {rule} ===   H1 {'pass' if h1 else 'FAIL'}  H2 {'pass' if h2 else 'FAIL'}  "
              f"H3 {'pass' if h3 else 'FAIL'}")
        if thr[0] is not None and not isinstance(thr[0], dict):
            print("  threshold (Full) by fold: " + ", ".join(f"{t:.4f}" for t in thr))
        for s, e in r.items():
            print(f"  {s:<8} F1 {100 * e['f1']:6.2f}  recall {100 * e['recall']:6.2f}  precision "
                  f"{100 * e['precision']:6.2f}  FPR {100 * e['fpr']:.4f}% (FP {e['fp']:,})")
            print("           per attack: " + "  ".join(f"{name[a][:14]} {100 * v:5.1f}" for a, v in e["per"].items()))

    print("\n=== DECISION (docs/FUSION_RULE.md, section 5; H3 on the out-of-fold calibration benign; "
          f"H1/H2 {'as first registered' if ORIGINAL else 'corrected: zero Full recall is not kept'}) ===")
    ok = [k for k, v in res.items() if v["H1"] and v["H2"] and v["H3"]]
    if ok:
        best = max(100 * res[k]["scen"]["Full"]["recall"] for k in ok)
        tied = [k for k in ok if best - 100 * res[k]["scen"]["Full"]["recall"] <= F.TIE]
        chosen = min(tied, key=F.SIMPLICITY.index)
        print(f"rules meeting H1-H3: {ok}; best Full recall {best:.2f}%; tied within {F.TIE} pp: {tied}")
    else:
        nmet = {k: v["H1"] + v["H2"] + v["H3"] for k, v in res.items()}
        top = max(nmet.values())
        cand = [k for k in res if nmet[k] == top]
        best = max(res[k]["scen"]["Full"]["recall"] for k in cand)
        tied = [k for k in cand if 100 * (best - res[k]["scen"]["Full"]["recall"]) <= F.TIE]
        chosen = min(tied, key=F.SIMPLICITY.index)
        print(f"no rule meets H1-H3; most constraints met ({top}): {cand}")
    print(f"CHOSEN RULE: {chosen}")
    json.dump({"chosen": chosen, "features": meta["features"], "folds": FOLDS,
               "rules": {k: {kk: vv for kk, vv in v.items() if kk != "scen"} |
                         {"scen": {s: {m: e[m] for m in ("recall", "fpr", "f1", "fp")} |
                                   {"per": {name[a]: v for a, v in e["per"].items()}}
                                   for s, e in v["scen"].items()}}
                         for k, v in res.items()}},
              open(f"{OUT}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        phase_train()
        sys.exit(0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open(f"{OUT}.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
