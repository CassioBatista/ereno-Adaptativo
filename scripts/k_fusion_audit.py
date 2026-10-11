#!/usr/bin/env python3
"""Audit of k-of-n fusion on ERENO: does k >= 2 contribute anything beyond a threshold?

Operating point of the paper (N=14, 2 specialists per attack, 24 features, seed 42,
benign cap 500k, full test file; same protocol as scripts/redund_k2_fulltest.py). The two
specialists of an attack are trained on disjoint halves of that attack and disjoint benign
slices of the SAME distribution, so they are correlated; k >= 2 between them may act as an
ensemble threshold rather than as corroboration.

Test 1  k >= 2 (14 specialists, GL union) vs ONE specialist per attack with k = 1, at the
        same false-positive rate (threshold on the max score; both halves reported).
Test 2  mean of the two specialists of each attack (k = 1 on the max over attacks), same FPR.
Test 3  FL shrinking 14 -> 3 (first n nodes kept, as in redund_k2_fulltest.py): recall lost
        to ABSENCE (attack with no specialist left) vs to VETO (attack with one specialist
        left, which k >= 2 cannot confirm).

Thresholds are matched on the test benign (in sample) for every option alike: the
comparison is between operating curves, not a deployment calibration.

Three processes, so that the full train load is freed before the test is read (a single
process reached 8.7 GB):
  python scripts/k_fusion_audit.py train     # specialists -> results/.cache/k2audit_models/
  python scripts/k_fusion_audit.py score     # 24 test columns only -> probabilities cache
  python scripts/k_fusion_audit.py           # the three tests
Out: results/k_fusion_audit.txt; models and probabilities in results/.cache/ (gitignored)
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
N, SEED = 14, 42
XGB_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 2}
CACHE = "results/.cache/ereno_probs_n14_k2audit.npz"


MODELS = "results/.cache/k2audit_models"


def lean_load_arff(path, keep=FEATURES):
    """Drop-in for python.util.load_arff with the same outputs (X, y, class_values, and
    util.normal_class = y[0]) but reading only the selected columns, as float32, in chunks:
    util.load_arff builds Python lists of all 58 columns and peaks near 9 GB on ERENO.
    Unselected columns are left at 0 (filter_features never reads them). Rows util.load_arff
    would skip (wrong field count, unparsable) would surface here as NaN, so they are
    checked for and must not exist."""
    import pandas as pd
    import python.util as util
    classes, skip, nattr = None, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            skip += 1
            s = line.strip()
            if s.lower().startswith("@attribute"):
                nattr += 1
                if "{" in s:
                    classes = [c.strip() for c in s.split(None, 2)[2].strip("{}").split(",")]
            elif s.lower().startswith("@data"):
                break
    cols = [f - 1 for f in sorted(keep)]
    cmap = {c: i for i, c in enumerate(classes)}
    Xs, ys = [], []
    for ch in pd.read_csv(path, skiprows=skip, header=None, usecols=cols + [nattr - 1],
                          chunksize=500_000, skip_blank_lines=True):
        x = ch[cols].to_numpy(np.float32)
        yy = ch[nattr - 1].astype(str).str.strip().map(cmap)
        assert not np.isnan(x).any() and not yy.isna().any(), "malformed rows: lean loader not equivalent"
        Xs.append(x)
        ys.append(yy.to_numpy(np.int64))
    Xsel, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs
    X = np.zeros((len(y), nattr - 1), np.float32)
    X[:, cols] = Xsel
    del Xsel
    util.normal_class = int(y[0])
    print(f"[lean] {path}: {len(y):,} rows, {len(cols)} of {nattr - 1} columns read", flush=True)
    return X, y, classes


def phase_train():
    """Separate process: the full train load must be freed before the test is read."""
    import json
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    import python.util as util
    from fd.dataset import load_and_partition
    util.load_arff = lean_load_arff          # same outputs, ~1 GB instead of ~9 GB
    parts, _, _ = load_and_partition(f"{DATASET}.csv", FEATURES, N, seed=SEED,
                                     partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    os.makedirs(MODELS, exist_ok=True)
    spec = []
    for i, (X_c, y_c) in enumerate(parts):
        atk = np.unique(y_c[y_c != nc])
        assert len(atk) == 1, atk
        spec.append(int(atk[0]))
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        yy = (yct != nc).astype(int)
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = dict(XGB_PARAMS)
        if p and q:
            pr["scale_pos_weight"] = q / p
        xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10,
                  verbose_eval=False).save_model(f"{MODELS}/spec{i:02d}.json")
        print(f"[train] specialist {i:02d}: attack class {spec[-1]}", flush=True)
    json.dump({"nc": int(nc), "spec": spec}, open(f"{MODELS}/meta.json", "w"))


def phase_score():
    """Separate process: read only the 24 feature columns and the class of the test file."""
    import json
    import xgboost as xgb
    meta = json.load(open(f"{MODELS}/meta.json"))
    Xfull, y, classes = lean_load_arff(f"{TESTFILE}.csv")
    X = Xfull[:, [f - 1 for f in sorted(FEATURES)]]
    del Xfull
    y = y.astype(np.int16)
    d = xgb.DMatrix(X)
    del X
    P = []
    for i in range(N):
        b = xgb.Booster()
        b.load_model(f"{MODELS}/spec{i:02d}.json")
        P.append(b.predict(d).astype(np.float32))
    np.savez_compressed(CACHE, P=np.column_stack(P), y=y, nc=meta["nc"], spec=np.array(meta["spec"]),
                        cv=np.array(classes))
    print(f"[score] cached {CACHE}", flush=True)


def load():
    z = np.load(CACHE, allow_pickle=False)
    print(f"[cache] {CACHE}")
    return z["P"], z["y"], int(z["nc"]), z["spec"], list(z["cv"])


def stats(pred, atk):
    tp = int((pred & atk).sum()); fp = int((pred & ~atk).sum())
    rec = tp / atk.sum(); fpr = fp / (~atk).sum(); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return 100 * f1, 100 * rec, 100 * prec, 100 * fpr, fp


def thr_for_fp(score, atk, n_fp):
    """Lowest threshold whose flag set (score > thr) has at most n_fp benign samples."""
    b = np.sort(score[~atk])[::-1]
    return float(b[n_fp]) if n_fp < len(b) else -np.inf


def main():
    P, y, nc, spec, cv = load()
    atk = y != nc
    name = (lambda c: cv[c] if cv and c < len(cv) else str(c))
    classes = sorted(set(spec.tolist()))
    pairs = {c: np.where(spec == c)[0] for c in classes}
    print(f"test: {len(y):,} samples, {int(atk.sum()):,} attack, {int((~atk).sum()):,} benign")
    print("specialists per attack: " + ", ".join(f"{name(c)}={list(pairs[c])}" for c in classes))
    V = (P >= 0.5)

    def line(lbl, pred):
        f1, rec, prec, fpr, fp = stats(pred, atk)
        per = "  ".join(f"{name(c)[:14]} {100 * pred[y == c].mean():5.1f}" for c in classes)
        print(f"  {lbl:<44} F1 {f1:6.2f}  rec {rec:6.2f}  prec {prec:6.2f}  FPR {fpr:.4f}%  FP {fp:>6}")
        print(f"      recall per attack: {per}")

    # ---- reference: k >= 2 over the 14 specialists at 0.5 (paper operating point)
    ref = V.sum(1) >= 2
    fp_ref = int((ref & ~atk).sum())
    print("\n=== reference (paper): GL union, 14 specialists, k >= 2, each at 0.5 ===")
    line("k>=2 @0.5", ref)
    print("  k = 1 at 0.5 for comparison:")
    line("k=1 @0.5 (14 specialists)", V.sum(1) >= 1)

    # ---- Test 1: one specialist per attack, k = 1, threshold on max score matched to FP of ref
    print(f"\n=== Test 1: one specialist per attack, k = 1, threshold matched to FP = {fp_ref} ===")
    for h in (0, 1):
        idx = np.array([pairs[c][h] for c in classes])
        s = P[:, idx].max(1)
        t = thr_for_fp(s, atk, fp_ref)
        line(f"half {h}: max of 7, thr {t:.4f}", s > t)
        line(f"half {h}: max of 7 @0.5", s >= 0.5)

    # ---- Test 2: mean of the two specialists per attack, max over attacks, matched FP
    print(f"\n=== Test 2: mean of the 2 specialists per attack, k = 1, matched FP = {fp_ref} ===")
    M = np.column_stack([P[:, pairs[c]].mean(1) for c in classes])
    s = M.max(1)
    t = thr_for_fp(s, atk, fp_ref)
    line(f"mean pair, max of 7, thr {t:.4f}", s > t)
    s14 = P.max(1)
    t14 = thr_for_fp(s14, atk, fp_ref)
    line(f"max of all 14, thr {t14:.4f}", s14 > t14)
    # k >= 2 curve itself at other per-specialist thresholds (is 0.5 a good point?)
    print("  k>=2 with a common per-specialist threshold t (curve):")
    for tt in (0.3, 0.4, 0.5, 0.6, 0.7):
        f1, rec, prec, fpr, fp = stats((P >= tt).sum(1) >= 2, atk)
        print(f"      t={tt:.1f}: F1 {f1:6.2f} rec {rec:6.2f} FPR {fpr:.4f}% FP {fp}")

    # ---- Test 3: FL shrinking, absence vs veto
    print("\n=== Test 3: FL keeps the first n nodes; absence vs veto ===")
    print("  n  | FL k>=2: F1 / rec / FPR | FL k=1: F1 / rec / FPR | FL k=1 matched FP: F1 / rec | "
          "attacks with 0 / 1 / 2 specialists | recall lost to absence / to veto (pp)")
    for n in range(14, 2, -1):
        S = np.arange(n)
        v = V[:, S].sum(1)
        k2, k1 = v >= 2, v >= 1
        a2, a1 = stats(k2, atk), stats(k1, atk)
        sm = P[:, S].max(1); tm = thr_for_fp(sm, atk, fp_ref)
        am = stats(sm > tm, atk)
        cnt = {c: int(np.isin(pairs[c], S).sum()) for c in classes}
        n_atk = atk.sum()
        absent = sum((y == c).sum() for c in classes if cnt[c] == 0) / n_atk * 100
        # veto: recall that k=1 recovers over k>=2 on attacks with exactly one specialist left
        veto = sum(((k1 & ~k2) & (y == c)).sum() for c in classes if cnt[c] == 1) / n_atk * 100
        z = [sum(1 for c in classes if cnt[c] == j) for j in (0, 1, 2)]
        print(f"  {n:>2} | {a2[0]:6.2f} / {a2[1]:6.2f} / {a2[3]:.4f}% | {a1[0]:6.2f} / {a1[1]:6.2f} / "
              f"{a1[3]:.4f}% | {am[0]:6.2f} / {am[1]:6.2f} | {z[0]} / {z[1]} / {z[2]} | "
              f"{absent:5.2f} / {veto:5.2f}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phase = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if phase == "train":
        sys.exit(phase_train())
    if phase == "score":
        sys.exit(phase_score())
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/k_fusion_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
