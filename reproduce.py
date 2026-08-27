#!/usr/bin/env python3
"""ReSIDS -- minimal end-to-end reproduction (single command).

    python reproduce.py            # min-run: verify hashes + train + evaluate
    python reproduce.py --strict   # also hash the 1.7 GB train/test CSVs

Clean-environment recipe (see REPRODUCE.md):
    python -m venv venv && venv/bin/pip install -r requirements.lock.txt
    venv/bin/python reproduce.py

The min-run loads the persisted GRASP feature manifest (NO GRASP re-run, no
distributed simulation), trains the ten specialists under the fixed seeds,
evaluates the full held-out ERENO test set, and emits, under results/:
  - reproduce_table.csv       headline metrics (OR k>=1 and corroboration k>=2)
  - reproduce_confusion.txt   TP/FP/TN/FN audit + per-attack recall
  - reproduce_figure.png      per-attack recall (OR vs k>=2)
It verifies input SHA-256 hashes against MANIFEST.json and checks the headline
numbers against the manifest's expected values, printing PASS/FAIL.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import python.util as util
from fd.dataset import load_and_partition

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
N, SEED = 10, 42
TOL = 0.05   # pp tolerance for headline F1/FPR match


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(man, strict):
    print("== input hashes (SHA-256) ==")
    ok = True
    for entry in man["inputs"]:
        p = os.path.join(ROOT, entry["path"])
        big = entry.get("bytes", 0) > 100_000_000
        if big and not strict:
            print(f"  {entry['path']:<40} [skipped -- pass --strict to hash "
                  f"{entry['bytes']/1e9:.2f} GB]")
            continue
        if not os.path.exists(p):
            print(f"  {entry['path']:<40} MISSING"); ok = False; continue
        got = sha256(p); exp = entry.get("sha256")
        tag = "ok" if (exp and got == exp) else ("SET" if not exp else "MISMATCH")
        if tag == "MISMATCH":
            ok = False
        print(f"  {entry['path']:<40} {got[:16]}...  [{tag}]")
    return ok


def load_features(man):
    fp = os.path.join(ROOT, man["feature_manifest"]["path"])
    with open(fp) as fh:
        feats = json.load(fh)["features"]
    print(f"== feature set: {len(feats)} features from "
          f"{man['feature_manifest']['path']} ==")
    return feats


def train_specialists(features):
    parts, _, _ = load_and_partition(f"{DATASET}.csv", features, N, seed=SEED,
                                     partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    fl = []
    for X_c, y_c in parts:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        yy = (yct != nc).astype(int)
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 12}
        if p and q:
            pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10,
                            verbose_eval=False))
    return fl, nc


def evaluate(fl, nc, features):
    names = _class_names()
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    Xte = util.filter_features(Xte_raw, features); del Xte_raw
    is_atk = (yte_m != nc); n_ben = int((~is_atk).sum()); n_atk = int(is_atk.sum())
    P = np.column_stack([b.predict(xgb.DMatrix(Xte)) for b in fl])
    votes = (P >= 0.5).astype(np.int16).sum(1)
    res = {}
    for k in (1, 2):
        pred = votes >= k
        tp = int((pred & is_atk).sum()); fp = int((pred & ~is_atk).sum())
        fn = int((~pred & is_atk).sum()); tn = int((~pred & ~is_atk).sum())
        fpr = 100 * fp / n_ben; rec = 100 * tp / n_atk
        prec = 100 * tp / max(tp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        mcc = matthews_corrcoef(is_atk.astype(int), pred.astype(int))
        per = {}
        for c in sorted(int(v) for v in np.unique(yte_m) if v != nc):
            m = (yte_m == c)
            per[names[c] if names else str(c)] = 100 * int((pred & m).sum()) / int(m.sum())
        res[k] = dict(f1=f1, recall=rec, precision=prec, fpr=fpr, mcc=mcc,
                      tp=tp, fp=fp, fn=fn, tn=tn, per_attack=per)
    return res


def _class_names():
    try:
        with open(f"{DATASET}.csv", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if "@class@" in s and "{" in s:
                    return [x.strip() for x in s[s.index("{") + 1:s.index("}")].split(",")]
    except OSError:
        pass
    return None


def write_outputs(res):
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    tbl = os.path.join(ROOT, "results", "reproduce_table.csv")
    with open(tbl, "w") as fh:
        fh.write("k,f1,recall,precision,fpr,mcc,tp,fp,fn,tn\n")
        for k in (1, 2):
            r = res[k]
            fh.write(f"{k},{r['f1']:.4f},{r['recall']:.4f},{r['precision']:.4f},"
                     f"{r['fpr']:.4f},{r['mcc']:.4f},{r['tp']},{r['fp']},{r['fn']},{r['tn']}\n")
    conf = os.path.join(ROOT, "results", "reproduce_confusion.txt")
    with open(conf, "w") as fh:
        for k in (1, 2):
            r = res[k]
            fh.write(f"=== k>={k} (tau=0.5) ===\n")
            fh.write(f"TP={r['tp']:,}  FP={r['fp']:,}  FN={r['fn']:,}  TN={r['tn']:,}\n")
            fh.write(f"F1={r['f1']:.2f} recall={r['recall']:.2f} precision={r['precision']:.2f} "
                     f"FPR={r['fpr']:.3f}% MCC={r['mcc']:.4f}\n")
            fh.write("per-attack recall:\n")
            for cn, v in r["per_attack"].items():
                fh.write(f"  {cn:<24} {v:6.2f}%\n")
            fh.write("\n")
    _figure(res)
    return tbl, conf


def _figure(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] matplotlib unavailable -- skipping figure")
        return
    labels = list(res[1]["per_attack"].keys())
    x = np.arange(len(labels)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, [res[1]["per_attack"][l] for l in labels], w, label="OR (k>=1)")
    ax.bar(x + w / 2, [res[2]["per_attack"][l] for l in labels], w, label="corroboration (k>=2)")
    ax.set_ylabel("per-attack recall (%)"); ax.set_ylim(0, 105)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend(); ax.set_title("ReSIDS reproduce -- per-attack recall (full ERENO test)")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "reproduce_figure.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] {out}")


def check_expected(res, man):
    exp = man.get("expected_min_run")
    if not exp:
        print("== no expected values in manifest -- skipping acceptance check ==")
        return True
    ok = True
    print("== acceptance check (vs MANIFEST.expected_min_run) ==")
    for k in (1, 2):
        e = exp[str(k)]; r = res[k]
        for metric in ("f1", "fpr"):
            d = abs(r[metric] - e[metric])
            passed = d <= TOL
            ok = ok and passed
            print(f"  k>={k} {metric:>4}: got {r[metric]:.3f} exp {e[metric]:.3f} "
                  f"(|d|={d:.3f}) {'PASS' if passed else 'FAIL'}")
        if "n_fp" in e:
            print(f"  k>={k}  #FP: got {r['fp']:,} exp {e['n_fp']:,}")
    return ok


def main():
    strict = "--strict" in sys.argv[1:]
    with open(os.path.join(ROOT, "MANIFEST.json")) as fh:
        man = json.load(fh)
    print(f"ReSIDS reproduce -- commit {man.get('git',{}).get('commit','?')} "
          f"-- Zenodo {man.get('zenodo_doi','?')}\n")
    hashes_ok = verify_hashes(man, strict)
    feats = load_features(man)
    print("\n[reproduce] training 10 specialists (seed 42, min-run)...")
    fl, nc = train_specialists(feats)
    print("[reproduce] evaluating full held-out test...")
    res = evaluate(fl, nc, feats)
    tbl, conf = write_outputs(res)
    print("\n" + open(conf).read())
    print(f"[reproduce] table  -> {os.path.relpath(tbl, ROOT)}")
    print(f"[reproduce] audit  -> {os.path.relpath(conf, ROOT)}")
    acc_ok = check_expected(res, man)
    print("\n==> " + ("REPRODUCE OK" if (acc_ok and (hashes_ok or not strict))
                       else "REPRODUCE: CHECK WARNINGS ABOVE"))
    sys.exit(0 if acc_ok else 1)


if __name__ == "__main__":
    main()
