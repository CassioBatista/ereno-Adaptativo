#!/usr/bin/env python3
"""Evaluation-v1, step 1: specialists and references, per-sample results on the test split
(docs/EVALUATION_PROTOCOL.md, sections 2.3, 3, 4 and 7). One domain and one seed per run.

What is trained, on the training split of the domain and its selected features
(features/all_in_one_<ds>_train_combined.json, the evaluation-v1 selection):
  - Specialists (section 2.3). One per attack type; the types under 100 training rows share
    one; placement on the 14 nodes by decreasing training rows, in contiguous blocks
    (scripts/ev1_partition_n14.py). With fewer specialists than nodes (ERENO), the extra nodes
    go to the largest specialists and a specialist on k nodes has its rows split into k
    disjoint random parts: the `attack` partitioner of fd/dataset.py, reproduced here draw for
    draw (benign cap, benign slices, class shuffles), so ERENO gets the same 14 partitions as
    the fusion rule selection (scripts/fusion_rule_selection_ev1.py), there trained on 80 %.
    Node i holds the i-th of 14 disjoint random slices of the training benign rows (ERENO:
    capped at 500,000 first); every specialist on node i is one XGBoost trained on its own
    attack rows against that slice.
  - IID reference: the same training rows (after the cap) split into 14 equal random parts
    (the `iid` partitioner of fd/dataset.py, same seed), one binary XGBoost per part with
    round(10 S / 14) rounds.
  - Centralized reference: one binary XGBoost of 10 S rounds on the pooled training rows.
  - Every model: max_depth 4, eta 0.1, binary:logistic, scale_pos_weight = benign / attack of
    its own training rows, seed = the run seed; specialists 10 rounds.

What is reported, on the full test split (section 7), per sample:
  - Attack-based, OR at 0.5 over all S specialists (section 3; with all nodes present, FL = GL
    under retention, so one detector); attack-based, k = 2 on the same scores (secondary);
    IID, OR at 0.5 over its 14 models; Centralized at 0.5; Local: each node with its own
    specialists only (OR), per node and the mean over nodes.
  - Recall, false-positive rate, precision and F1-score at the prevalence of the test split;
    F1-score at 6.8 %; precision at 1 % and 0.1 %, P = R pi / (R pi + FPR (1 - pi)), with FPR
    replaced by its 95 % upper bound 3/n when there is no false positive (marked ">=");
    per-type recall with exact (Clopper-Pearson) 95 % intervals, flagged for the types with
    fewer than 100 test samples.

Not here: node loss (section 5), windows (section 6), the trust-level replay (section 9).

  python scripts/ev1_step1.py --domain ereno|cicids|morris4|ciciot [--seed 42]
  python scripts/ev1_step1.py --domain morris4 --smoke   # training file only: 30 % of the
         training rows (seeded) stand in for the test split; the test file is never opened;
         outputs go to results/.cache/ev1_step1_smoke/ and are not results
Out: results/ev1_step1_<domain>_s<seed>.{txt,json}, results/ev1_step1_env_<domain>_s<seed>.txt;
scores in results/.cache/ev1_step1_<domain>_s<seed>.npz, models in results/.cache/ev1_step1_models_<domain>_s<seed>/.
One job at a time: refuses to start while a GRASP job or another run of this script is running.
XGBoost threads: NTHREAD in the environment (default 8), recorded with the results.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

N, TREES, SMALL = 14, 10, 100
ERENO_PREV = 0.068
OP_PREV = (0.01, 0.001)
NTHREAD = int(os.environ.get("NTHREAD", "8"))
DOMAINS = {
    "ereno":   dict(name="ERENO", train="all_in_one_ereno_train.csv", test="all_in_one_ereno_test.csv",
                    sel="features/all_in_one_ev1_ereno_train_combined.json", cap=500_000),
    "cicids":  dict(name="CICIDS2017, improved", train="all_in_one_ev1_cicids_train.csv",
                    test="all_in_one_ev1_cicids_test.csv",
                    sel="features/all_in_one_ev1_cicids_train_combined.json", cap=None),
    "morris4": dict(name="Morris-4", train="all_in_one_ev1_morris4_train.csv",
                    test="all_in_one_ev1_morris4_test.csv",
                    sel="features/all_in_one_ev1_morris4_train_combined.json", cap=None,
                    method_must="crc rate as content"),
    "ciciot":  dict(name="CICIoT2023", train="all_in_one_ev1_ciciot_train.csv",
                    test="all_in_one_ev1_ciciot_test.csv",
                    sel="features/all_in_one_ev1_ciciot_train_combined.json", cap=None),
}


# ── data ──────────────────────────────────────────────────────────────────────

def load(path, feats):
    """Selected columns (1-based indices) as float32 and the class codes (declaration order).
    Benign = the class of the first data row (python/util.load_arff convention)."""
    import pandas as pd
    classes, skip, nattr = None, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            skip += 1
            s = line.strip()
            if s.lower().startswith("@attribute"):
                nattr += 1
                if "{" in s:
                    classes = [c.strip().strip("'\"") for c in s[s.index("{") + 1:s.rindex("}")].split(",")]
            elif s.lower().startswith("@data"):
                break
    cols = [f - 1 for f in feats]
    assert max(cols) < nattr - 1, (max(cols), nattr)
    cmap = {c: i for i, c in enumerate(classes)}
    Xs, ys = [], []
    for ch in pd.read_csv(path, skiprows=skip, header=None, usecols=cols + [nattr - 1],
                          chunksize=500_000, skip_blank_lines=True):
        x = ch[cols].to_numpy(np.float32)
        yy = ch[nattr - 1].astype(str).str.strip().str.strip("'\"").map(cmap)
        assert not yy.isna().any(), f"rows with an undeclared class in {path}"
        Xs.append(x)
        ys.append(yy.to_numpy(np.int16))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    # NaN is a value of the data (Morris-4: fields absent from a message type), passed to
    # XGBoost as missing, as in the GRASP runs; reported so that it can be checked
    nan = np.isnan(X).sum(0)
    print(f"[load] {path}: {len(y):,} rows, {len(cols)} of {nattr - 1} columns"
          + (f"; NaN per column {dict((f, int(k)) for f, k in zip([c + 1 for c in cols], nan) if k)}"
             if nan.any() else ""), flush=True)
    return X, y, classes, int(y[0])


def selection(dom):
    j = json.load(open(dom["sel"]))
    m = j.get("method", "")
    assert "best-subset retention rule" in m, m
    if dom.get("method_must"):
        assert dom["method_must"] in m, f"{dom['sel']} is not the selection of the amended data: {m}"
    return sorted(int(f) for f in j["features"]), j


# ── partition (section 2.3) ───────────────────────────────────────────────────

def capped(y, nc, cap, seed):
    """Rows kept after the benign cap: fd/dataset.py, first draw of a default_rng(seed)."""
    rng = np.random.default_rng(seed)
    keep = np.ones(len(y), bool)
    if cap:
        normal_idx = np.where(y == nc)[0]
        if len(normal_idx) > cap:
            keep[rng.permutation(normal_idx)[cap:]] = False
    return rng, np.where(keep)[0]


def attack_partition(y, nc, classes, cap, seed):
    """[(name, type codes, node, row indices into y)] and the 14 benign slices (indices into y)."""
    import ev1_partition_n14 as P
    P.N = N                                             # --nodes (default 14)
    rng, kept = capped(y, nc, cap, seed)
    yk = y[kept]
    benign = np.where(yk == nc)[0].copy()
    rng.shuffle(benign)
    slices = [kept[s] for s in np.array_split(benign, N)]
    counts = np.bincount(yk, minlength=len(classes)).tolist()
    units = P.specialists(classes, nc, counts)          # (name, codes, rows, node)
    out = []
    if len(units) == N and len({u[0] for u in units}) < N:
        # fewer specialists than nodes: fd/dataset.py `_attack_per_client`, redundant branch
        types = sorted((int(c) for c in np.unique(yk) if c != nc), key=lambda c: -int((yk == c).sum()))
        per = {c: sum(1 for u in units if u[1] == [c]) for c in types}
        cid = 0
        for c in types:
            idx = np.where(yk == c)[0].copy()
            rng.shuffle(idx)
            for part in np.array_split(idx, per[c]):
                u = units[cid]
                assert u[1] == [c] and u[3] == cid and u[2] == len(part), (u, c, cid, len(part))
                out.append((u[0], [c], cid, kept[np.sort(part)]))
                cid += 1
    else:
        for name, codes, rows, node in units:
            idx = np.where(np.isin(yk, codes))[0]
            assert len(idx) == rows
            out.append((name, codes, node, kept[idx]))
    return out, slices, kept


def iid_partition(y, nc, cap, seed):
    """fd/dataset.py `iid` after the same cap: a fresh default_rng(seed), cap draw, permutation."""
    rng, kept = capped(y, nc, cap, seed)
    perm = rng.permutation(len(kept))
    return [kept[np.sort(p)] for p in np.array_split(perm, N)]


# ── models ────────────────────────────────────────────────────────────────────

def train(X, yb, rounds, seed):
    import xgboost as xgb
    p, q = int(yb.sum()), int((yb == 0).sum())
    pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": seed,
          "nthread": NTHREAD}
    if p and q:
        pr["scale_pos_weight"] = q / p
    return xgb.train(pr, xgb.DMatrix(X, label=yb), num_boost_round=rounds, verbose_eval=False), p, q


# ── metrics (section 7) ───────────────────────────────────────────────────────

def cp(k, n):
    from scipy.stats import beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def metrics(pred, y, nc, types):
    atk, ben = y != nc, y == nc
    tp, fp = int((pred & atk).sum()), int((pred & ben).sum())
    na, nb = int(atk.sum()), int(ben.sum())
    R = tp / na if na else float("nan")
    fpr = fp / nb
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * R / (prec + R) if tp else 0.0
    fpr_b = fpr if fp else 3 / nb                      # 95 % upper bound when no false positive

    def p_at(pi):
        return R * pi / (R * pi + fpr_b * (1 - pi)) if R > 0 else 0.0
    p68 = p_at(ERENO_PREV)
    per = {}
    for c in types:
        m = y == c
        n, k = int(m.sum()), int((pred & m).sum())
        per[c] = {"n": n, "k": k, "recall": k / n if n else float("nan"), "ci95": cp(k, n),
                  "small": n < 100}
    return {"tp": tp, "fp": fp, "n_attack": na, "n_benign": nb, "recall": R, "fpr": fpr,
            "precision": prec, "f1": f1, "fp_zero_bound": fp == 0,
            "f1_at_6.8": 2 * p68 * R / (p68 + R) if R > 0 else 0.0,
            "precision_at": {str(pi): p_at(pi) for pi in OP_PREV}, "per_type": per}


# ── run ───────────────────────────────────────────────────────────────────────

def busy():
    me = os.getpid()
    out = subprocess.run(["pgrep", "-af", r"grasp_(por_ataque|global_penalizado)\.py|ev1_step1\.py"],
                         capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if int(l.split()[0]) not in (me, os.getppid()) and "pgrep" not in l]


def env_record(path, dom, feats, seed):
    import xgboost, sklearn, scipy, pandas
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout
    with open(path, "w") as f:
        f.write(f"{time.strftime('%F %T')}\nHEAD {git('rev-parse', 'HEAD')}"
                f"status:\n{git('status', '--short', 'scripts/ev1_step1.py', 'scripts/ev1_partition_n14.py', dom['sel'], 'docs')}"
                f"diff of the script:\n{git('diff', 'HEAD', '--', 'scripts/ev1_step1.py')}\n"
                f"python {sys.version.split()[0]}, xgboost {xgboost.__version__}, numpy {np.__version__}, "
                f"scikit-learn {sklearn.__version__}, scipy {scipy.__version__}, pandas {pandas.__version__}; "
                f"nthread {NTHREAD}; seed {seed}; features {feats}\n")


def specialist_count(path):
    """S, the number of specialists of the domain (types under SMALL training rows share one);
    read from the label column only, before the node count is fixed."""
    import ev1_partition_n14 as P
    classes, benign, counts = P.label_counts(path)
    P.N = 1                                             # all on one node: one unit per specialist
    return len(P.specialists(classes, benign, counts))


def run(domain, seed, smoke):
    dom = DOMAINS[domain]
    tag = f"{domain}_s{seed}" + ("" if N == 14 else f"_n{N}")
    if smoke:
        cache = out_base = os.path.join("results/.cache/ev1_step1_smoke", tag)
        os.makedirs(cache, exist_ok=True)
        feats, _ = (list(range(1, 17)), None) if domain == "morris4" else selection(dom)
    else:
        cache, out_base = "results/.cache", f"results/ev1_step1_{tag}"
        feats, _ = selection(dom)
        env_record(f"results/ev1_step1_env_{tag}.txt", dom, feats, seed)
    mdir = os.path.join(cache, f"ev1_step1_models_{tag}")
    os.makedirs(mdir, exist_ok=True)
    t0 = time.time()

    # training split
    X, y, classes, nc = load(dom["train"], feats)
    if smoke:                                          # 30 % of the training rows stand in for the test
        r = np.random.default_rng(seed).random(len(y)) < 0.3
        r[0] = False                                   # keep the first (benign) row in training
        Xt, yt = X[r], y[r]
        X, y = X[~r], y[~r]
    units, slices, kept = attack_partition(y, nc, classes, dom["cap"], seed)
    S = len(units)
    types = [c for c in range(len(classes)) if c != nc and (y == c).any()]
    print(f"[partition] {dom['name']}: {len(types)} attack types, {S} specialists, "
          f"{len(kept):,} training rows after the cap; benign per node {min(map(len, slices)):,}-"
          f"{max(map(len, slices)):,}", flush=True)
    spec = []
    for j, (name, codes, node, idx) in enumerate(units):
        rows = np.concatenate([idx, slices[node]])
        b, p, q = train(X[rows], np.isin(y[rows], codes).astype(np.int8), TREES, seed)
        b.save_model(f"{mdir}/spec{j:02d}.json")
        spec.append({"j": j, "name": name, "codes": codes, "node": node + 1, "attack": p, "benign": q})
        print(f"[spec {j:02d}] node {node + 1:2d} {name[:60]}: {p:,} attack / {q:,} benign", flush=True)
    iid_rounds = int(round(TREES * S / N))
    iid = []
    for i, idx in enumerate(iid_partition(y, nc, dom["cap"], seed)):
        b, p, q = train(X[idx], (y[idx] != nc).astype(np.int8), iid_rounds, seed)
        b.save_model(f"{mdir}/iid{i:02d}.json")
        iid.append({"node": i + 1, "attack": p, "benign": q})
    print(f"[iid] {N} models of {iid_rounds} rounds", flush=True)
    b, p, q = train(X[kept], (y[kept] != nc).astype(np.int8), TREES * S, seed)
    b.save_model(f"{mdir}/central.json")
    print(f"[central] {TREES * S} rounds on {p:,} attack / {q:,} benign", flush=True)
    del X, y, kept, slices, units

    # test split (never read in smoke mode)
    import xgboost as xgb
    if not smoke:
        Xt, yt, classes_t, nct = load(dom["test"], feats)
        assert classes_t == classes and nct == nc, "train and test class declarations differ"
    d = xgb.DMatrix(Xt)
    del Xt

    def score(fn):
        bst = xgb.Booster()
        bst.load_model(f"{mdir}/{fn}.json")
        bst.set_param({"nthread": NTHREAD})
        return bst.predict(d).astype(np.float32)
    Ps = np.column_stack([score(f"spec{j:02d}") for j in range(S)])
    Pi = np.column_stack([score(f"iid{i:02d}") for i in range(N)])
    Pc = score("central")
    np.savez_compressed(os.path.join(cache, f"ev1_step1_{tag}.npz"), Ps=Ps, Pi=Pi, Pc=Pc, y=yt, nc=nc,
                        classes=np.array(classes), node=np.array([s["node"] for s in spec]))
    print(f"[score] {len(yt):,} test rows scored", flush=True)

    # per-sample results
    hit = Ps >= 0.5
    det = {"attack-based OR": hit.any(1), "attack-based k=2": hit.sum(1) >= 2,
           "IID OR": (Pi >= 0.5).any(1), "Centralized": Pc >= 0.5}
    res = {k: metrics(v, yt, nc, types) for k, v in det.items()}
    nodes = np.array([s["node"] for s in spec])
    local = {}
    for n in range(1, N + 1):
        own = sorted({c for s in spec if s["node"] == n for c in s["codes"]})
        m = metrics(hit[:, nodes == n].any(1), yt, nc, types)
        mo = np.isin(yt, own)
        m["recall_own_types"] = float(hit[mo][:, nodes == n].any(1).mean()) if mo.any() else float("nan")
        m["own_types"] = [classes[c] for c in own]
        local[n] = m
    res["Local (mean over nodes)"] = {k: float(np.mean([local[n][k] for n in local]))
                                      for k in ("recall", "fpr", "precision", "f1", "recall_own_types")}

    # report
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"evaluation-v1 step 1 -- {dom['name']} -- seed {seed} -- {N} nodes"
              + ("  [SMOKE: training rows only, not results]" if smoke else ""))
        print(f"features ({len(feats)}): {feats}")
        print(f"specialists {S} (10 rounds each), IID {N} x {iid_rounds} rounds, Centralized "
              f"{TREES * S} rounds; nthread {NTHREAD}; test {len(yt):,} rows "
              f"({int((yt != nc).sum()):,} attack, prevalence {100 * (yt != nc).mean():.2f} %)")
        print(f"\n{'detector':<22}{'F1':>8}{'recall':>9}{'prec.':>9}{'FPR %':>10}{'FP':>9}"
              f"{'F1@6.8%':>9}{'P@1%':>9}{'P@0.1%':>9}")
        for k in det:
            e = res[k]
            b = ">=" if e["fp_zero_bound"] else "  "
            print(f"{k:<22}{100 * e['f1']:8.2f}{100 * e['recall']:9.2f}{100 * e['precision']:9.2f}"
                  f"{100 * e['fpr']:10.4f}{e['fp']:9,}{100 * e['f1_at_6.8']:9.2f}"
                  f"{b}{100 * e['precision_at']['0.01']:7.2f}{b}{100 * e['precision_at']['0.001']:7.2f}")
        L = res["Local (mean over nodes)"]
        print(f"{'Local (mean, nodes)':<22}{100 * L['f1']:8.2f}{100 * L['recall']:9.2f}"
              f"{100 * L['precision']:9.2f}{100 * L['fpr']:10.4f}   (recall on own types {100 * L['recall_own_types']:.2f})")
        print("\nper-type recall % [exact 95 % interval] (* fewer than 100 test samples)")
        print(f"{'type':<40}{'n':>9}" + "".join(f"{k[:16]:>18}" for k in det))
        for c in types:
            n = res["attack-based OR"]["per_type"][c]["n"]
            flag = "*" if n < 100 else " "
            cells = "".join(
                f"{100 * res[k]['per_type'][c]['recall']:7.2f}" +
                (f" [{100 * res[k]['per_type'][c]['ci95'][0]:.0f}-{100 * res[k]['per_type'][c]['ci95'][1]:.0f}]"
                 if n < 100 else "").ljust(11) for k in det)
            print(f"{flag}{classes[c][:39]:<39}{n:9,}{cells}")
        print("\nLocal, per node: recall (all attacks) / recall (own types) / FPR %")
        for n, m in local.items():
            print(f"  node {n:2d}: {100 * m['recall']:6.2f} / {100 * m['recall_own_types']:6.2f} / "
                  f"{100 * m['fpr']:.4f}   {', '.join(m['own_types'])[:90]}")
        print(f"\nelapsed {time.time() - t0:.0f} s")
    print(buf.getvalue(), end="")
    with open(f"{out_base}.txt" if not smoke else os.path.join(out_base, "report.txt"), "w") as f:
        f.write(buf.getvalue())

    def js(m):
        return {k: (v if k != "per_type" else {classes[c]: x for c, x in v.items()}) for k, v in m.items()}
    json.dump({"domain": dom["name"], "seed": seed, "nodes": N, "smoke": smoke, "features": feats, "nthread": NTHREAD,
               "specialists": spec, "iid": iid, "iid_rounds": iid_rounds, "central_rounds": TREES * S,
               "results": {k: js(v) for k, v in res.items() if k in det},
               "local_mean": res["Local (mean over nodes)"], "local": {n: js(m) for n, m in local.items()}},
              open(f"{out_base}.json" if not smoke else os.path.join(out_base, "report.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nodes", default="14",
                    help="node count: an integer (default 14), or 'spec' for one node per specialist "
                         "(N = S; outputs tagged _n<N> when N != 14)")
    a = ap.parse_args()
    os.chdir(ROOT)
    N = specialist_count(DOMAINS[a.domain]["train"]) if a.nodes == "spec" else int(a.nodes)
    if not a.smoke:
        b = busy()
        if b:
            sys.exit("another job is running; not starting:\n" + "\n".join(b))
    run(a.domain, a.seed, a.smoke)
