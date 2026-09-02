#!/usr/bin/env python3
"""Testa a hipotese do conselheiro multi-view para o masquerade.

Para cada classe masquerade, treina 3 especialistas em VISOES de features
(electrical / protocol / temporal) e mede, no teste cheio:
  - FP e recall de cada visao;
  - CORRELACAO dos FP entre visoes (|intersecao| vs individual; Jaccard);
  - arbitragem AND (as 3 concordam) e >=2 (maioria): FP e recall resultantes.
Responde: as visoes erram de forma independente (arbitragem reduz FP) ou
correlacionada (nao reduz)? Sem re-treinar GRASP. Saida: results/masquerade_views.csv
Rodar: ~/venv-ereno314/bin/python scripts/masquerade_views.py
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
import python.util as util

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
SEED, BENCAP = 42, 200000

fam = json.load(open("features/feature_names.json"))["families"]
VIEWS = {
    "electrical": fam["electrical_instantaneous"] + fam["electrical_rms"] + fam["electrical_trap_area"],
    "protocol":   fam["protocol_counter"],
    "temporal":   fam["derived_diff_temporal"],
}
MASQ = ["masquerade_fake_fault", "masquerade_fake_normal"]


def class_names(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if "@class@" in s and "{" in s:
                return [x.strip() for x in s[s.index("{") + 1:s.index("}")].split(",")]
    return None


def main():
    names = class_names(f"{DATASET}.csv")
    print("[mv] carregando train...", flush=True)
    X_all, y_all, _ = util.load_arff(f"{DATASET}.csv")
    nc = util.normal_class
    print("[mv] carregando teste...", flush=True)
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc
    is_ben = (yte_m == nc); n_ben = int(is_ben.sum())
    ben_all = np.where(y_all == nc)[0]
    rng = np.random.default_rng(SEED)
    ben_keep = rng.permutation(ben_all)[:BENCAP]

    rows = []
    for cls_name in MASQ:
        cls = names.index(cls_name)
        atk_tr = np.where(y_all == cls)[0]
        tr_idx = np.concatenate([atk_tr, ben_keep])
        ytr = (y_all[tr_idx] == cls).astype(int)
        cls_te = (yte_m == cls); n_cls = int(cls_te.sum())
        print(f"\n[mv] {cls_name}: train atk={len(atk_tr):,} ben={len(ben_keep):,} | "
              f"test atk={n_cls:,}", flush=True)

        fp_sets, rec, fpn = {}, {}, {}
        fire = {}  # view -> boolean array over test (fired)
        for vname, feats in VIEWS.items():
            cols = [i - 1 for i in sorted(feats)]
            Xtr = X_all[np.ix_(tr_idx, cols)]
            p, q = int(ytr.sum()), int((ytr == 0).sum())
            pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
                  "seed": SEED, "nthread": 12}
            if p and q:
                pr["scale_pos_weight"] = q / p
            bst = xgb.train(pr, xgb.DMatrix(Xtr, label=ytr), num_boost_round=10,
                            verbose_eval=False)
            pf = (bst.predict(xgb.DMatrix(Xte_raw[:, cols])) >= 0.5)
            fire[vname] = pf
            fp_sets[vname] = set(np.where(pf & is_ben)[0].tolist())
            fpn[vname] = int((pf & is_ben).sum())
            rec[vname] = 100 * int((pf & cls_te).sum()) / max(n_cls, 1)
            print(f"    view {vname:<10} FP={fpn[vname]:>7,}  recall={rec[vname]:6.2f}%  "
                  f"(feat={len(cols)})", flush=True)
            del Xtr

        # correlacao dos FP entre as 3 visoes
        inter3 = fp_sets["electrical"] & fp_sets["protocol"] & fp_sets["temporal"]
        union3 = fp_sets["electrical"] | fp_sets["protocol"] | fp_sets["temporal"]
        jacc = {}
        for a, b in itertools.combinations(VIEWS, 2):
            u = len(fp_sets[a] | fp_sets[b]); i = len(fp_sets[a] & fp_sets[b])
            jacc[f"{a[:4]}-{b[:4]}"] = i / u if u else 0.0
        # arbitragem
        votes = np.vstack([fire[v].astype(np.int8) for v in VIEWS]).sum(0)
        fp_and = int(((votes >= 3) & is_ben).sum()); rec_and = 100 * int(((votes >= 3) & cls_te).sum()) / max(n_cls, 1)
        fp_maj = int(((votes >= 2) & is_ben).sum()); rec_maj = 100 * int(((votes >= 2) & cls_te).sum()) / max(n_cls, 1)

        print(f"  --- {cls_name}: FP-correlation ---", flush=True)
        print(f"    individual FP: " + ", ".join(f"{v}={fpn[v]:,}" for v in VIEWS), flush=True)
        print(f"    shared by ALL 3 (intersection) = {len(inter3):,}  |  union = {len(union3):,}", flush=True)
        print(f"    pairwise Jaccard: " + ", ".join(f"{k}={j:.2f}" for k, j in jacc.items()), flush=True)
        print(f"    AND (>=3 agree): FP={fp_and:,} recall={rec_and:.2f}%  |  "
              f">=2: FP={fp_maj:,} recall={rec_maj:.2f}%", flush=True)
        minfp = min(fpn.values())
        print(f"    => AND reduces FP to {fp_and:,} (best single view {minfp:,}); "
              f"{'INDEPENDENT (helps)' if fp_and < 0.5*minfp else 'CORRELATED (little help)'}", flush=True)
        rows.append((cls_name, fpn, rec, len(inter3), len(union3), jacc, fp_and, rec_and, fp_maj, rec_maj))

    os.makedirs("results", exist_ok=True)
    with open("results/masquerade_views.csv", "w") as fh:
        fh.write("class,view,FP,recall\n")
        for (cn, fpn, rec, i3, u3, jc, fa, ra, fm, rm) in rows:
            for v in VIEWS:
                fh.write(f"{cn},{v},{fpn[v]},{rec[v]:.4f}\n")
            fh.write(f"{cn},AND(>=3),{fa},{ra:.4f}\n")
            fh.write(f"{cn},MAJ(>=2),{fm},{rm:.4f}\n")
            fh.write(f"{cn},_intersection3,{i3},\n")
            fh.write(f"{cn},_union3,{u3},\n")
    print("\n[mv] CSV -> results/masquerade_views.csv", flush=True)


if __name__ == "__main__":
    main()
