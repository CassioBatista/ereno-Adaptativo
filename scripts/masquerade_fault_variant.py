#!/usr/bin/env python3
"""Variante reliability-aware para masquerade_fake_fault (o caso 'sistematico').

O teste multi-view mostrou: para fake_fault, a visao INDEPENDENTE (electrical)
tem recall baixo (50%), e as de recall alto (protocol/temporal) tem FP
correlacionados. Aqui testamos a arbitragem CIENTE DE CONFIABILIDADE:
  - DROPAR a electrical (visao nao-confiavel p/ fake_fault);
  - protocol AND temporal, com limiar tau selecionado na VALIDACAO (leakage-free).
Reporta FP/recall no teste cheio para varias fusoes + o tau* de validacao.
Saida: results/masquerade_fault_variant.csv
Rodar: ~/venv-ereno314/bin/python scripts/masquerade_fault_variant.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import python.util as util

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
SEED, BENCAP = 42, 200000
CLS_NAME = "masquerade_fake_fault"

fam = json.load(open("features/feature_names.json"))["families"]
VIEWS = {
    "electrical": fam["electrical_instantaneous"] + fam["electrical_rms"] + fam["electrical_trap_area"],
    "protocol":   fam["protocol_counter"],
    "temporal":   fam["derived_diff_temporal"],
}


def cnames(path):
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if "@class@" in s and "{" in s:
            return [x.strip() for x in s[s.index("{") + 1:s.index("}")].split(",")]


def prf(pred, is_atk, is_ben, n_atk, n_ben):
    tp = int((pred & is_atk).sum()); fp = int((pred & is_ben).sum())
    rec = 100 * tp / max(n_atk, 1); prec = 100 * tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return f1, rec, fp


def main():
    names = cnames(f"{DATASET}.csv"); cls = names.index(CLS_NAME)
    print("[fv] load train...", flush=True)
    X_all, y_all, _ = util.load_arff(f"{DATASET}.csv"); nc = util.normal_class
    print("[fv] load test...", flush=True)
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv"); util.normal_class = nc
    is_atk = (yte_m == cls); is_ben = (yte_m == nc)
    n_atk = int(is_atk.sum()); n_ben = int(is_ben.sum())

    # treino: fault + benign; separa 20% p/ VALIDACAO (selecao de tau, leakage-free)
    atk = np.where(y_all == cls)[0]
    ben = np.random.default_rng(SEED).permutation(np.where(y_all == nc)[0])[:BENCAP]
    idx = np.concatenate([atk, ben]); y = (y_all[idx] == cls).astype(int)
    tr, va = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=SEED, stratify=y)
    va_atk = y[va] == 1; va_ben = y[va] == 0
    na_v, nb_v = int(va_atk.sum()), int(va_ben.sum())

    prob_te, prob_va = {}, {}
    for vn, feats in VIEWS.items():
        cols = [i - 1 for i in sorted(feats)]
        Xtr = X_all[np.ix_(idx[tr], cols)]; ytr = y[tr]
        p, q = int(ytr.sum()), int((ytr == 0).sum())
        pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
              "seed": SEED, "nthread": 12}
        if p and q: pr["scale_pos_weight"] = q / p
        bst = xgb.train(pr, xgb.DMatrix(Xtr, label=ytr), num_boost_round=10, verbose_eval=False)
        prob_va[vn] = bst.predict(xgb.DMatrix(X_all[np.ix_(idx[va], cols)]))
        prob_te[vn] = bst.predict(xgb.DMatrix(Xte_raw[:, cols]))
        print(f"[fv] view {vn} trained (feat={len(cols)})", flush=True)

    # seleciona tau* (aplicado a protocol E temporal) por F1 na VALIDACAO, dropando electrical
    best = (-1, 0.5)
    for tau in [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]:
        pred = (prob_va["protocol"] >= tau) & (prob_va["temporal"] >= tau)
        f1v, _, _ = prf(pred, va_atk, va_ben, na_v, nb_v)
        if f1v > best[0]: best = (f1v, tau)
    tau_star = best[1]
    print(f"[fv] validation-selected tau* (protocol&temporal) = {tau_star} (F1_val={best[0]:.2f})", flush=True)

    def te(pred): return prf(pred, is_atk, is_ben, n_atk, n_ben)
    rows = []
    fusions = {
        "temporal@0.5":            prob_te["temporal"] >= 0.5,
        "protocol&temporal@0.5":   (prob_te["protocol"] >= 0.5) & (prob_te["temporal"] >= 0.5),
        f"protocol&temporal@tau*({tau_star})": (prob_te["protocol"] >= tau_star) & (prob_te["temporal"] >= tau_star),
        "all3&@0.5":               (prob_te["electrical"] >= 0.5) & (prob_te["protocol"] >= 0.5) & (prob_te["temporal"] >= 0.5),
    }
    print("\n===== fake_fault reliability-aware fusions (full test) =====", flush=True)
    print(f"{'fusion':<34}{'F1':>8}{'recall':>9}{'#FP':>12}", flush=True)
    for name, pred in fusions.items():
        f1, rec, fp = te(pred)
        rows.append((name, f1, rec, fp))
        print(f"{name:<34}{f1:>8.2f}{rec:>9.2f}{fp:>12,}", flush=True)

    # sweep de tau (protocol&temporal) no teste, para a sensibilidade
    print("\n--- sensitivity: protocol&temporal, tau sweep (test) ---", flush=True)
    sweep = []
    for tau in [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]:
        pred = (prob_te["protocol"] >= tau) & (prob_te["temporal"] >= tau)
        f1, rec, fp = te(pred); sweep.append((tau, f1, rec, fp))
        print(f"  tau={tau:.2f}  F1={f1:6.2f}  recall={rec:6.2f}  #FP={fp:,}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/masquerade_fault_variant.csv", "w") as fh:
        fh.write("kind,fusion_or_tau,F1,recall,FP\n")
        for name, f1, rec, fp in rows:
            fh.write(f'fusion,"{name}",{f1:.4f},{rec:.4f},{fp}\n')
        for tau, f1, rec, fp in sweep:
            fh.write(f"sweep,{tau},{f1:.4f},{rec:.4f},{fp}\n")
        fh.write(f"meta,tau_star,{tau_star},,\n")
    print("\n[fv] CSV -> results/masquerade_fault_variant.csv", flush=True)


if __name__ == "__main__":
    main()
