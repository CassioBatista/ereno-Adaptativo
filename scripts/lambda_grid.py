#!/usr/bin/env python3
"""Grid de lambda (parcimonia) reconstruido do trace do GRASP global.

Le as avaliacoes EV;idx;F1;[features] do log one-shot do GRASP (busca global
VND, seed 5), monta a fronteira F1*(k) = melhor F1 de CV por cardinalidade k,
e para uma grade de lambda escolhe o subconjunto que maximiza (F1 - lambda*k).
Saida obrigatoria do revisor: F1 x complexidade (k) por lambda. Nao re-roda o
GRASP (cada rodada ~3.5 h); usa o trace ja persistido.

Rodar: ~/venv-ereno314/bin/python scripts/lambda_grid.py [log]
"""
import ast
import sys

LAMBDAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]


def frontier(logpath):
    best = {}   # k -> (f1, sorted feats)
    with open(logpath, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("EV;"):
                continue
            p = line.rstrip("\n").split(";", 3)
            if len(p) < 4:
                continue
            try:
                f1 = float(p[2]); feats = sorted(ast.literal_eval(p[3]))
            except (ValueError, SyntaxError):
                continue
            k = len(feats)
            if not k:
                continue
            if k not in best or f1 > best[k][0]:
                best[k] = (f1, feats)
    return best


def main():
    logpath = sys.argv[1] if len(sys.argv) > 1 else \
        "results/grasp_oneshot_ereno.log"
    front = frontier(logpath)
    ks = sorted(front)
    fmax = max(f for f, _ in front.values())
    print(f"[lambda-grid] log={logpath}")
    print(f"[lambda-grid] avaliacoes com fronteira: k in [{ks[0]},{ks[-1]}], "
          f"F1_max(CV)={fmax:.3f}")

    print("\n===== FRONTEIRA F1*(k) (CV, objetivo do GRASP) =====")
    print(" k   F1*_cv")
    for k in ks:
        print(f"{k:2d}   {front[k][0]:7.3f}")

    print("\n===== GRID DE LAMBDA: argmax_k (F1*(k) - lambda*k) =====")
    print(" lambda   k*   F1*_cv   features")
    rows = []
    for lam in LAMBDAS:
        kstar = max(ks, key=lambda k: front[k][0] - lam * k)
        f1, feats = front[kstar]
        rows.append((lam, kstar, f1, feats))
        print(f" {lam:5.2f}   {kstar:2d}   {f1:7.3f}   {feats}")

    print("\n[lambda-grid] CSV -> results/lambda_grid.csv")
    with open("results/lambda_grid.csv", "w") as fh:
        fh.write("lambda,k,f1_cv,features\n")
        for lam, k, f1, feats in rows:
            fh.write(f'{lam},{k},{f1:.4f},"{feats}"\n')


if __name__ == "__main__":
    main()
