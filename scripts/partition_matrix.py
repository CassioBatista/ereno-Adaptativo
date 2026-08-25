#!/usr/bin/env python3
"""Matriz cliente x classe x split x exemplos para o particionador 'attack'.

Carrega all_in_one_ereno_train UMA vez e, para cada N, reproduz
benign_cap=500000 + _attack_per_client (seed=42) + split local 80/20
(train_test_split, random_state=42) — os mesmos passos dos experimentos.
Publica results/partition_matrix_N{N}.csv (auditavel) para cada N.

Rodar: ~/venv-ereno314/bin/python scripts/partition_matrix.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import train_test_split
import python.util as util
from fd.dataset import _attack_per_client

DATASET = "all_in_one_ereno_train"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
SEED, BENIGN_CAP = 42, 500000
NS = [14, 10, 7, 5, 3]

# nomes das classes a partir do cabecalho ARFF (indice = posicao na lista @class@)
names = None
with open(f"{DATASET}.csv", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if "@class@" in s and "{" in s:
            names = [x.strip() for x in s[s.index("{") + 1:s.index("}")].split(",")]
            break

print("[matrix] carregando all_in_one_ereno_train (uma vez)...")
X_all, y_all_full, _ = util.load_arff(f"{DATASET}.csv")
nc = util.normal_class
cols = [i - 1 for i in sorted(FEATURES)]
Xf_full = X_all[:, cols]
print(f"[matrix] {len(y_all_full):,} linhas; normal_class={nc}")

os.makedirs("results", exist_ok=True)

for N in NS:
    rng = np.random.default_rng(SEED)               # fresh rng por N (= load_and_partition)
    normal_idx = np.where(y_all_full == nc)[0]
    if len(normal_idx) > BENIGN_CAP:                 # benign_cap
        drop = rng.permutation(normal_idx)[BENIGN_CAP:]
        keep = np.ones(len(y_all_full), dtype=bool); keep[drop] = False
        Xf, y_all = Xf_full[keep], y_all_full[keep]
    else:
        Xf, y_all = Xf_full, y_all_full
    splits = _attack_per_client(Xf, y_all, N, rng)
    partitions = [(Xf[idx], y_all[idx]) for idx in splits]

    rows, summary = [], []
    for cid, (Xc, yc) in enumerate(partitions):
        _, _, ytr, yva = train_test_split(Xc, yc, test_size=0.2, random_state=SEED)
        per = {}
        for split, ys in (("train", ytr), ("val", yva)):
            uniq, cnts = np.unique(ys, return_counts=True)
            per[split] = {int(u): int(c) for u, c in zip(uniq, cnts)}
            for cls, c in sorted(per[split].items()):
                cname = "normal" if cls == nc else (names[cls] if names else str(cls))
                rows.append((cid, cname, split, "benign" if cls == nc else "attack", c))
        atk = sorted(c for c in per["train"] if c != nc)
        an = "|".join(names[c] if names else str(c) for c in atk)
        btr, bva = per["train"].get(nc, 0), per["val"].get(nc, 0)
        atr = sum(v for k, v in per["train"].items() if k != nc)
        ava = sum(v for k, v in per["val"].items() if k != nc)
        summary.append((cid, an, btr, bva, atr, ava, btr + bva + atr + ava))

    with open(f"results/partition_matrix_N{N}.csv", "w") as fh:
        fh.write("client,class,split,kind,examples\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")

    print(f"\n===== N = {N} =====")
    print(f"{'cli':>3} {'attack class(es)':<26} {'ben_tr':>7} {'ben_va':>7} {'atk_tr':>7} {'atk_va':>7} {'total':>8}")
    for cid, an, btr, bva, atr, ava, tot in summary:
        print(f"{cid:>3} {an:<26} {btr:>7,} {bva:>7,} {atr:>7,} {ava:>7,} {tot:>8,}")
    print(f"[N={N}] total particionado = {sum(s[6] for s in summary):,}  -> results/partition_matrix_N{N}.csv")

print("\n[matrix] concluido.")
