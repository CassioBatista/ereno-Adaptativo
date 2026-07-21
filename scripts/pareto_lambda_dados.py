"""Emite os dados da fronteira de Pareto e da varredura de lambda (JSON).

A partir do log do GRASP por-ataque (subconjunto, F1 de cada avaliacao),
reconstroi a fronteira F1*(k) por ataque e, para uma grade de lambda, o
subconjunto que maximiza (F1 - lambda*k) por ataque + o superset resultante.
Serve de dado para a figura de Pareto / justificativa do lambda.
"""
import ast
import json
import sys
from collections import defaultdict

GLOBAL15 = [5, 7, 8, 11, 19, 31, 33, 40, 41, 42, 44, 45, 50, 57, 58]
LAMBDAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]


def parse(logpath):
    per_attack = defaultdict(dict)
    cur = None
    with open(logpath, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "grasp-atk] ATAQUE" in line:
                cur = line.split("ATAQUE", 1)[1].split("(", 1)[0].strip()
            elif line.startswith("EV;") and cur:
                p = line.rstrip("\n").split(";", 3)
                if len(p) < 4:
                    continue
                try:
                    f1 = float(p[2]); feats = frozenset(ast.literal_eval(p[3]))
                except (ValueError, SyntaxError):
                    continue
                if feats and (feats not in per_attack[cur] or f1 > per_attack[cur][feats]):
                    per_attack[cur][feats] = f1
    return per_attack


def frontier(subsets):
    by_k = {}
    for feats, f1 in subsets.items():
        k = len(feats)
        if k not in by_k or f1 > by_k[k][0]:
            by_k[k] = (f1, sorted(feats))
    return by_k  # {k: (f1, set)}


def main():
    logpath = sys.argv[1] if len(sys.argv) > 1 else "/home/cassi/grasp_por_ataque.log"
    per_attack = parse(logpath)
    order = ["random_replay", "inverse_replay", "masquerade_fake_fault",
             "masquerade_fake_normal", "injection", "high_StNum", "poisoned_high_rate"]
    attacks = [a for a in order if a in per_attack]

    out = {"attacks": attacks, "pareto": {}, "lambda_sweep": []}
    fronts = {a: frontier(per_attack[a]) for a in attacks}
    for a in attacks:
        ks = sorted(fronts[a])
        out["pareto"][a] = [[k, round(fronts[a][k][0], 4)] for k in ks]

    for lam in LAMBDAS:
        picks = {}
        for a in attacks:
            f = fronts[a]
            kstar = max(f, key=lambda k: f[k][0] - lam * k)
            picks[a] = {"k": kstar, "f1": round(f[kstar][0], 4),
                        "features": f[kstar][1]}
        superset = sorted(set().union(*[set(picks[a]["features"]) for a in attacks]))
        combined = sorted(set(GLOBAL15) | set(superset))
        out["lambda_sweep"].append({
            "lambda": lam,
            "k_por_ataque": {a: picks[a]["k"] for a in attacks},
            "f1_por_ataque": {a: picks[a]["f1"] for a in attacks},
            "superset": superset, "n_superset": len(superset),
            "n_combined": len(combined),
            "picks": {a: picks[a]["features"] for a in attacks},
        })
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
