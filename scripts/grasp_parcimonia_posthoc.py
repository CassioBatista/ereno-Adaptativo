"""Varredura de parcimonia POST-HOC sobre o log do GRASP por-ataque.

O GRASP logou (subconjunto, F1) de TODAS as avaliacoes (linhas EV;). Sem
treinar nada de novo, reconstruimos a fronteira de Pareto F1 x n-features de
cada ataque e aplicamos uma regra de parcimonia: o MENOR subconjunto cujo F1
esta a <= epsilon (pp) do melhor F1 encontrado (regra tipo 1-SE / e-tolerancia).

Mostra, por nivel de agressividade epsilon:
  - por ataque: menor-k, F1 e conjunto;
  - o SUPERSET (uniao) resultante e seu tamanho;
  - o COMBINADO com global-15.

Uso: python scripts/grasp_parcimonia_posthoc.py [caminho_do_log]
"""
import ast
import sys
from collections import defaultdict

GLOBAL15 = [5, 7, 8, 11, 19, 31, 33, 40, 41, 42, 44, 45, 50, 57, 58]
EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]   # pontos percentuais de F1


def parse(logpath):
    """-> {attack: {frozenset(features): best_f1}}"""
    per_attack = defaultdict(dict)
    cur = None
    with open(logpath, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "grasp-atk] ATAQUE" in line:
                cur = line.split("ATAQUE", 1)[1].split("(", 1)[0].strip()
            elif line.startswith("EV;") and cur:
                parts = line.rstrip("\n").split(";", 3)
                if len(parts) < 4:
                    continue
                try:
                    f1 = float(parts[2])
                    feats = frozenset(ast.literal_eval(parts[3]))
                except (ValueError, SyntaxError):
                    continue
                if not feats:
                    continue
                d = per_attack[cur]
                if feats not in d or f1 > d[feats]:
                    d[feats] = f1
    return per_attack


def pareto(subsets):
    """-> {k: (best_f1, best_set)} melhor F1 para cada cardinalidade."""
    by_k = {}
    for feats, f1 in subsets.items():
        k = len(feats)
        if k not in by_k or f1 > by_k[k][0]:
            by_k[k] = (f1, sorted(feats))
    return by_k


def minimal_within(subsets, eps):
    """Menor-k subconjunto com F1 >= best_f1 - eps (desempate: maior F1)."""
    best_f1 = max(subsets.values())
    cand = [(len(f), -f1, sorted(f)) for f, f1 in subsets.items()
            if f1 >= best_f1 - eps]
    k, negf1, feats = min(cand)
    return feats, -negf1, best_f1


def main():
    logpath = sys.argv[1] if len(sys.argv) > 1 else "/home/cassi/grasp_por_ataque.log"
    per_attack = parse(logpath)
    order = ["random_replay", "inverse_replay", "masquerade_fake_fault",
             "masquerade_fake_normal", "injection", "high_StNum", "poisoned_high_rate"]
    attacks = [a for a in order if a in per_attack] + \
              [a for a in per_attack if a not in order]

    # Fronteira de Pareto por ataque
    print("="*72)
    print("FRONTEIRA DE PARETO — melhor F1 por nº de features (por ataque)")
    print("="*72)
    for a in attacks:
        by_k = pareto(per_attack[a])
        ks = sorted(by_k)
        best = max(f for f, _ in by_k.values())
        line = "  ".join(f"k={k}:{by_k[k][0]:.3f}" for k in ks[:8])
        print(f"\n{a} (best F1={best:.3f}, {len(per_attack[a])} subconj. únicos):")
        print(f"  {line}")

    # Varredura de agressividade
    print("\n" + "="*72)
    print("PARCIMÔNIA: menor conjunto a <= epsilon pp do melhor F1")
    print("="*72)
    print(f"\n{'eps(pp)':>8}{'superset':>10}{'combinado':>11}   por-ataque (k)")
    for eps in EPS_GRID:
        picks = {a: minimal_within(per_attack[a], eps) for a in attacks}
        superset = sorted(set().union(*[set(p[0]) for p in picks.values()]))
        combined = sorted(set(GLOBAL15) | set(superset))
        ks = " ".join(f"{len(picks[a][0])}" for a in attacks)
        print(f"{eps:>8.2f}{len(superset):>10}{len(combined):>11}   [{ks}]")

    # Detalhe para um eps intermediário
    print("\n" + "="*72)
    for eps in (0.1, 0.5):
        print(f"\nDETALHE eps={eps} pp:")
        picks = {a: minimal_within(per_attack[a], eps) for a in attacks}
        for a in attacks:
            feats, f1, best = picks[a]
            print(f"  {a:<24} k={len(feats)} F1={f1:.3f} (best {best:.3f}) {feats}")
        superset = sorted(set().union(*[set(p[0]) for p in picks.values()]))
        combined = sorted(set(GLOBAL15) | set(superset))
        print(f"  -> superset_grasp ({len(superset)}): {superset}")
        print(f"  -> combinado ({len(combined)}): {combined}")


if __name__ == "__main__":
    main()
