"""Figura da fronteira de Pareto (F1 x nº features por ataque) + superset x lambda.

Justifica o parametro de penalidade lambda do GRASP parcimonioso: mostra que
lambda=0.2 esta no "vale" — toda a parcimonia barata dos ataques faceis ja foi
capturada e nenhuma feature essencial do masquerade foi cortada.

Le o JSON de scripts/pareto_lambda_dados.py. Uso:
    python scripts/pareto_lambda_dados.py > /tmp/pareto.json
    python scripts/plot_pareto.py /tmp/pareto.json results/pareto_features_ereno.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDA_STAR = 0.2
EASY = {"random_replay", "inverse_replay", "injection", "high_StNum", "poisoned_high_rate"}
COL = {"masquerade_fake_fault": "#e34948", "masquerade_fake_normal": "#eb6834"}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pareto.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/pareto_features_ereno.png"
    d = json.load(open(src, encoding="utf-8"))
    attacks = d["attacks"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # ── painel 1: fronteira de Pareto F1(k) ─────────────────────────────────
    for a in attacks:
        ks = [k for k, _ in d["pareto"][a]]
        f1 = [v for _, v in d["pareto"][a]]
        if a in EASY:
            ax1.plot(ks, f1, color="#b4b2a9", lw=1.4, zorder=1)
        else:
            ax1.plot(ks, f1, color=COL[a], lw=2.6, label=a, zorder=3)
    # pontos da selecao lambda*
    sweep = {s["lambda"]: s for s in d["lambda_sweep"]}
    star = sweep[LAMBDA_STAR]
    for a in attacks:
        k = star["k_por_ataque"][a]; f1 = star["f1_por_ataque"][a]
        ax1.scatter([k], [f1], s=55, color="#2a78d6", edgecolor="white",
                    linewidth=1.5, zorder=5)
    ax1.plot([], [], color="#b4b2a9", lw=1.4, label="5 ataques fáceis")
    ax1.scatter([], [], s=55, color="#2a78d6", edgecolor="white",
                linewidth=1.5, label=f"seleção λ={LAMBDA_STAR}")
    ax1.set_xlabel("nº de features (k)"); ax1.set_ylabel("F1 (%) — 5-fold CV")
    ax1.set_xlim(1, 8); ax1.set_ylim(82, 100.6)
    ax1.set_title("Fronteira de Pareto por ataque")
    ax1.grid(True, color="#e1e0d9", lw=0.6); ax1.legend(fontsize=8, loc="lower right")

    # ── painel 2: superset x lambda ─────────────────────────────────────────
    lams = [s["lambda"] for s in d["lambda_sweep"]]
    sup = [s["n_superset"] for s in d["lambda_sweep"]]
    ax2.plot(lams, sup, color="#5f5e5a", lw=2, marker="o", ms=4)
    ax2.scatter([LAMBDA_STAR], [sweep[LAMBDA_STAR]["n_superset"]], s=80,
                color="#2a78d6", edgecolor="white", linewidth=1.5, zorder=5)
    ax2.axvline(LAMBDA_STAR, color="#2a78d6", lw=0.8, ls="--", alpha=0.6)
    ax2.annotate(f"λ={LAMBDA_STAR}\n{sweep[LAMBDA_STAR]['n_superset']} features",
                 (LAMBDA_STAR, sweep[LAMBDA_STAR]["n_superset"]),
                 textcoords="offset points", xytext=(12, 12), fontsize=9,
                 color="#185fa5")
    ax2.set_xlabel("λ (penalidade por feature, pp de F1)")
    ax2.set_ylabel("nº features no superset")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, max(sup) + 3)
    ax2.set_title("Agressividade: superset x λ")
    ax2.grid(True, color="#e1e0d9", lw=0.6)

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[plot] salvo em {out}")


if __name__ == "__main__":
    main()
