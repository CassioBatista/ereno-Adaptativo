"""Curvas de convergência a partir dos logs das simulações.

Extrai as linhas 'ROUND;<n>;<modo>;f1=..;recall=..;fpr=..' e
'[GLow] união do head: <k> boosters' de um ou mais logs e gera:
  1. <saida>_metricas.png  — F1/Recall/FPR × round (uma curva por log)
  2. <saida>_difusao.png   — nº de boosters difundidos × round (logs gossip)
  3. <saida>.csv           — dados tabulados (log, round, modo, f1, recall, fpr, boosters)

Uso:
    python scripts/plot_convergencia.py <saida> [--titulo "..."] \
        <rótulo>=<log.log> [<rótulo>=<log.log> ...]

Exemplo:
    python scripts/plot_convergencia.py results/conv_ereno_c10 \
        --titulo "ERENO, 10 clientes" \
        Federado=results/ereno_fed_percli_c10.log \
        Gossip=results/ereno_gossip_percli_c10.log
"""

import csv
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUND_RE = re.compile(
    r"^ROUND;(\d+);(\w+);f1=([\d.]+);recall=([\d.]+);fpr=([\d.]+)")
UNION_RE = re.compile(r"união do head:\s*(\d+)\s*boosters")


def parse_log(path):
    rounds = {}          # round -> {mode, f1, recall, fpr}
    boosters = []        # k por chamada de agregação (ordem = rounds gossip)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ROUND_RE.match(line)
            if m:
                r = int(m.group(1))
                rounds[r] = {
                    "mode": m.group(2),
                    "f1": float(m.group(3)),
                    "recall": float(m.group(4)),
                    "fpr": float(m.group(5)),
                }
            u = UNION_RE.search(line)
            if u:
                boosters.append(int(u.group(1)))
    return rounds, boosters


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = sys.argv[1]
    args = sys.argv[2:]
    titulo = "10 clientes"
    if "--titulo" in args:
        i = args.index("--titulo")
        titulo = args[i + 1]
        del args[i:i + 2]
    series = []
    for arg in args:
        label, path = arg.split("=", 1)
        rounds, boosters = parse_log(path)
        series.append((label, rounds, boosters))

    # ── CSV ────────────────────────────────────────────────────────────────
    with open(f"{out}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["serie", "round", "modo", "f1", "recall", "fpr", "boosters"])
        for label, rounds, boosters in series:
            for r in sorted(rounds):
                d = rounds[r]
                bk = boosters[r - 1] if r - 1 < len(boosters) else ""
                w.writerow([label, r, d["mode"], d["f1"], d["recall"], d["fpr"], bk])
    print(f"[plot] CSV: {out}.csv")

    # ── 1. métricas × round ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, title in zip(
            axes, ["f1", "recall", "fpr"], ["F1-score (%)", "Recall (%)", "FPR (%)"]):
        for label, rounds, _ in series:
            xs = sorted(rounds)
            ys = [rounds[r][metric] for r in xs]
            ax.plot(xs, ys, marker="o", markersize=3, label=label)
        ax.set_xlabel("round")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Convergência por round — federado vs gossip ({titulo})")
    fig.tight_layout()
    fig.savefig(f"{out}_metricas.png", dpi=130)
    print(f"[plot] {out}_metricas.png")

    # ── 2. difusão (boosters × round) — só séries com gossip ───────────────
    gossip_series = [(l, b) for l, _, b in series if b]
    if gossip_series:
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
        for label, boosters in gossip_series:
            ax2.plot(range(1, len(boosters) + 1), boosters,
                     marker="s", markersize=4, label=label)
        ax2.set_xlabel("round")
        ax2.set_ylabel("nº de boosters na união do head")
        ax2.set_title("Difusão do conhecimento no gossip")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(f"{out}_difusao.png", dpi=130)
        print(f"[plot] {out}_difusao.png")


if __name__ == "__main__":
    main()
