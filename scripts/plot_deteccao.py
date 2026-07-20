"""Round de detecção — Centralizado × FL × GL em paralelo.

Sobrepõe o recall por round das três arquiteturas e marca o "round de
detecção": o primeiro round em que o recall atinge o limiar (default 99%).

- Centralizado: treinado uma vez (bloco MONOLÍTICO do log) → linha
  horizontal de referência, disponível "desde o round 0";
- FL / GL: recall por round das linhas ROUND;.

Uso:
    python scripts/plot_deteccao.py <saida> --titulo "..." [--limiar 99] \
        FL=<log_federado> GL=<log_gossip>

O bloco MONOLÍTICO é lido de qualquer um dos logs.
"""

import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROUND_RE = re.compile(r"^ROUND;(\d+);\w+;f1=[\d.]+;recall=([\d.]+)")
MONO_RE  = re.compile(r"MONOL")
REC_RE   = re.compile(r"Recall\s*:\s*([\d.]+)")


def per_round_recall(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ROUND_RE.match(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def mono_recall(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        seen = False
        for line in fh:
            if MONO_RE.search(line):
                seen = True
            elif seen:
                m = REC_RE.search(line)
                if m:
                    return float(m.group(1))
    return None


def deteccao(rounds, limiar):
    for r in sorted(rounds):
        if rounds[r] >= limiar:
            return r
    return None


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out = sys.argv[1]
    args = sys.argv[2:]
    titulo, limiar = "10 clientes", 99.0
    for flag, cast in (("--titulo", str), ("--limiar", float)):
        if flag in args:
            i = args.index(flag)
            val = cast(args[i + 1])
            if flag == "--titulo": titulo = val
            else: limiar = val
            del args[i:i + 2]

    series, mono = {}, None
    for arg in args:
        label, path = arg.split("=", 1)
        series[label] = per_round_recall(path)
        if mono is None:
            mono = mono_recall(path)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    max_r = 1
    print(f"[detecção] limiar de recall = {limiar}%")
    for label, rounds in series.items():
        xs = sorted(rounds)
        ax.plot(xs, [rounds[r] for r in xs], marker="o", markersize=4, label=f"{label} (distribuído)")
        max_r = max(max_r, xs[-1] if xs else 1)
        d = deteccao(rounds, limiar)
        if d:
            ax.axvline(d, color=ax.lines[-1].get_color(), ls=":", alpha=0.6)
            ax.annotate(f"{label}: round {d}", (d, limiar),
                        textcoords="offset points", xytext=(5, -14),
                        color=ax.lines[-1].get_color(), fontsize=9)
        print(f"[detecção] {label:<4} → round {d}")

    if mono is not None:
        ax.axhline(mono, color="black", ls="--", alpha=0.7,
                   label=f"Centralizado (referência, {mono:.1f}%)")
        print(f"[detecção] Centralizado → round 0 (treino único), recall {mono:.2f}%")

    ax.axhline(limiar, color="red", ls="-", alpha=0.25)
    ax.set_xlabel("round")
    ax.set_ylabel("Recall (%)")
    ax.set_title(f"Round de detecção — {titulo} (limiar {limiar:.0f}%)")
    ax.xaxis.set_major_locator(MultipleLocator(2 if max_r <= 30 else 5))
    ax.set_xlim(left=1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=130)
    print(f"[detecção] {out}.png")


if __name__ == "__main__":
    main()
