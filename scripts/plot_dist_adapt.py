#!/usr/bin/env python3
"""Curvas da v2 adaptativa distribuída vs baseline estático + eventos do monitor.

Lê os logs por-round (ROUND;r;mode;f1=;recall=;fpr=) das duas simulações e a
trilha de auditoria do monitor (architecture_change) da run adaptativa, e
sobrepõe F1 e FPR por round, marcando (i) as faltas injetadas no ambiente e
(ii) os commits de troca detectados (node_failure / recovery).

Saída: results/dist_adapt_vs_static.{png,pdf}
Rodar: ~/venv-ereno314/bin/python scripts/plot_dist_adapt.py
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--adapt", default="results/dist_adapt_c10.log")
_p.add_argument("--adapt2", default=None, help="segunda curva adaptativa (ex.: pré-fix)")
_p.add_argument("--adapt2-label", default="Adaptativo (variante)")
_p.add_argument("--adapt-label", default="Adaptativo")
_p.add_argument("--static", default="results/dist_static_c10.log")
_p.add_argument("--trail", default="results/monitor_events_dist_adapt.jsonl")
_p.add_argument("--faults", default="10:nó 7 cai;16:nó 4 cai;24:nós 4,7 voltam")
_p.add_argument("--out", default="results/dist_adapt_vs_static")
_p.add_argument("--title", default="ERENO v2 — troca distribuída/automática vs baseline estático sob node-loss")
_a = _p.parse_args()
ADAPT_LOG = _a.adapt
STATIC_LOG = _a.static
TRAIL = _a.trail
FAULTS = {}
for _f in _a.faults.split(";"):
    if ":" in _f:
        _r, _t = _f.split(":", 1)
        FAULTS[int(_r)] = _t
OUT = _a.out

_ROUND = re.compile(r"ROUND;(\d+);(\w+);f1=([\d.]+);recall=([\d.]+);fpr=([\d.]+)")


def parse_log(path):
    rounds, f1, fpr, modes = [], [], [], []
    if not os.path.exists(path):
        print("[plot] AVISO: log ausente:", path)
        return rounds, f1, fpr, modes
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = _ROUND.search(line)
        if m:
            rounds.append(int(m.group(1)))
            modes.append(m.group(2))
            f1.append(float(m.group(3)))
            fpr.append(float(m.group(5)))
    return rounds, f1, fpr, modes


def parse_trail(path):
    switches = []
    if not os.path.exists(path):
        print("[plot] AVISO: trilha ausente:", path)
        return switches
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("type") == "architecture_change":
            switches.append((ev["round"], ev["from_mode"], ev["to_mode"],
                             ev.get("reason")))
    return switches


def main():
    ar, af1, afpr, amodes = parse_log(ADAPT_LOG)
    sr, sf1, sfpr, _ = parse_log(STATIC_LOG)
    switches = parse_trail(TRAIL)
    if not ar:
        sys.exit("[plot] sem dados na run adaptativa — nada a plotar.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    a2r, a2f1, a2fpr, _ = parse_log(_a.adapt2) if _a.adapt2 else ([], [], [], [])

    # F1
    ax1.plot(ar, af1, "-o", ms=3, lw=1.8, color="#1b7837", label=_a.adapt_label)
    if a2r:
        ax1.plot(a2r, a2f1, "-.^", ms=3, lw=1.5, color="#e08214",
                 label=_a.adapt2_label)
    if sr:
        ax1.plot(sr, sf1, "--s", ms=3, lw=1.5, color="#b2182b",
                 label="Estático federado (piso)")
    ax1.set_ylabel("F1-score (%)")
    ax1.set_title(_a.title)

    # FPR
    ax2.plot(ar, afpr, "-o", ms=3, lw=1.8, color="#1b7837", label=_a.adapt_label)
    if a2r:
        ax2.plot(a2r, a2fpr, "-.^", ms=3, lw=1.5, color="#e08214", label=_a.adapt2_label)
    if sr:
        ax2.plot(sr, sfpr, "--s", ms=3, lw=1.5, color="#b2182b", label="Estático")
    ax2.set_ylabel("FPR (%)")
    ax2.set_xlabel("round")

    # sombreia a fase GOSSIP do adaptativo
    gl_rounds = [r for r, m in zip(ar, amodes) if m == "gossip"]
    if gl_rounds:
        for ax in (ax1, ax2):
            ax.axvspan(min(gl_rounds) - 0.5, max(gl_rounds) + 0.5,
                       color="#1b7837", alpha=0.06,
                       label="_fase gossip (adaptativo)")

    # faltas injetadas (ground truth do ambiente)
    for r, txt in FAULTS.items():
        for ax in (ax1, ax2):
            ax.axvline(r, color="0.5", ls=":", lw=1)
        ax1.annotate(txt, xy=(r, ax1.get_ylim()[0]), xytext=(r, ax1.get_ylim()[0]),
                     rotation=90, va="bottom", ha="right", fontsize=7, color="0.4")

    # commits de troca detectados (monitor)
    for r, fm, tm, reason in switches:
        for ax in (ax1, ax2):
            ax.axvline(r, color="#2166ac", ls="-", lw=1.2, alpha=0.8)
        ax1.annotate(f"{fm}→{tm}\n({reason})", xy=(r, ax1.get_ylim()[1]),
                     xytext=(r + 0.3, ax1.get_ylim()[1]), fontsize=7,
                     color="#2166ac", va="top")

    ax1.legend(loc="lower left", fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=150)
    fig.savefig(OUT + ".pdf")
    print("[plot] ->", OUT + ".png / .pdf")
    print("[plot] switches:", switches)
    if sr:
        import statistics as st
        print("[plot] F1 médio  adapt=%.2f  static=%.2f" %
              (st.mean(af1), st.mean(sf1)))
        print("[plot] F1 mínimo adapt=%.2f  static=%.2f" %
              (min(af1), min(sf1)))


if __name__ == "__main__":
    main()
