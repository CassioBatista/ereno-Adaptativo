#!/usr/bin/env python3
"""Consolidation of the trust-level replay grid (evaluation-v1 section 9; paper 2, 5.3).

Input: results/ev1_tl/grid_d60/<domain>_<trace>_<policy>_<holdings>_d60/node_eval.csv
(scripts/ev1_tl_node_eval.py: per round and node, recall on the types present, recall on the
types of each late block, FPR, reachable flag).

A node is PROTECTED against a late block (arrival a = 1800 or 3600 s) at a round when its
recall on the types of that block is at least 95 % of the reference recall on them, the
largest recall any node reaches on them in any run of the domain (the detector that holds
every specialist). Per run and block:
  - nodes protected at the end of the trace, and the reachable / unreachable split;
  - median latency from arrival to protection over the nodes protected by the end;
  - exposure: node-minutes from arrival to the end of the trace without protection, as a
    share of the node-minutes available (14 nodes x (t_end - a)).
Per run: mean and minimum over the nodes, at the last round, of the recall on all types and
of the FPR.

  python scripts/ev1_tl_grid_summary.py
Out: results/ev1_tl_grid_summary.{txt,csv}, results/ev1_tl_grid.{pdf,png}
"""
import csv
import os

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
G = os.path.join(ROOT, "results/ev1_tl/grid_d60")
DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017"), ("morris4", "Morris-4"), ("ciciot", "CICIoT2023")]
TRACES = [("T1", 4980), ("T2", 5880)]
CONFIGS = [("switching", "retain", "ReSIDS (switching, retention)"),
           ("switching", "reformed", "Switching, re-formed"),
           ("fixed_local", "retain", "Local only"),
           ("fixed_gossip", "retain", "Gossip fixed"),
           ("fixed_federated", "retain", "FL fixed, retention"),
           ("fixed_federated", "reformed", "FL fixed, re-formed")]
BLOCKS = (1800, 3600)
REL = 0.95


def load(dom, tr, pol, hold):
    rows = list(csv.DictReader(open(os.path.join(G, f"{dom}_{tr}_{pol}_{hold}_d60", "node_eval.csv"))))
    f = lambda v: float(v) if v not in ("", "nan") else np.nan
    return [{"round": int(r["round"]), "t": float(r["t_s"]), "node": int(r["node"]),
             "reach": r["reachable"] == "1", "rec": f(r["recall_present"]), "fpr": f(r["fpr"]),
             **{a: f(r[f"recall_late_{a}"]) for a in BLOCKS}} for r in rows]


def main():
    data = {(d, tr, p, h): load(d, tr, p, h) for d, _ in DOMS for tr, _ in TRACES for p, h, _ in CONFIGS}
    ref = {d: {a: np.nanmax([r[a] for (dd, *_), rows in data.items() if dd == d for r in rows])
               for a in BLOCKS} for d, _ in DOMS}
    out, lines, curves = [], [], {}
    lines.append(f"trust-level replay grid, delta 60 s, seed 42; protected = recall on the block's types "
                 f">= {100 * REL:.0f} % of the domain reference")
    for d, dn in DOMS:
        lines.append(f"\n==== {dn}  (reference recall: block 30 min {100 * ref[d][1800]:.2f} %, "
                     f"block 60 min {100 * ref[d][3600]:.2f} %)")
        for tr, tend in TRACES:
            lines.append(f"--- trace {tr}")
            lines.append(f"{'configuration':<31}{'end: recall mean/min':>22}{'FPR mean %':>12}"
                         + "".join(f"{f'| {a // 60} min: prot. end (U)':>24}{'median lat.':>12}{'exposure':>10}"
                                   for a in BLOCKS))
            for p, h, label in CONFIGS:
                rows = data[(d, tr, p, h)]
                ts = sorted({r["t"] for r in rows})
                last = [r for r in rows if r["t"] == ts[-1]]
                rec = [r["rec"] for r in last]
                rowd = {"domain": dn, "trace": tr, "config": label,
                        "end_recall_mean": np.mean(rec), "end_recall_min": np.min(rec),
                        "end_fpr_mean": np.mean([r["fpr"] for r in last])}
                cells = ""
                for a in BLOCKS:
                    thr = REL * ref[d][a]
                    prot = {}                               # node -> first time protected
                    for r in sorted(rows, key=lambda r: r["t"]):
                        if r["t"] >= a and not np.isnan(r[a]) and r[a] >= thr and r["node"] not in prot:
                            prot[r["node"]] = r["t"]
                    end_prot = [r["node"] for r in last if r[a] >= thr]
                    unreach = [r["node"] for r in last if not r["reach"]]
                    lat = [prot[n] - a for n in prot]
                    # exposure: node-rounds after a without protection, x delta
                    after = [r for r in rows if r["t"] >= a]
                    expo = sum(1 for r in after if not (r[a] >= thr)) / max(len(after), 1)
                    rowd.update({f"b{a}_protected_end": len(end_prot),
                                 f"b{a}_protected_end_unreachable": len(set(end_prot) & set(unreach)),
                                 f"b{a}_unreachable_end": len(unreach),
                                 f"b{a}_median_latency_min": np.median(lat) / 60 if lat else np.nan,
                                 f"b{a}_exposure": expo})
                    cells += (f"{f'| {len(end_prot):2d}/14 ({len(set(end_prot) & set(unreach))}/{len(unreach)} U)':>24}"
                              f"{(f'{np.median(lat) / 60:.0f} min' if lat else '-'):>12}{100 * expo:>9.1f}%")
                    # curve: share of the 14 nodes protected against every block arrived by t
                key = (d, tr, label)
                curves[key] = []
                for t in ts:
                    rr = [r for r in rows if r["t"] == t]
                    ok = [all(r[a] >= REL * ref[d][a] for a in BLOCKS if a <= t) for r in rr]
                    curves[key].append((t, np.mean(ok)))
                out.append(rowd)
                lines.append(f"{label:<31}{f'{100 * np.mean(rec):.2f}/{100 * np.min(rec):.2f}':>22}"
                             f"{100 * rowd['end_fpr_mean']:>12.3f}{cells}")
    lines.append("\nprot. end: nodes protected at the last round (of them, unreachable / unreachable nodes); "
                 "median lat.: from arrival to protection, nodes protected by the end; exposure: share of the "
                 "node-rounds after arrival without protection.")
    txt = "\n".join(lines)
    open(os.path.join(ROOT, "results/ev1_tl_grid_summary.txt"), "w").write(txt + "\n")
    with open(os.path.join(ROOT, "results/ev1_tl_grid_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(txt)
    plot(curves)
    plot_main(curves)


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = ["#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#e87ba4", "#008300"]   # categorical slots 1-6
    sty = ["-", "--", ":", "-.", "-", "--"]
    ink2 = "#52514e"
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.linewidth": 0.6})
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex="row", sharey=True)
    for i, (tr, tend) in enumerate(TRACES):
        for j, (d, dn) in enumerate(DOMS):
            a = ax[i, j]
            for k, (_, _, label) in enumerate(CONFIGS):
                c = curves[(d, tr, label)]
                a.step([t / 60 for t, _ in c], [100 * v for _, v in c], where="post", color=col[k],
                       ls=sty[k], lw=1.8 if k == 0 else 1.1, label=label, zorder=3 if k == 0 else 2)
            for x in (30, 60):
                a.axvline(x, color="#c3c2b7", lw=0.6, zorder=1)
            a.set_ylim(-3, 103)
            a.set_xlim(0, tend / 60)
            a.grid(axis="y", color="#e4e3df", lw=0.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            if i == 0:
                a.set_title(dn, fontsize=7.5)
            if j == 0:
                a.set_ylabel(f"{tr}: nodes protected (%)")
            if i == 1:
                a.set_xlabel("Time (min)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"results/ev1_tl_grid_all.{ext}"), dpi=200)


def plot_main(curves):
    """Paper figure: ReSIDS against the two fixed modes that exchange models (gossip, FL with
    retention), ERENO and CICIDS2017 (in Morris-4 and CICIoT2023 the late types are detected by
    the specialists already present). FL fixed is drawn dashed over ReSIDS, so the stretches where
    the two coincide stay visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [("ReSIDS (switching, retention)", "ReSIDS", "#2a78d6", "-", 2.6, 1),
           ("Gossip fixed", "Fixed GL", "#1baf7a", "-.", 1.3, 2),
           ("FL fixed, retention", "Fixed FL", "#eb6834", (0, (3, 2)), 1.3, 3)]
    ink2 = "#52514e"
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.linewidth": 0.6})
    for tr, tend in TRACES:                      # one figure per trace, the four domains
        fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
        for j, (d, dn) in enumerate(DOMS):
            a = ax[j]
            for key, lab, c, ls, lw, z in sel:
                pts = curves[(d, tr, key)]
                a.step([t / 60 for t, _ in pts], [100 * v for _, v in pts], where="post", color=c,
                       ls=ls, lw=lw, label=lab, zorder=z)
            for x in (30, 60):
                a.axvline(x, color="#c3c2b7", lw=0.6, zorder=0)
            a.set_ylim(-3, 103)
            a.set_xlim(0, tend / 60)
            a.set_xticks(range(0, int(tend / 60) + 1, 15))
            a.grid(axis="y", color="#e4e3df", lw=0.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.set_title(dn, fontsize=8)
            a.set_xlabel("Time (min)")
            if j == 0:
                a.set_ylabel("Nodes protected (%)")
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(ROOT, f"results/ev1_tl_grid_{tr}.{ext}"), dpi=220)


if __name__ == "__main__":
    main()
