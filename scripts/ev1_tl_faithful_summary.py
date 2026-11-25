#!/usr/bin/env python3
"""Faithful replay of the Disaster-FD traces (paper 2, 5.3): ReSIDS (switching FL/GL on the trust
level), FL (fixed) and GL (fixed), all with retention, under the same network of the scenario
(links: scenario) and rounds at 60 r + 30 s; results/ev1_tl/faithful/<domain>_<trace>_<policy>.

1. Per run and late block (nodes 3, 4, 11, 12 at 30 min; nodes 9, 10 at 60 min): nodes protected
   at the end, median latency from arrival to protection, exposure (share of node-rounds after
   the arrival without protection) over all nodes and over the nodes that reach the aggregator
   or a ring neighbour (same network for every policy). Protected = recall on the block's types
   >= 95 % of the domain reference (largest recall on them in any run of the domain).
2. Regret: exposure minus the best of the three, over the cells where they differ by > 1 point.
3. The questions of scenario 2 (scenario2.md), from the holdings logs (who holds which booster):
   a. does the specialist of clients 3 and 4 leave the region {2..7} before it is lost
      (45 and 60 min)?
   b. does client 5 hold the specialist of clients 11 and 12 before 45 min?
   c. does the specialist of clients 9 and 10 reach clients 1 and 8 at 75 min?
   A booster belongs to a block when it first appears, at or after the block's arrival, in the
   holdings of one of the block's clients.
4. Figures: one per trace, the four domains, mean recall of the nodes on the attacks present
   in the test set at the round (mean_present; changed 2026-11-25 from the share of nodes
   protected against every late block arrived so far, which was 100 % by definition before
   30 min; protected_share is kept for the share of nodes protected against the present ones).

  python scripts/ev1_tl_faithful_summary.py
Out: results/ev1_tl_faithful_summary.{txt,csv}, results/ev1_tl_faithful_{T1,T2}.{pdf,png}
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
D = os.path.join(ROOT, "results/ev1_tl/faithful")
DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017"), ("morris4", "Morris-4"), ("ciciot", "CICIoT2023")]
TRACES = [("T1", 4980), ("T2", 5880)]
POLS = [("switching", "ReSIDS"), ("fixed_federated", "FL"), ("fixed_gossip", "GL")]
BLOCKS = {1800: (3, 4, 11, 12), 3600: (9, 10)}
REL = 0.95


def rdir(dom, tr, pol):
    return os.path.join(D, f"{dom}_{tr}_{pol}")


def have(dom, tr, pol):
    return os.path.exists(os.path.join(rdir(dom, tr, pol), "node_eval.csv"))


def node_eval(dom, tr, pol):
    f = lambda v: float(v) if v not in ("", "nan") else np.nan
    return [{"t": float(r["t_s"]), "node": int(r["node"]), "reach": r["reachable"] == "1",
             "present": f(r["recall_present"]), "mode": r["node_mode"],
             **{a: f(r[f"recall_late_{a}"]) for a in BLOCKS}}
            for r in csv.DictReader(open(os.path.join(rdir(dom, tr, pol), "node_eval.csv")))]


def period(t):
    """Start of the period of t: 0 (initial types only), or the arrival of the last block."""
    return max([0] + [a for a in BLOCKS if a <= t])


def ref_present(runs):
    """Reference recall on the attacks present, per period: the largest recall on them of any
    node in any run of the domain (as for the blocks)."""
    out = {}
    for p in [0] + list(BLOCKS):
        v = [r["present"] for rows in runs.values() for r in rows if period(r["t"]) == p]
        out[p] = np.nanmax(v) if v else np.nan
    return out


def mean_present(rows):
    """Figure: mean over the 14 nodes of the recall on the attacks present in the test set at
    the round (scenario documents: 4, then 6, then 7 attacks); per-sample recall."""
    return [(t, np.nanmean([r["present"] for r in rows if r["t"] == t])) for t in sorted({r["t"] for r in rows})]


def exchange_modes(rows):
    """Mode of ReSIDS per round, for the background of the figure: federated if any node
    federates, else gossip if any node gossips, else local."""
    by_t = {}
    for r in rows:
        by_t.setdefault(r["t"], set()).add(r["mode"])
    return [(t, "federated" if "federated" in m else "gossip" if "gossip" in m else "local")
            for t, m in sorted(by_t.items())]


def protected_share(rows, ref, refp):
    """Figure: share of nodes protected against the attacks present in the test set at the
    round (scenario documents: 4, then 6, then 7 attacks): recall on the present attacks
    >= REL of the period's reference, and recall on every late block arrived >= REL of its
    reference (the per-sample recall on the present attacks alone could hide a small block)."""
    out = []
    for t in sorted({r["t"] for r in rows}):
        rt = [r for r in rows if r["t"] == t]
        out.append((t, np.mean([r["present"] >= REL * refp[period(t)] and
                                all(r[a] >= REL * ref[a] for a in BLOCKS if a <= t) for r in rt])))
    return out


def block_boosters(rounds, arrival, clients, others=()):
    """Hashes that first appear, at or after `arrival`, in the holdings of one of `clients` and
    of none of `others` (clients of another block arriving at the same time; a hash first seen
    in both, as after a federated round, cannot be attributed and is left out)."""
    seen, out = set(), set()
    for r in rounds:
        first = {}
        for i, hs in enumerate(r["holdings"]):
            for h in hs:
                if h not in seen:
                    first.setdefault(h, set()).add(i + 1)
        if r["t_s"] >= arrival:
            out |= {h for h, who in first.items() if who & set(clients) and not who & set(others)}
        for hs in r["holdings"]:
            seen.update(hs)
    return out


def holders(rounds, t_before, hashes):
    """Clients holding any of `hashes` at the last round before t_before (or at the end)."""
    rr = [r for r in rounds if r["t_s"] < t_before] if t_before else rounds
    if not rr or not hashes:
        return []
    last = rr[-1]
    return [i + 1 for i, hs in enumerate(last["holdings"]) if set(hs) & hashes]


def main():
    lines, out, cells, curves = ["faithful replay (links: scenario, rounds at 60 r + 30 s, retention); "
                                 "per block: protected at end / median latency / exposure all / reachable", ""], [], {}, {}
    gains, modes = {}, {}
    for dom, dn in DOMS:
        runs = {(tr, pol): node_eval(dom, tr, pol) for tr, _ in TRACES for pol, _ in POLS if have(dom, tr, pol)}
        if not runs:
            continue
        ref = {a: max(np.nanmax([r[a] for r in rows]) for rows in runs.values()) for a in BLOCKS}
        refp = ref_present(runs)
        lines.append(f"==== {dn} (reference: block 30 min {100 * ref[1800]:.2f} %, block 60 min {100 * ref[3600]:.2f} %; "
                     f"attacks present 0/30/60 min {100 * refp[0]:.2f} / {100 * refp[1800]:.2f} / {100 * refp[3600]:.2f} %)")
        for tr, _ in TRACES:                                  # figure, lower row: what switching adds
            if all((tr, p) in runs for p, _ in POLS):
                ps = {p: dict(protected_share(runs[(tr, p)], ref, refp)) for p, _ in POLS}
                ts = sorted(ps["switching"])
                gains[(dom, tr)] = {k: [(t, 100 * (ps["switching"][t] - ps[p][t])) for t in ts]
                                    for k, p in (("FL", "fixed_federated"), ("GL", "fixed_gossip"))}
                if tr not in modes:
                    modes[tr] = exchange_modes(runs[(tr, "switching")])
        for tr, tend in TRACES:
            lines.append(f"--- {tr}")
            for pol, lab in POLS:
                if (tr, pol) not in runs:
                    lines.append(f"  {lab:<7} (missing)")
                    continue
                rows = runs[(tr, pol)]
                ts = sorted({r["t"] for r in rows})
                last = [r for r in rows if r["t"] == ts[-1]]
                rec, txt = {"domain": dn, "trace": tr, "config": lab}, ""
                for a in BLOCKS:
                    thr = REL * ref[a]
                    first = {}
                    for r in sorted(rows, key=lambda r: r["t"]):
                        if r["t"] >= a and r[a] >= thr and r["node"] not in first:
                            first[r["node"]] = r["t"]
                    lat = [first[n] - a for n in first]
                    after = [r for r in rows if r["t"] >= a]
                    reach = [r for r in after if r["reach"]]
                    e_all = float(np.mean([not (r[a] >= thr) for r in after]))
                    e_rch = float(np.mean([not (r[a] >= thr) for r in reach])) if reach else np.nan
                    prot = sum(1 for r in last if r[a] >= thr)
                    mlat = np.median(lat) / 60 if lat else np.nan
                    rec.update({f"b{a // 60}_protected_end": prot, f"b{a // 60}_median_latency_min": mlat,
                                f"b{a // 60}_exposure": e_all, f"b{a // 60}_exposure_reachable": e_rch})
                    cells.setdefault((dn, tr, a), {})[lab] = 100 * e_all
                    txt += (f"  | {a // 60} min: {prot:2d}/14 {('-' if np.isnan(mlat) else f'{mlat:.0f} min'):>7} "
                            f"{100 * e_all:5.1f}% {100 * e_rch:5.1f}%")
                out.append(rec)
                lines.append(f"  {lab:<7}{txt}")
                curves[(dom, tr, lab)] = mean_present(rows)
        lines.append("")
    inf = {k: v for k, v in cells.items() if len(v) == 3 and max(v.values()) - min(v.values()) > 1.0}
    lines.append(f"regret (exposure minus the best of the three), {len(inf)} informative cells of {len(cells)}")
    for k in sorted(inf):
        v = inf[k]
        b = min(v.values())
        lines.append(f"  {k[0]:<11}{k[1]} block {k[2] // 60} min: " + "  ".join(
            f"{p} {v[p]:5.1f} (+{v[p] - b:.1f})" for _, p in POLS))
    for _, p in POLS:
        if inf:
            reg = [v[p] - min(v.values()) for v in inf.values()]
            worst = sum(1 for v in inf.values() if v[p] == max(v.values()))
            best = sum(1 for v in inf.values() if v[p] - min(v.values()) <= 1.0)
            lines.append(f"  {p:<7} mean {np.mean(reg):5.1f}, max {np.max(reg):5.1f}; best (within 1) {best}, worst {worst}")
    lines.append("\ngain of ReSIDS over each fixed mode (figure, lower row): points of nodes protected against "
                 "the attacks present, mean / max / min over the rounds")
    for (dom, tr), g in sorted(gains.items()):
        lines.append(f"  {dict(DOMS)[dom]:<11}{tr} " + "  ".join(
            f"vs {k}: {np.mean([v for _, v in g[k]]):5.1f} / {max(v for _, v in g[k]):5.1f} / {min(v for _, v in g[k]):6.1f}"
            for k in ("FL", "GL")))
    lines.append("\nscenario 2 questions (holdings logs): clients holding the block's specialists")
    for dom, dn in DOMS:
        for pol, lab in POLS:
            if not have(dom, "T2", pol):
                continue
            rounds = [json.loads(l) for l in open(os.path.join(rdir(dom, "T2", pol), "rounds.jsonl"))]
            b34 = block_boosters(rounds, 1800, {3, 4}, {11, 12})
            b1112 = block_boosters(rounds, 1800, {11, 12}, {3, 4})
            b910 = block_boosters(rounds, 3600, {9, 10})
            out_region = lambda hs: [c for c in hs if c not in range(2, 8)]
            h45, h60, hend = holders(rounds, 2700, b34), holders(rounds, 3600, b34), holders(rounds, None, b34)
            q2 = 5 in holders(rounds, 2700, b1112)
            q3 = holders(rounds, None, b910)
            lines.append(f"  {dn:<11}{lab:<7} a) 3-4 outside region: before 45 min {out_region(h45)}, "
                         f"before 60 min {out_region(h60)}, end {out_region(hend)} | b) client 5 holds 11-12 "
                         f"before 45 min: {q2} | c) 9-10 at the end held by {q3} (1 and 8: "
                         f"{1 in q3 and 8 in q3})")
    txt = "\n".join(lines)
    open(os.path.join(ROOT, "results/ev1_tl_faithful_summary.txt"), "w").write(txt + "\n")
    if out:
        with open(os.path.join(ROOT, "results/ev1_tl_faithful_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            for r in out:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(txt)
    plot(curves, gains, modes)


def plot(curves, gains, modes):
    """One figure per trace, two rows over the four domains: above, the mean recall of the nodes
    on the attacks present; below, the gain of ReSIDS over each fixed mode in points of nodes
    protected against the attacks present, over the mode ReSIDS takes in each round."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    sel = [("ReSIDS", "ReSIDS", "#2a78d6", "-", 2.6, 1),
           ("GL", "GL (fixed)", "#1baf7a", "-.", 1.3, 2),
           ("FL", "FL (fixed)", "#eb6834", (0, (3, 2)), 1.3, 3)]
    gsel = [("FL", "ReSIDS − FL (fixed)", "#eb6834"), ("GL", "ReSIDS − GL (fixed)", "#1baf7a")]
    band = [("local", "ReSIDS in local learning", "#ecebe7"), ("federated", "ReSIDS in FL", "#dce8f7"),
            ("gossip", "ReSIDS in GL", "#dcf1e8")]
    ink2 = "#52514e"
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.linewidth": 0.6})
    for tr, tend in TRACES:
        if not any(k[1] == tr for k in curves):
            continue
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), sharey="row", sharex="col")
        ax = axes[0]
        for j, (d, dn) in enumerate(DOMS):
            g = axes[1][j]
            ms = modes.get(tr, [])
            for i, (t, m) in enumerate(ms):
                t1 = ms[i + 1][0] if i + 1 < len(ms) else tend
                g.axvspan(0 if i == 0 else t / 60, t1 / 60, color=dict((b[0], b[2]) for b in band)[m], lw=0, zorder=0)
            g.axhline(0, color=ink2, lw=0.6, zorder=1)
            for key, lab, c in gsel:
                pts = gains.get((d, tr), {}).get(key)
                if pts:
                    g.step([t / 60 for t, _ in pts], [v for _, v in pts], where="post", color=c, lw=1.4,
                           label=lab, zorder=2)
            for x in (30, 60):
                g.axvline(x, color="#9c9b94", lw=0.6, ls=":", zorder=1)
            lo = min([v for k in ("FL", "GL") for _, v in gains.get((d, tr), {}).get(k, [])] + [0])
            g.set_ylim(-5 if lo >= 0 else -25, 105)    # negative gains shown only if any
            g.set_xlim(0, tend / 60)
            g.set_xticks(range(0, int(tend / 60) + 1, 15))
            for sp in ("top", "right"):
                g.spines[sp].set_visible(False)
            g.set_xlabel("Time (min)")
            if j == 0:
                g.set_ylabel("Gain of ReSIDS, nodes\nprotected (points)")
            a = ax[j]
            for key, lab, c, ls, lw, z in sel:
                pts = curves.get((d, tr, key))
                if pts:
                    a.step([t / 60 for t, _ in pts], [100 * v for _, v in pts], where="post", color=c,
                           ls=ls, lw=lw, label=lab, zorder=z)
            for x in (30, 60):
                a.axvline(x, color="#c3c2b7", lw=0.6, zorder=0)
            # first 15 min: trust levels being built, every node learns locally (both traces)
            a.axvspan(0, 15, color="#ecebe7", lw=0, zorder=0)
            a.text(7.5, 99, "Trust levels forming;\nlocal training", rotation=90, ha="center",
                   va="top", fontsize=4.5, color=ink2, zorder=0.5)
            a.set_ylim(-3, 103)
            a.set_xlim(0, tend / 60)
            a.set_xticks(range(0, int(tend / 60) + 1, 15))
            a.grid(axis="y", color="#e4e3df", lw=0.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.set_title(dn, fontsize=8)
            if j == 0:
                a.set_ylabel("Mean recall on\npresent attacks (%)")
        h, l = ax[0].get_legend_handles_labels()
        h2, l2 = axes[1][0].get_legend_handles_labels()
        # three columns (filled by column): configurations | gains | modes of ReSIDS
        h += h2 + [Patch(alpha=0)] + [Patch(color=c) for _, _, c in band]
        l += l2 + [""] + [lab for _, lab, _ in band]
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5)
        fig.tight_layout(rect=(0, 0, 1, 0.87))
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(ROOT, f"results/ev1_tl_faithful_{tr}.{ext}"), dpi=220)


if __name__ == "__main__":
    main()
