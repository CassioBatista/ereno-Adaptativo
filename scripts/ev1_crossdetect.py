#!/usr/bin/env python3
"""Cross-detection matrix of the step-1 specialists (seed 42), the four domains.

For every attack type t and every specialist s: the share of the test rows of t that s flags
(score >= 0.5). Each specialist was trained on its own type(s) against the benign rows of its
node only, so it flags whatever departs from benign traffic the way its type does.

Derived per type (cover threshold 50 %):
  - own: recall of its own specialist;
  - covers: specialists other than its own that flag at least 50 % of its rows, and the nodes
    that hold them (the redundancy that keeps the type detected when its node is lost);
  - group: connected components of the graph in which two types are linked when one precise
    specialist (benign false-positive rate <= 1 %) covers both: a behaviour shared under the
    cover threshold. A specialist that flags many benign rows flags almost everything, so it
    would link types without a shared behaviour; it is still listed as covering.
Benign: the false-positive rate of each specialist.

No training; reads results/.cache/ev1_step1_<domain>_s42.npz and results/ev1_step1_<domain>_s42.json.
  python scripts/ev1_crossdetect.py [--cover 0.5]
Out: results/ev1_crossdetect_<domain>.{csv,txt}, results/ev1_crossdetect.{pdf,png}
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017"), ("morris4", "Morris-4"), ("ciciot", "CICIoT2023")]
PRECISE = 0.01      # a specialist links types only if it flags at most 1 % of the benign rows


def components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in edges:
        parent[find(a)] = find(b)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def run(dom, name, cover):
    z = np.load(os.path.join(ROOT, f"results/.cache/ev1_step1_{dom}_s42.npz"))
    hit, y, nc = z["Ps"] >= 0.5, z["y"], int(z["nc"])
    classes = [str(c) for c in z["classes"]]
    spec = json.load(open(os.path.join(ROOT, f"results/ev1_step1_{dom}_s42.json")))["specialists"]
    assert len(spec) == hit.shape[1]
    types = [c for c in range(len(classes)) if c != nc and (y == c).any()]
    own_of = {c: [s["j"] for s in spec if c in s["codes"]] for c in types}
    M = np.array([[hit[y == c, s["j"]].mean() for s in spec] for c in types])   # types x specialists
    fpr = np.array([hit[y == nc, s["j"]].mean() for s in spec])
    n_rows = np.array([(y == c).sum() for c in types])
    cov = M >= cover
    precise = fpr <= PRECISE                       # a specialist that flags many benign rows
    covp = cov & precise[None, :]                  # flags everything; it links no behaviour
    edges = [(i, k) for s in range(len(spec)) for i in np.where(covp[:, s])[0]
             for k in np.where(covp[:, s])[0] if i < k]
    grp = components(len(types), edges)
    info = []
    for i, c in enumerate(types):
        own = own_of[c]
        own_nodes = {spec[j]["node"] for j in own}
        others = [s["j"] for s in spec if cov[i, s["j"]] and s["j"] not in own]
        others_p = [j for j in others if precise[j]]
        info.append({"type": classes[c], "rows": int(n_rows[i]),
                     "own": float(max(M[i, j] for j in own)),
                     "own_node": sorted(own_nodes),
                     "covered_by": [spec[j]["name"] for j in others_p],
                     "cover_nodes": sorted({spec[j]["node"] for j in others_p} - own_nodes),
                     "cover_nodes_any": sorted({spec[j]["node"] for j in others} - own_nodes),
                     "group": grp[i]})
    # groups: ordered by size, then rows
    gids = sorted(set(grp), key=lambda g: (-sum(1 for x in grp if x == g),
                                           -sum(n_rows[i] for i in range(len(types)) if grp[i] == g)))
    rank = {g: r for r, g in enumerate(gids)}
    for x in info:
        x["group"] = rank[x["group"]] + 1
    order = sorted(range(len(types)), key=lambda i: (info[i]["group"], -info[i]["rows"]))
    # specialist order follows the first type each one owns
    sorder, seen = [], set()
    for i in order:
        for j in own_of[types[i]]:
            if j not in seen:
                sorder.append(j)
                seen.add(j)
    sorder += [j for j in range(len(spec)) if j not in seen]

    with open(os.path.join(ROOT, f"results/ev1_crossdetect_{dom}.csv"), "w") as f:
        f.write("type,rows,group," + ",".join(f"spec{j:02d}_{spec[j]['name'][:40].replace(',', ';')}"
                                              for j in sorder) + "\n")
        for i in order:
            f.write(f"{classes[types[i]]},{n_rows[i]},{info[i]['group']}," +
                    ",".join(f"{M[i, j]:.4f}" for j in sorder) + "\n")
        f.write("benign_fpr,," + "," + ",".join(f"{fpr[j]:.6f}" for j in sorder) + "\n")

    ng = len(gids)
    att_rows = n_rows.sum()
    multi = [x for x in info if x["cover_nodes"]]
    multi_any = [x for x in info if x["cover_nodes_any"]]
    lines = [f"cross-detection -- {name} (step-1 specialists, seed 42; cover = flags >= {100 * cover:.0f} % "
             f"of the rows; precise = benign FPR <= {100 * PRECISE:g} %)",
             f"{len(types)} attack types, {len(spec)} specialists on 14 nodes, {int(precise.sum())} precise; "
             f"{ng} groups of types linked by a shared precise specialist",
             f"types covered by a precise specialist of another node: {len(multi)} of {len(types)} "
             f"({100 * sum(x['rows'] for x in multi) / att_rows:.1f} % of the attack rows); by any specialist "
             f"of another node: {len(multi_any)} ({100 * sum(x['rows'] for x in multi_any) / att_rows:.1f} %)", "",
             f"{'grp':>3} {'type':<34}{'rows':>9}{'own %':>7} {'own node':>9}  {'other nodes, precise':<22}"
             f"{'other nodes, any':<26} covered by (precise)"]
    for i in order:
        x = info[i]
        lines.append(f"{x['group']:>3} {x['type'][:33]:<34}{x['rows']:>9,}{100 * x['own']:>7.1f} "
                     f"{','.join(map(str, x['own_node'])):>9}  {','.join(map(str, x['cover_nodes'])) or '-':<22}"
                     f"{','.join(map(str, x['cover_nodes_any'])) or '-':<26} "
                     + ("; ".join(n[:24] for n in x["covered_by"]) or "-"))
    lines.append("\nper group: types, attack rows %, nodes holding a precise specialist that covers every type of the group")
    for g in range(1, ng + 1):
        idx = [i for i in range(len(types)) if info[i]["group"] == g]
        full = [s["node"] for s in spec if all(covp[i, s["j"]] for i in idx)]
        lines.append(f"  group {g:2d}: {len(idx):2d} types, {100 * n_rows[idx].sum() / att_rows:5.1f} % of rows; "
                     f"nodes covering all: {sorted(set(full)) or '-'}; "
                     + ", ".join(classes[types[i]][:22] for i in idx))
    lines.append("\nbenign false-positive rate per specialist (%): " + ", ".join(
        f"{spec[j]['name'][:18]} {100 * fpr[j]:.2f}" for j in sorder))
    txt = "\n".join(lines)
    open(os.path.join(ROOT, f"results/ev1_crossdetect_{dom}.txt"), "w").write(txt + "\n")
    print(txt + "\n")
    return {"M": M[np.ix_(order, sorder)], "types": [classes[types[i]] for i in order],
            "specs": [spec[j]["name"] for j in sorder], "groups": [info[i]["group"] for i in order],
            "own": [[1 if types[i] in spec[j]["codes"] else 0 for j in sorder] for i in order],
            "fpr": [float(fpr[j]) for j in sorder]}


def short(s, n=22):
    s = s.replace("_Attempted", " att.").replace("_", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281", "#0d366b"])
    ink2 = "#52514e"
    plt.rcParams.update({"font.size": 6, "xtick.color": ink2, "ytick.color": ink2})
    fig, axes = plt.subplots(2, 2, figsize=(11, 12))
    fig.subplots_adjust(wspace=0.45, hspace=0.32)
    axes = axes.ravel()
    for a, (dom, name) in zip(axes, DOMS):
        r = res[dom]
        im = a.imshow(100 * r["M"], cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="none")
        own = np.array(r["own"])
        for i, j in zip(*np.where(own == 1)):
            a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=0.8))
        g = r["groups"]
        for i in range(1, len(g)):
            if g[i] != g[i - 1]:
                a.axhline(i - 0.5, color=ink2, lw=0.4)
        a.set_yticks(range(len(r["types"])))
        a.set_yticklabels([short(t) for t in r["types"]])
        a.set_xticks(range(len(r["specs"])))
        a.set_xticklabels([short(s, 16) for s in r["specs"]], rotation=90)
        a.set_title(name, fontsize=8)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.02)
    cb.set_label("Rows of the type (row) flagged by the specialist (column), %", fontsize=6)
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(ROOT, "results/ev1_crossdetect.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(ROOT, "results/ev1_crossdetect.png"), dpi=170, bbox_inches="tight")


def plot_nodes(res):
    """Compact version: columns = the 14 nodes (OR of the specialists each node trained), rows =
    types in the order of the full figure, plus a benign row (false-positive rate of each node)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281", "#0d366b"])
    ink2 = "#52514e"
    plt.rcParams.update({"font.size": 5, "xtick.color": ink2, "ytick.color": ink2})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.0), gridspec_kw={"wspace": 1.15})
    for a, (dom, name) in zip(axes, DOMS):
        z = np.load(os.path.join(ROOT, f"results/.cache/ev1_step1_{dom}_s42.npz"))
        hit, y, nc, node = z["Ps"] >= 0.5, z["y"], int(z["nc"]), z["node"]
        classes = [str(c) for c in z["classes"]]
        spec = json.load(open(os.path.join(ROOT, f"results/ev1_step1_{dom}_s42.json")))["specialists"]
        H = np.column_stack([hit[:, node == n].any(1) for n in range(1, 15)])
        types = res[dom]["types"]
        codes = [classes.index(t) for t in types]
        M = np.array([H[y == c].mean(0) for c in codes] + [H[y == nc].mean(0)])
        a.imshow(100 * M, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="none")
        for i, c in enumerate(codes):
            for n in {s["node"] for s in spec if c in s["codes"]}:
                a.add_patch(plt.Rectangle((n - 1.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=0.6))
        g = res[dom]["groups"]
        for i in range(1, len(g)):
            if g[i] != g[i - 1]:
                a.axhline(i - 0.5, color=ink2, lw=0.3)
        a.axhline(len(types) - 0.5, color=ink2, lw=0.8)
        for n in range(14):                                  # benign row: print FPR >= 1 %
            if M[-1, n] >= 0.01:
                a.text(n, len(types), f"{100 * M[-1, n]:.0f}", ha="center", va="center", fontsize=3.6,
                       color="#0b0b0b" if M[-1, n] < 0.5 else "#fcfcfb")
        abbr = lambda t: (t.replace("_Attempted", " att.").replace("Web_Attack_", "Web ")
                          .replace("masquerade_fake_", "masquerade ").replace("Infiltration_", "Infiltr. ")
                          .replace("Fragmentation", "Frag.").replace("DictionaryBruteForce", "Dictionary BF")
                          .replace("BrowserHijacking", "Browser Hijack").replace("specific_", ""))
        lab = [short(abbr(t), 19) for t in types] + ["benign (FPR %)"]
        a.set_yticks(range(len(lab)))
        a.set_yticklabels(lab)
        a.set_xticks(range(14))
        a.set_xticklabels(range(1, 15), fontsize=4)
        a.set_xlabel("Node", fontsize=5)
        a.set_title(name + (" (specific)" if dom == "morris4" else ""), fontsize=6.5)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(length=0, pad=1)
    fig.savefig(os.path.join(ROOT, "results/ev1_crossdetect_nodes.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(ROOT, "results/ev1_crossdetect_nodes.png"), dpi=300, bbox_inches="tight")


def plot_numbers(res, grid=(1, 4)):
    """Compact version of the full matrix: rows and columns labelled by attack number only
    (Morris-4: its own 'specific' number; other domains: 1..T in the row order of the figure);
    a column carries the number of the type its specialist was trained on ('sh' = specialist
    shared by several small types). Last row: false-positive rate of each specialist, printed
    when at least 1 %. Every panel is a square. Key: results/ev1_crossdetect_key.tex.
    grid (1, 4) -> ev1_crossdetect_num.*; (2, 2) -> ev1_crossdetect_num2x2.*"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281", "#0d366b"])
    ink2 = "#52514e"
    side = 105 if grid == (1, 4) else 215                  # points per panel side, for label sizes
    plt.rcParams.update({"font.size": 4.2, "xtick.color": ink2, "ytick.color": ink2})
    fig, axes = plt.subplots(*grid, figsize=(7.2, 2.2) if grid == (1, 4) else (6.4, 6.6),
                             gridspec_kw={"wspace": 0.22, "hspace": 0.28})
    axes = np.atleast_1d(axes).ravel()
    key = []
    for a, (dom, name) in zip(axes, DOMS):
        r = res[dom]
        types, own = r["types"], np.array(r["own"])
        num = [t.split("_")[1] for t in types] if dom == "morris4" else [str(i + 1) for i in range(len(types))]
        M = np.vstack([100 * r["M"], 100 * np.array(r["fpr"])[None, :]])
        a.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="none")
        for i, j in zip(*np.where(own == 1)):
            a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=0.5))
        g = r["groups"]
        for i in range(1, len(g)):
            if g[i] != g[i - 1]:
                a.axhline(i - 0.5, color=ink2, lw=0.3)
        a.axhline(len(types) - 0.5, color=ink2, lw=0.7)
        for j, f in enumerate(r["fpr"]):
            if f >= 0.01:
                a.text(j, len(types), f"{100 * f:.0f}", ha="center", va="center",
                       fontsize=min(4.5, 0.6 * side / len(r["fpr"])),
                       color="#0b0b0b" if f < 0.5 else "#fcfcfb")
        cl = []
        for j in range(own.shape[1]):
            rows = np.where(own[:, j] == 1)[0]
            cl.append(num[rows[0]] if len(rows) == 1 else "sh")
        a.set_box_aspect(1)                                  # every panel a square
        fy = min(6.0, 0.75 * side / (len(types) + 1))        # label size follows the cell size
        fx = min(6.0, 0.75 * side / len(cl))
        a.set_xticks(range(len(cl)))
        a.set_xticklabels(cl, rotation=90, fontsize=fx)
        a.set_yticks(range(len(types) + 1))
        a.set_yticklabels(num + ["B"], fontsize=fy)
        a.set_title(name, fontsize=6 if grid == (1, 4) else 8)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(length=0, pad=1)
        if dom != "morris4":
            sh = [num[i] for j in range(own.shape[1]) for i in np.where(own[:, j] == 1)[0]
                  if own[:, j].sum() > 1]
            items = "; ".join(f"{num[i]} {types[i].replace('_', ' ')}" for i in range(len(types)))
            key.append(f"% {name}\n\\emph{{{name}}}: {items}"
                       + (f"; sh: specialist shared by types {', '.join(sh)}" if sh else "") + ".")
    fs = 5 if grid == (1, 4) else 6.5
    for a in axes[::grid[1]]:
        a.set_ylabel("Attack type (row)", fontsize=fs)
    for a in axes[-grid[1]:]:
        a.set_xlabel("Specialist (column), by attack type", fontsize=fs)
    tag = "num" if grid == (1, 4) else "num2x2"
    fig.savefig(os.path.join(ROOT, f"results/ev1_crossdetect_{tag}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(ROOT, f"results/ev1_crossdetect_{tag}.png"), dpi=300, bbox_inches="tight")
    open(os.path.join(ROOT, "results/ev1_crossdetect_key.tex"), "w").write("\n".join(key) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", type=float, default=0.5)
    a = ap.parse_args()
    os.chdir(ROOT)
    res = {d: run(d, n, a.cover) for d, n in DOMS}
    plot(res)
    plot_nodes(res)
    plot_numbers(res)
    plot_numbers(res, grid=(2, 2))
