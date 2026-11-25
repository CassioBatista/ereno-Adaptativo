#!/usr/bin/env python3
"""Static node loss, evaluation-v1 section 5 (paper 2: opening of the mode-switching results).

From the cached step-1 scores (results/.cache/ev1_step1_<domain>_s42.npz: one score column per
specialist, its node, the test labels), nodes are removed in a fixed order, a permutation of the
14 nodes drawn with numpy default_rng(42), from 14 survivors down to 3:
  - retention: every survivor keeps the specialists it received before the loss, so the fused
    detector is the step-1 attack-based OR, unchanged (assumes the exchange completed);
  - re-formed: the aggregate holds only the specialists of the nodes still present.
Fusion by OR at 0.5 (k = 1). Reported at every N: recall, false-positive rate, per-type recall,
and the attack types left without their own specialist. The same curves over 1,000 further
random orders (default_rng(2026)) give the spread that the fixed order sits in.

No training and no model is read: the detector of any node subset is the OR of its specialists'
hit columns, computed from the rows grouped by hit pattern.

  python scripts/ev1_nodeloss.py [--orders 1000]
Out: results/ev1_nodeloss_<domain>.{json,txt}, results/ev1_nodeloss.{pdf,png}
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/ereno-Adaptativo")
DOMS = [("ereno", "ERENO"), ("cicids", "CICIDS2017"), ("morris4", "Morris-4"), ("ciciot", "CICIoT2023")]
N_NODES, N_MIN, SEED, SEED_ORDERS = 14, 3, 42, 2026


def grouped(dom):
    z = np.load(os.path.join(ROOT, f"results/.cache/ev1_step1_{dom}_s42.npz"))
    hit, y, nc = z["Ps"] >= 0.5, z["y"].astype(np.int64), int(z["nc"])
    node, classes = z["node"].astype(int), [str(c) for c in z["classes"]]
    S = hit.shape[1]
    assert S < 63
    code = (hit.astype(np.int64) << np.arange(S, dtype=np.int64)).sum(1)
    key, cnt = np.unique(np.stack([code, y], 1), axis=0, return_counts=True)
    return {"code": key[:, 0], "y": key[:, 1], "cnt": cnt, "nc": nc, "node": node,
            "classes": classes, "S": S}


def mask_of(g, survivors):
    bits = [j for j in range(g["S"]) if g["node"][j] in survivors]
    return int(np.sum(np.int64(1) << np.array(bits, dtype=np.int64))) if bits else 0


def evaluate(g, survivors, per_type=False):
    det = (g["code"] & mask_of(g, survivors)) != 0
    att, cnt = g["y"] != g["nc"], g["cnt"]
    out = {"recall": float(cnt[det & att].sum() / cnt[att].sum()),
           "fpr": float(cnt[det & ~att].sum() / cnt[~att].sum())}
    if per_type:
        out["per_type"] = {}
        for c in sorted(set(g["y"][att].tolist())):
            m = g["y"] == c
            out["per_type"][g["classes"][c]] = float(cnt[m & det].sum() / cnt[m].sum())
    return out


def main(n_orders):
    order = np.random.default_rng(SEED).permutation(N_NODES) + 1        # removal order
    rng = np.random.default_rng(SEED_ORDERS)
    others = [rng.permutation(N_NODES) + 1 for _ in range(n_orders)]
    Ns = list(range(N_NODES, N_MIN - 1, -1))
    summary = {}
    for dom, name in DOMS:
        g = grouped(dom)
        spec = json.load(open(os.path.join(ROOT, f"results/ev1_step1_{dom}_s42.json")))
        cover = {}                                    # attack type -> nodes holding its specialist
        for u in spec["specialists"]:
            for c in u["codes"]:
                cover.setdefault(g["classes"][c], set()).add(int(u["node"]))
        assert [u["node"] for u in spec["specialists"]] == g["node"].tolist()
        att_rows = {t: v["n"] for t, v in spec["results"]["IID OR"]["per_type"].items()}
        full = evaluate(g, set(range(1, N_NODES + 1)), per_type=True)
        rows = []
        for N in Ns:
            surv = set(order[N_NODES - N:].tolist())
            r = evaluate(g, surv, per_type=True)
            lost = sorted(t for t, ns in cover.items() if not ns & surv)
            spread = np.array([[evaluate(g, set(o[N_NODES - N:].tolist()))[k] for k in ("recall", "fpr")]
                               for o in others])
            rows.append({"N": N, "removed": sorted(order[:N_NODES - N].tolist()), **r,
                         "types_without_specialist": lost,
                         "rows_without_specialist": int(sum(att_rows.get(t, 0) for t in lost)),
                         "orders_recall_p5_p95": np.percentile(spread[:, 0], [5, 95]).tolist(),
                         "orders_recall_min_max": [float(spread[:, 0].min()), float(spread[:, 0].max())],
                         "orders_fpr_p5_p95": np.percentile(spread[:, 1], [5, 95]).tolist()})
        total_att = sum(att_rows.values())
        res = {"domain": name, "removal_order": order.tolist(), "seed_order": SEED,
               "seed_other_orders": SEED_ORDERS, "n_other_orders": n_orders,
               "retention": {"recall": full["recall"], "fpr": full["fpr"]},
               "re_formed": rows, "attack_test_rows": total_att, "type_cover": {
                   t: sorted(ns) for t, ns in cover.items()}}
        json.dump(res, open(os.path.join(ROOT, f"results/ev1_nodeloss_{dom}.json"), "w"), indent=1)
        lines = [f"evaluation-v1 section 5 -- static node loss -- {name} (step-1 scores, seed 42)",
                 f"removal order (default_rng({SEED}).permutation): {order.tolist()}",
                 f"retention (every N): recall {100 * full['recall']:.2f}  FPR {100 * full['fpr']:.3f} %",
                 f"re-formed; spread over {n_orders} other orders (default_rng({SEED_ORDERS})): "
                 "recall p5-p95 and min-max, FPR p5-p95", "",
                 f"{'N':>3} {'recall':>8} {'FPR %':>8} {'types lost':>10} {'rows lost %':>11}"
                 f"  {'recall p5-p95':>15} {'min-max':>15} {'FPR p5-p95':>15}  removed"]
        for r in rows:
            lines.append(f"{r['N']:>3} {100 * r['recall']:8.2f} {100 * r['fpr']:8.3f} "
                         f"{len(r['types_without_specialist']):>10} "
                         f"{100 * r['rows_without_specialist'] / total_att:11.2f}  "
                         f"{100 * r['orders_recall_p5_p95'][0]:7.2f}-{100 * r['orders_recall_p5_p95'][1]:6.2f} "
                         f"{100 * r['orders_recall_min_max'][0]:7.2f}-{100 * r['orders_recall_min_max'][1]:6.2f} "
                         f"{100 * r['orders_fpr_p5_p95'][0]:7.3f}-{100 * r['orders_fpr_p5_p95'][1]:6.3f}  "
                         f"{r['removed']}")
        lines.append("\nper-type recall % of the re-formed aggregate (types without specialist marked *)")
        types = list(full["per_type"])
        lines.append(f"{'type':<36}" + "".join(f"{r['N']:>7}" for r in rows))
        for t in types:
            lines.append(f"{t[:35]:<36}" + "".join(
                f"{100 * r['per_type'][t]:6.1f}" + ("*" if t in r["types_without_specialist"] else " ")
                for r in rows))
        txt = "\n".join(lines)
        open(os.path.join(ROOT, f"results/ev1_nodeloss_{dom}.txt"), "w").write(txt + "\n")
        print(txt + "\n")
        summary[dom] = res
    plot(summary, Ns)


def plot(summary, Ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches  # noqa: F401
    import matplotlib.pyplot as plt
    RET, REF = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated pair)
    ink, ink2 = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
    for j, (dom, name) in enumerate(DOMS):
        s = summary[dom]
        rows = s["re_formed"]
        for i, (k, lab) in enumerate((("recall", "Recall (%)"), ("fpr", "FPR (%)"))):
            a = ax[i, j]
            band = np.array([r[f"orders_{k}_p5_p95"] for r in rows]) * 100
            a.fill_between(Ns, band[:, 0], band[:, 1], color=REF, alpha=0.18, lw=0)
            a.plot(Ns, [100 * r[k] for r in rows], color=REF, lw=1.6, marker="o", ms=3,
                   label="Re-formed aggregate")
            a.axhline(100 * s["retention"][k], color=RET, lw=1.6, ls="--", label="Retention")
            a.grid(axis="y", color="#e4e3df", lw=0.5)
            a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            if i == 0:
                a.set_title(name, fontsize=8.5, color=ink)
            else:
                a.set_xlabel("Nodes present, $N$")
            if j == 0:
                a.set_ylabel(lab)
            if i == 0:
                a.set_ylim(0, 103)                  # one recall scale for the four domains
            a.set_xticks([14, 11, 8, 5, 3])
    ax[0, 0].set_xlim(14.5, 2.5)                    # shared x: nodes are lost left to right
    h, l = ax[0, 0].get_legend_handles_labels()
    band = matplotlib.patches.Patch(color=REF, alpha=0.18, lw=0,
                                    label="Re-formed, 5th-95th percentile over 1,000 orders")
    fig.legend(h + [band], l + [band.get_label()], loc="upper center", ncol=3, frameon=False,
               fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"results/ev1_nodeloss.{ext}"), dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=1000)
    a = ap.parse_args()
    os.chdir(ROOT)
    main(a.orders)
