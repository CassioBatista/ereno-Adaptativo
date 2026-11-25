"""Gossip band at 15 vs 30: ReSIDS (band 30, results/ev1_tl/faithful), ReSIDS with t_gl 15
(results/ev1_tl/faithful_band15), fixed GL gl_peers both (faithful), fixed GL gl_peers any
(faithful_gl_any) and fixed FL. Per domain, trace and block: exposure (share of node-rounds after
the arrival without protection) and nodes protected at the end; per run: node-rounds in gossip
(cost proxy). References as in ev1_tl_faithful_summary (main runs). Read-only."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo/scripts"))
import ev1_tl_faithful_summary as S
import numpy as np

R = os.path.expanduser("~/ereno-Adaptativo/results/ev1_tl")
def load(path):
    import csv
    f = lambda v: float(v) if v not in ("", "nan") else np.nan
    return [{"t": float(r["t_s"]), "node": int(r["node"]), "mode": r["node_mode"],
             "present": f(r["recall_present"]), **{a: f(r[f"recall_late_{a}"]) for a in S.BLOCKS}}
            for r in csv.DictReader(open(os.path.join(path, "node_eval.csv")))]

CFG = [("ReSIDS-30", "faithful", "switching"), ("ReSIDS-15", "faithful_band15", "switching_gl15"),
       ("GL-both", "faithful", "fixed_gossip"), ("GL-any", "faithful_gl_any", "fixed_gossip"),
       ("FL", "faithful", "fixed_federated")]
tot = {c[0]: [] for c in CFG}
for dom, dn in S.DOMS:
    main = {(tr, p): S.node_eval(dom, tr, p) for tr, _ in S.TRACES for p, _ in S.POLS if S.have(dom, tr, p)}
    ref = {a: max(np.nanmax([r[a] for r in rows]) for rows in main.values()) for a in S.BLOCKS}
    refp = S.ref_present(main)
    print(f"==== {dn}")
    for tr, _ in S.TRACES:
        print(f"--- {tr}   exposure block 30 / block 60 (%), protected at end b30 b60, mean share protected (present), gossip node-rounds")
        for lab, sub, pol in CFG:
            rows = load(os.path.join(R, sub, f"{dom}_{tr}_{pol}"))
            ts = sorted({r["t"] for r in rows}); last = [r for r in rows if r["t"] == ts[-1]]
            e = {a: 100 * np.mean([not (r[a] >= S.REL * ref[a]) for r in rows if r["t"] >= a]) for a in S.BLOCKS}
            pe = {a: sum(r[a] >= S.REL * ref[a] for r in last) for a in S.BLOCKS}
            sh = 100 * np.mean([v for _, v in S.protected_share(rows, ref, refp)])
            g = sum(r["mode"] == "gossip" for r in rows)
            tot[lab].append((e[1800], e[3600]))
            print(f"  {lab:<10} {e[1800]:6.1f} {e[3600]:6.1f}   {pe[1800]:2d} {pe[3600]:2d}   {sh:5.1f}   {g:4d}")
print("\nmean exposure over the 16 cells (block 30, block 60):")
for lab in tot:
    a = np.array(tot[lab]); print(f"  {lab:<10} {a[:,0].mean():5.1f} {a[:,1].mean():5.1f}")
