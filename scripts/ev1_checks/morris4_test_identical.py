"""Morris-4 test split: benign-identical share per attack type (amendment 2026-11-17, section 7),
and the share of benign test rows whose vector also occurs among attack rows; per-type recall
of step 1 (seed 42) next to it, grouped by category. Vectors = the 16 selected features, NaN
compared as a value."""
import json, os, sys
import numpy as np
os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
sys.path.insert(0, "scripts")
import ev1_step1 as E
CAT = {1: "NMRI", 2: "CMRI", 3: "MSCI", 4: "MPCI", 5: "MFCI", 6: "DoS", 7: "Recon"}
cat_of = {}
for l in open("results/morris4_q2d_training_diagnostic.txt"):
    p = l.split()
    if len(p) == 6 and p[0].isdigit():
        cat_of[f"specific_{p[0]}"] = CAT[int(p[1])]
dom = E.DOMAINS["morris4"]
feats, _ = E.selection(dom)
Xt, yt, classes, nc = E.load(dom["test"], feats)
Xr, yr, _, _ = E.load(dom["train"], feats)
key = lambda X: [tuple(np.where(np.isnan(r), -9.99e9, r)) for r in X]
kt, kr = key(Xt), key(Xr)
ben_t = {k for k, y in zip(kt, yt) if y == nc}
att_all = {k for k, y in zip(kt, yt) if y != nc} | {k for k, y in zip(kr, yr) if y != nc}
bmask = yt == nc
bi = np.array([k in att_all for k, b in zip(kt, bmask) if b])
print(f"benign test rows: {bmask.sum():,}; identical to an attack row (train or test): {100 * bi.mean():.1f} %")
s42 = json.load(open("results/ev1_step1_morris4_s42.json"))["results"]
rows = []
for c in range(len(classes)):
    if c == nc or not (yt == c).any():
        continue
    m = yt == c
    share = np.mean([k in ben_t for k, mm in zip(kt, m) if mm])
    n = classes[c]
    rows.append((cat_of.get(n, "?"), n, int(m.sum()), 100 * share,
                 100 * s42["attack-based OR"]["per_type"][n]["recall"],
                 100 * s42["IID OR"]["per_type"][n]["recall"],
                 100 * s42["Centralized"]["per_type"][n]["recall"]))
print(f"{'cat':<6}{'type':<13}{'n':>6}{'ident%':>8}{'AB':>8}{'IID':>8}{'Cen':>8}")
for r in sorted(rows):
    print(f"{r[0]:<6}{r[1]:<13}{r[2]:>6}{r[3]:>8.1f}{r[4]:>8.1f}{r[5]:>8.1f}{r[6]:>8.1f}")
print("\nby category (row-weighted):")
for cat in sorted({r[0] for r in rows}):
    rr = [r for r in rows if r[0] == cat]
    n = sum(r[2] for r in rr)
    w = lambda i: sum(r[i] * r[2] for r in rr) / n
    print(f"{cat:<6} types {len(rr):2d} rows {n:6,} ident {w(3):5.1f}  AB {w(4):5.1f}  IID {w(5):5.1f}  Cen {w(6):5.1f}")
