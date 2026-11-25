"""Morris-4, step 1 seed 42: where the attack-based false positives come from.
(a) benign test vectors never seen among benign training vectors (time split shift);
(b) concentration of the flagged benign rows on few distinct vectors;
(c) FPR on benign test rows whose vector was / was not seen in benign training."""
import os, sys
import numpy as np
os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
sys.path.insert(0, "scripts")
import ev1_step1 as E
from collections import Counter
dom = E.DOMAINS["morris4"]
feats, _ = E.selection(dom)
Xt, yt, classes, nc = E.load(dom["test"], feats)
Xr, yr, _, _ = E.load(dom["train"], feats)
key = lambda X: [tuple(np.where(np.isnan(r), -9.99e9, r)) for r in X]
kt, kr = key(Xt), key(Xr)
ben_r = {k for k, y in zip(kr, yr) if y == nc}
z = np.load("results/.cache/ev1_step1_morris4_s42.npz")
assert (z["y"] == yt).all()
Ps, Pi, Pc = z["Ps"], z["Pi"], z["Pc"]
b = yt == nc
seen = np.array([k in ben_r for k in kt])
print(f"benign test rows {b.sum():,}; vector seen among benign training rows: {100 * seen[b].mean():.1f} %")
print(f"distinct benign test vectors {len({k for k, bb in zip(kt, b) if bb}):,}")
for name, pred in (("attack-based OR", (Ps >= 0.5).any(1)), ("IID OR", (Pi >= 0.5).any(1)), ("Centralized", Pc >= 0.5)):
    fs, fu = pred[b & seen].mean(), pred[b & ~seen].mean() if (b & ~seen).any() else float("nan")
    print(f"{name:<16} FPR seen {100 * fs:5.1f} %  unseen {100 * fu:5.1f} %  overall {100 * pred[b].mean():5.1f} %")
fp = (Ps >= 0.5).any(1) & b
c = Counter(k for k, f in zip(kt, fp) if f)
tot = sum(c.values())
top = c.most_common(10)
print(f"\nattack-based FP rows {tot:,} on {len(c):,} distinct vectors; top 10 vectors hold {100 * sum(v for _, v in top) / tot:.1f} %")
nan_fp = np.mean([np.isnan(Xt[i, 3]) for i in np.where(fp)[0]])
nan_b = np.isnan(Xt[b, 3]).mean()
print(f"share of rows with feature 4 missing (response-type rows): benign {100 * nan_b:.1f} %, attack-based FP {100 * nan_fp:.1f} %")
# which specialists cause them
hit = (Ps >= 0.5) & b[:, None]
only = hit.sum(1) == 1
print("specialists with most benign hits (count, of which sole firer):")
for j in np.argsort(-hit.sum(0))[:8]:
    print(f"  spec {j:02d}: {hit[:, j].sum():6,}  sole {int((hit[:, j] & only).sum()):6,}")
