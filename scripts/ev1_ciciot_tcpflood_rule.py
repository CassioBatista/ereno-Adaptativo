"""What rule did the CICIoT2023 DDoS_TCP_Flood specialist learn? (step 1, seed 42, post hoc)
1. Its training rows: attack and benign counts, Header_Length quantiles of each.
2. Split thresholds on Header_Length used by its trees.
3. Test: flagged share and Header_Length quantiles per class, for the classes it flags and
   those it does not."""
import json, os, re, sys
import numpy as np
os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
sys.path.insert(0, "scripts")
import ev1_step1 as E
import xgboost as xgb

dom = E.DOMAINS["ciciot"]
feats, _ = E.selection(dom)
k = feats.index(2)                                   # Header_Length = F2
X, y, classes, nc = E.load(dom["train"], feats)
units, slices, kept = E.attack_partition(y, nc, classes, dom["cap"], 42)
name, codes, node, idx = [u for u in units if u[0] == "DDoS_TCP_Flood"][0]
q = lambda v: ", ".join(f"{np.percentile(v, p):.4g}" for p in (1, 5, 50, 95, 99))
print(f"training rows of the specialist (node {node + 1}): {len(idx):,} DDoS_TCP_Flood, "
      f"{len(slices[node]):,} benign")
print(f"  Header_Length quantiles 1/5/50/95/99 %: attack {q(X[idx, k])}")
print(f"                                           benign {q(X[slices[node], k])}")

spec = json.load(open("results/ev1_step1_ciciot_s42.json"))["specialists"]
j = [s["j"] for s in spec if s["name"] == "DDoS_TCP_Flood"][0]
b = xgb.Booster()
b.load_model(f"results/.cache/ev1_step1_models_ciciot_s42/spec{j:02d}.json")
dump = b.get_dump()
thr = sorted({float(m) for t in dump for m in re.findall(rf"\[f{k}<([-0-9.e+]+)\]", t)})
nsplit = sum(len(re.findall(r"\[f\d+<", t)) for t in dump)
nhl = sum(len(re.findall(rf"\[f{k}<", t)) for t in dump)
print(f"\ntrees {len(dump)}; splits {nsplit}, on Header_Length {nhl}; thresholds: "
      + ", ".join(f"{t:.4g}" for t in thr))
print("first tree:\n" + dump[0])

Xt, yt, _, _ = E.load(dom["test"], feats)
P = b.predict(xgb.DMatrix(Xt)) >= 0.5
print(f"\ntest, per class: flagged %, Header_Length quantiles 5/50/95 %")
for c in sorted(set(yt.tolist())):
    m = yt == c
    v = Xt[m, k]
    print(f"  {classes[c]:<26}{100 * P[m].mean():6.1f}   " +
          ", ".join(f"{np.percentile(v, p):.4g}" for p in (5, 50, 95)))
