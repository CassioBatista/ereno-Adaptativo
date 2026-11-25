"""Step 1, seeds 42-46: mean and 95 % CI (t, 4 df) per domain and detector."""
import json, os
import numpy as np
os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
T4 = 2.776
for d in ("ereno", "cicids", "ciciot", "morris4"):
    rs = [json.load(open(f"results/ev1_step1_{d}_s{s}.json")) for s in range(42, 47)]
    print(f"\n== {rs[0]['domain']} (seeds 42-46)")
    print(f"{'detector':<20}" + "".join(f"{m:>18}" for m in ("F1", "recall", "precision", "FPR %")))
    for k in rs[0]["results"]:
        cells = []
        for m in ("f1", "recall", "precision", "fpr"):
            v = np.array([100 * r["results"][k][m] for r in rs])
            ci = T4 * v.std(ddof=1) / np.sqrt(len(v))
            cells.append(f"{v.mean():.3f}±{ci:.3f}" if m == "fpr" else f"{v.mean():.2f}±{ci:.2f}")
        print(f"{k:<20}" + "".join(f"{c:>18}" for c in cells))
    v = np.array([100 * r["local_mean"]["recall"] for r in rs])
    print(f"Local mean recall {v.mean():.2f}±{T4 * v.std(ddof=1) / np.sqrt(5):.2f}")
