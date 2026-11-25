"""ERENO T1, ReSIDS: per node and round, recall on the block of 30 min and boosters held,
from 30 to 62 min. Read-only on results/ev1_tl/faithful/ereno_T1_switching."""
import csv, json, os
d = os.path.expanduser("~/ereno-Adaptativo/results/ev1_tl/faithful/ereno_T1_switching")
rows = list(csv.DictReader(open(os.path.join(d, "node_eval.csv"))))
modes = {}
for line in open(os.path.join(d, "rounds.jsonl")):
    r = json.loads(line)
    modes[r["round"]] = r["mode"]
print("min  mode     " + " ".join(f"n{n:<4}" for n in range(1, 15)))
by = {}
for r in rows:
    by.setdefault(int(r["round"]), {})[int(r["node"])] = r
for rd in sorted(by):
    t = float(by[rd][1]["t_s"])
    if not 1800 <= t <= 3780:
        continue
    cells = []
    for n in range(1, 15):
        x = by[rd][n]
        rec = float(x["recall_late_1800"])
        cells.append(f"{rec*100:3.0f}/{x['n_boosters']:<2}"[:6].ljust(6))
    print(f"{t/60:5.1f} {modes[rd]:<8} " + "".join(cells))
