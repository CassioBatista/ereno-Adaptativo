"""T1: do ReSIDS band 15 and band 30 hold the same specialists at 44.5 min (end of the period
in which only band 15 gossips)? And the GLow heads after 45 min. Read-only."""
import json, os
R = os.path.expanduser("~/ereno-Adaptativo/results/ev1_tl")
for dom in ("ereno", "cicids"):
    a = [json.loads(l) for l in open(f"{R}/faithful/{dom}_T1_switching/rounds.jsonl")]
    b = [json.loads(l) for l in open(f"{R}/faithful_band15/{dom}_T1_switching_gl15/rounds.jsonl")]
    ra = {r["t_s"]: r for r in a}; rb = {r["t_s"]: r for r in b}
    for t in (2670.0, 2730.0, 3210.0):
        same = [sorted(x) == sorted(y) for x, y in zip(ra[t]["holdings"], rb[t]["holdings"])]
        print(f"{dom} t={t/60:.1f} min: nodes with the same holdings {sum(same)}/14; "
              f"sizes band30 {[len(x) for x in ra[t]['holdings']]} band15 {[len(x) for x in rb[t]['holdings']]}")
