"""Faithful replay, T2: per node at the last round, recall on the 60-min block and reachability,
for ReSIDS / FL / GL, in the four domains (which nodes make the difference)."""
import csv, os
D = os.path.expanduser("~/ereno-Adaptativo/results/ev1_tl/faithful")
for dom in ("ereno", "cicids", "morris4", "ciciot"):
    print(f"== {dom}")
    for pol in ("switching", "fixed_federated", "fixed_gossip"):
        rows = list(csv.DictReader(open(os.path.join(D, f"{dom}_T2_{pol}", "node_eval.csv"))))
        tmax = max(float(r["t_s"]) for r in rows)
        last = [r for r in rows if float(r["t_s"]) == tmax]
        print(f"  {pol:<16}" + " ".join(f"{int(r['node'])}{'R' if r['reachable'] == '1' else 'u'}:"
                                        f"{100 * float(r['recall_late_3600']):.0f}/{100 * float(r['recall_late_1800']):.0f}"
                                        for r in last))
