"""Seed a GRASP evaluation checkpoint (python/grasp/checkpoint.py) from the log of
interrupted runs, so that a resumed run replays them instead of recomputing.

The log may hold several runs of the same configuration (each restart appends a new
"[global-pen] dataset=<dataset>; ..." section). All sections must be prefixes of one
another (same ordered feature lists, same F1, in the same order) -- the evidence that
the search is deterministic; the longest one is written to the checkpoint.

Each "EV;n;f1;[features]" line is preceded by one "XGBoost;acc;prec;rec;f1;VP;VN;FP;FN;t"
line per fold. The fold counts are averaged as Result.average does, the Result is
rebuilt, and its F1 must equal the logged F1 exactly; the record then carries the
full counts. The duration of each evaluation is not logged: it is estimated by spreading
the elapsed time of the "Fim ITERAÇÂO (n / X min)" markers evenly over the evaluations
between them (after the last marker, the mean of the last segment; with no marker at all,
0). It only feeds the virtual clock of the max_time check of GRASP.

Records seeded here have no RNG hash ("rng": null): the replay checks them on the ordered
feature lists; evaluations computed after the resume carry the hash.

For legacy logs only (runs without --checkpoint). Once a run has a checkpoint, the JSONL
is the record: a resumed run does not print the fold lines of replayed evaluations, so its
log cannot be used here.

Usage:
  python scripts/grasp_ckpt_from_log.py --log results/grasp_ev1_cicids.log \
      --dataset all_in_one_ev1_cicids_train \
      --out features/checkpoints/global_penalizado_ev1_cicids_L0.05.jsonl
"""
import argparse
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import python.config as config
from python.result import Result

FOLDS = 5


def parse_sections(lines, dataset):
    head = f"[global-pen] dataset={dataset};"
    ends = ("[global-pen]", "[ev1_", "[grasp-atk]", "[GRASP] convergiu")
    sections, cur = [], None
    for no, line in enumerate(lines, 1):
        if line.startswith(head):
            cur = {"start": no, "header": {}, "evs": [], "markers": [], "folds": []}
            sections.append(cur)
            m = re.search(r"FEATURE_PENALTY \(lambda\) = ([0-9.eE+-]+)", line)
            cur["header"]["feature_penalty"] = float(m.group(1))
            continue
        if cur is None:
            continue
        if line.startswith(ends):
            cur = None
            continue
        if line.startswith("[GRASP] subamostra estratificada:"):
            cur["header"]["sample"] = int(line.split(":")[1].split("de")[0].strip().replace(",", ""))
        elif line.startswith("[GRASP] classes raras"):
            body = line.split(":", 1)[1]
            cur["header"]["dropped_classes"] = ast.literal_eval(body[body.index("["):body.index("]") + 1])
        elif line.startswith("[GRASP] ") and " clf=" in line:
            m = re.match(r"\[GRASP\] (\S+) clf=(\d+) dataset=(\S+) \| parada: (\d+) .*teto (\d+)", line)
            cur["header"].update(grasp_method=m.group(1), classifier_idx=int(m.group(2)),
                                 dataset=m.group(3), no_improvement=int(m.group(4)),
                                 max_iterations=int(m.group(5)))
        elif line.startswith("XGBoost;"):
            cur["folds"].append(line)
        elif line.startswith("EV;"):
            try:
                _, n, f1, feats = line.split(";", 3)
                rec = {"n": int(n), "f1": float(f1), "features": ast.literal_eval(feats),
                       "folds": cur["folds"], "line": no}
            except (ValueError, SyntaxError):
                print(f"  line {no}: unparsable EV line (torn write?), section ends here")
                cur = None
                continue
            cur["evs"].append(rec)
            cur["folds"] = []
        elif line.startswith("######### Fim ITERA"):
            m = re.search(r"\((\d+) / ([0-9.]+)min\)", line)
            cur["markers"].append((len(cur["evs"]), float(m.group(2))))
    return sections


def rebuild(rec):
    folds = rec["folds"]
    if len(folds) != FOLDS:
        raise SystemExit(f"line {rec['line']}: {len(folds)} fold lines before EV {rec['n']}, "
                         f"expected {FOLDS}")
    rs = []
    config.FEATURE_SELECTION = rec["features"]
    for fl in folds:
        p = fl.split(";")
        vp, vn, fp, fn, t = int(p[5]), int(p[6]), int(p[7]), int(p[8]), float(p[9])
        rs.append(Result("XGBoost", vp, fn, vn, fp, t))
    avg = Result.average(rs)
    if avg.f1score != rec["f1"]:
        raise SystemExit(f"line {rec['line']}: rebuilt F1 {avg.f1score!r} != logged {rec['f1']!r}")
    return avg


def durations(evs, markers):
    """Seconds per evaluation, spread evenly between iteration markers. Without any
    marker the durations are unknown and set to 0 (the last field of a fold line is the
    mean prediction time per instance in ns, not the evaluation time)."""
    n = len(evs)
    if not markers:
        return [0.0] * n, "unknown (no iteration marker in the log): set to 0"
    d, prev_k, prev_t = [], 0, 0.0
    for k, t in markers:
        seg = (t - prev_t) * 60 / max(k - prev_k, 1)
        d += [seg] * (k - prev_k)
        prev_k, prev_t = k, t
    d += [seg] * (n - prev_k)
    return d, "estimated from iteration markers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; move it away first (this tool never overwrites)")

    lines = open(a.log, encoding="utf-8", errors="replace").read().split("\n")
    secs = [s for s in parse_sections(lines, a.dataset) if s["evs"]]
    if not secs:
        raise SystemExit("no section with evaluations")
    for s in secs:
        print(f"  section at line {s['start']}: {len(s['evs'])} evaluations, "
              f"{len(s['markers'])} iteration markers")
    best = max(secs, key=lambda s: len(s["evs"]))
    for s in secs:                                   # every section a prefix of the longest
        if s["header"] != best["header"]:
            raise SystemExit(f"section {s['start']}: header {s['header']} != {best['header']}")
        for x, y in zip(s["evs"], best["evs"]):
            if (x["n"], x["f1"], x["features"]) != (y["n"], y["f1"], y["features"]):
                raise SystemExit(f"section {s['start']} diverges at EV {x['n']} (line {x['line']})")
    evs = best["evs"]
    for i, e in enumerate(evs, 1):
        if e["n"] != i:
            raise SystemExit(f"line {e['line']}: EV {e['n']} out of sequence (expected {i})")

    dur, how = durations(evs, best["markers"])
    header = {"kind": "header", **best["header"], "seed": config.GRASP_SEED, "folds": FOLDS,
              "seeded_from": {"log": a.log, "section_line": best["start"],
                              "sections_consistent": len(secs), "durations": how}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(header) + "\n")
        for e, d in zip(evs, dur):
            r = rebuild(e)
            fh.write(json.dumps({"kind": "eval", "k": e["n"], "features": e["features"],
                                 "f1": e["f1"], "VP": r.VP, "FN": r.FN, "VN": r.VN, "FP": r.FP,
                                 "avg_time": r.avg_time, "dur_s": round(d, 3), "rng": None,
                                 "ts": None, "src": f"log:{e['line']}"}) + "\n")
    os.replace(tmp, a.out)
    print(f"{a.out}: header {best['header']}")
    print(f"  {len(evs)} evaluations from the section at line {best['start']} "
          f"(all {len(secs)} sections consistent; F1 rebuilt exactly from the fold counts)")
    print(f"  elapsed at the last one: {sum(dur) / 60:.1f} min ({how}); markers {best['markers']}")


if __name__ == "__main__":
    main()
