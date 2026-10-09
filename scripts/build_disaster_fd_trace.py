#!/usr/bin/env python3
"""Compose the phase-structured ERENO trace for the Disaster-FD trust-level scenario.

The original ERENO test trace interleaves all seven attacks over its whole span (97-100% of
each attack class falls inside the benign time span), so it cannot express a scenario in
which attacks become known -- and appear -- phase by phase. This script recomposes the
trace by phase, from rows of the ERENO release, without changing any row: features, labels
and per-sample scores are those of the original release; only the order and the time axis
are synthetic.

Composition (conf/scenarios/disaster_fd_trace.yaml):
  * phases of the scenario timeline (minutes), each with the mode Disaster-FD selects;
  * samples per phase = the mean rate of the original test trace x the phase duration;
  * attack share per phase = the share of the original test trace;
  * attacks in the traffic of a phase: those already known (traffic: known) or all seven
    (traffic: all, so that not-yet-known attacks are present and go undetected);
  * attack rows: contiguous chunks of each class, taken in their original order, so that
    the message sequence inside each attack is preserved; the share of each present class
    follows its share in the original test trace;
  * benign rows: contiguous blocks of the benign pool in original order -- the benign rows
    of the test split, then the benign TRAIN rows beyond the specialists' training cap,
    which no specialist has seen;
  * interleaving: within a phase, rows are placed at uniform random times (seeded), keeping
    each class's internal order;
  * a benign-only calibration block, outside the 83 minutes, for the window thresholds.

Out: results/disaster_fd_trace/trace.npz      source (0 test, 1 train), row, label, phase, t
     results/disaster_fd_trace/calibration.npz  benign rows for threshold calibration
     results/disaster_fd_trace/manifest.json    inputs (SHA-256), configuration, counts

  python scripts/build_disaster_fd_trace.py conf/scenarios/disaster_fd_trace.yaml
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import file_ref  # noqa: E402


def arff_header(path):
    names, classes, n = [], None, 0
    with open(path) as f:
        for line in f:
            n += 1
            s = line.strip()
            if s.lower().startswith("@attribute"):
                if "{" in s:
                    classes = [c.strip() for c in s[s.index("{") + 1:s.rindex("}")].split(",")]
                names.append(s.split()[1])
            elif s.lower().startswith("@data"):
                return names, classes, n


def time_and_label(path):
    """First column (time, s) and class of every data row, without loading the features."""
    names, classes, skip = arff_header(path)
    df = pd.read_csv(path, skiprows=skip, header=None, usecols=[0, len(names) - 1],
                     names=["t", "cls"], dtype={"t": np.float64, "cls": str})
    lab = df["cls"].str.strip().map({c: i for i, c in enumerate(classes)}).values
    return df["t"].values, lab.astype(int), classes


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    rng = np.random.default_rng(cfg["seed"])
    B = cfg["data"]
    test_path, train_path = f"{B['test']}.csv", f"{B['train']}.csv"
    t_te, y_te, classes = time_and_label(test_path)
    _, y_tr, _ = time_and_label(train_path)
    nc = classes.index("normal")
    name = {i: c for i, c in enumerate(classes)}

    # benign pool: test benign in original time order, then the train benign beyond the cap
    # (reproduces the cap of the scenario scripts: first permutation of the seeded rng)
    order_te = np.argsort(t_te, kind="stable")
    ben_te = order_te[y_te[order_te] == nc]
    cap_rng = np.random.default_rng(cfg["seed"])
    nidx = np.where(y_tr == nc)[0]
    unused_tr = np.sort(cap_rng.permutation(nidx)[B["benign_cap"]:]) if len(nidx) > B["benign_cap"] else np.array([], int)
    pool_src = np.r_[np.zeros(len(ben_te), int), np.ones(len(unused_tr), int)]
    pool_row = np.r_[ben_te, unused_tr]
    pos_b = 0

    span_min = (t_te.max() - t_te.min()) / 60
    rate = len(t_te) / span_min if cfg["rate"] == "original" else float(cfg["rate"])
    p_atk = float(np.mean(y_te != nc)) if cfg["prevalence"] == "original" else float(cfg["prevalence"])
    share = {c: float(np.mean(y_te == c)) for c in range(len(classes)) if c != nc}
    atk_rows = {c: order_te[y_te[order_te] == c] for c in share}      # original order
    pos_a = {c: 0 for c in share}
    known_from = {int(k): v for k, v in cfg["known_from"].items()}

    def known_at(minute):
        return [classes.index(a) for m, lst in sorted(known_from.items()) if m <= minute for a in lst]

    present_in = []
    for ph in cfg["phases"]:
        present_in.append(known_at(ph["start"]) if cfg["traffic"] == "known" else sorted(share))
    # each class's rows are spread over the phases in which it appears, in proportion to the
    # phases' durations, so that no row is reused and no phase runs out: an attack present
    # from minute 0 is diluted over all phases, one distributed at minute 60 over the last
    quota = {}
    for c in share:
        dur = {k: ph["end"] - ph["start"] for k, ph in enumerate(cfg["phases"]) if c in present_in[k]}
        tot = sum(dur.values())
        for k, d in dur.items():
            quota[(c, k)] = int(len(atk_rows[c]) * d / tot)

    src, row, lab, phase, tt, report = [], [], [], [], [], []
    for k, ph in enumerate(cfg["phases"]):
        dur = ph["end"] - ph["start"]
        n = int(round(rate * dur))
        present = present_in[k]
        # attack rows: the class quota, capped so that the phase keeps at most the original
        # attack share (p_atk); the cap only binds where many classes are present
        n_cap = int(round(p_atk * n))
        q = {c: quota[(c, k)] for c in present}
        scale = min(1.0, n_cap / max(1, sum(q.values())))
        want = {c: int(q[c] * scale) for c in present}
        got = {}
        p_src, p_row, p_lab = [], [], []
        for c in present:
            take = min(want[c], len(atk_rows[c]) - pos_a[c])
            got[c] = take
            p_row.append(atk_rows[c][pos_a[c]:pos_a[c] + take]); pos_a[c] += take
            p_src.append(np.zeros(take, int)); p_lab.append(np.full(take, c))
        n_ben = n - sum(got.values())
        if pos_b + n_ben > len(pool_row):
            raise SystemExit(f"benign pool exhausted in {ph['name']}")
        p_row.append(pool_row[pos_b:pos_b + n_ben]); p_src.append(pool_src[pos_b:pos_b + n_ben])
        p_lab.append(np.full(n_ben, nc)); pos_b += n_ben
        p_row, p_src, p_lab = map(np.concatenate, (p_row, p_src, p_lab))
        # uniform random times in the phase; each class keeps its internal order
        times = np.sort(rng.uniform(ph["start"] * 60, ph["end"] * 60, len(p_row)))
        slot = rng.permutation(len(p_row))          # which time slot each row gets ...
        t_row = np.empty(len(p_row))
        for c in np.unique(p_lab):                  # ... reassigned so class order is kept
            m = np.where(p_lab == c)[0]
            t_row[m] = np.sort(times[slot[m]])
        src.append(p_src); row.append(p_row); lab.append(p_lab)
        phase.append(np.full(len(p_row), k)); tt.append(t_row)
        report.append({"phase": ph["name"], "minutes": [ph["start"], ph["end"]], "mode": ph["mode"],
                       "trust": ph.get("trust"), "known": [name[c] for c in known_at(ph["start"])],
                       "in_traffic": [name[c] for c in present], "samples": int(len(p_row)),
                       "benign": int(n_ben), "attack": {name[c]: int(v) for c, v in got.items()},
                       "attack_share": round(sum(got.values()) / len(p_row), 5),
                       "shortfall": {name[c]: int(want[c] - got[c]) for c in present if want[c] > got[c]}})
    src, row, lab, phase, tt = map(np.concatenate, (src, row, lab, phase, tt))
    o = np.argsort(tt, kind="stable")
    out = "results/disaster_fd_trace"
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(f"{out}/trace.npz", source=src[o], row=row[o], label=lab[o],
                        phase=phase[o], t=tt[o], classes=np.array(classes))
    n_cal = int(round(rate * cfg["calibration_min"]))
    cal_src, cal_row = pool_src[pos_b:pos_b + n_cal], pool_row[pos_b:pos_b + n_cal]
    np.savez_compressed(f"{out}/calibration.npz", source=cal_src, row=cal_row,
                        t=np.sort(rng.uniform(0, cfg["calibration_min"] * 60, len(cal_row))))
    manifest = {
        "scenario": cfg["name"], "config": cfg_path,
        "inputs": [file_ref(test_path), file_ref(train_path)],
        "original_test_trace": {"rows": int(len(t_te)), "span_min": round(span_min, 2),
                                "attack_share": round(p_atk, 5),
                                "class_share": {name[c]: round(v, 5) for c, v in share.items()}},
        "rate_per_min": round(rate, 1),
        "benign_pool": {"test_benign": int(len(ben_te)), "unused_train_benign": int(len(unused_tr)),
                        "used_in_trace": int(pos_b), "calibration": int(len(cal_row))},
        "attack_rows_used": {name[c]: int(pos_a[c]) for c in share},
        "attack_rows_available": {name[c]: int(len(atk_rows[c])) for c in share},
        "phases": report,
        "note": "rows unchanged (features, labels, per-sample scores); order and time axis synthetic",
    }
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "conf/scenarios/disaster_fd_trace.yaml")
