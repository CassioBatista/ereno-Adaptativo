#!/usr/bin/env python3
"""EXPLORATORY (protocol change log 2026-11-26): noisy variants of the Disaster-FD traces T1 and T2,
to test the switching under imperfect trust levels. Constructed, not recorded.

Two kinds of noise, drawn independently per row with a seeded generator:
  - probe loss (--loss p): an answered probe becomes a timeout with probability p; the trust level
    of every monitor is then recomputed as the sum of the weights of the targets whose last probe
    was answered (50 aggregator, 15 per ring neighbour), the model of Disaster-FD. It changes the
    trust level and the link states, so it reaches every policy.
  - misreport (--misreport q): the trust level published after a probe is replaced, with
    probability q, by another value of {0, 15, 30, 50, 65, 80}, drawn uniformly; the probes are
    unchanged, so it reaches only the policies that read the trust level (ReSIDS), not the links.
The recomputation is first checked on the unchanged trace (share of rows where it reproduces the
recorded trust level). The time base (run_t0) of the original trace is written to state/t0.json,
so that injected values cannot move it.

  python scripts/build_noisy_trace.py --trace T1|T2 [--loss 0.05] [--misreport 0.0] [--seed 1]
Out: ~/datasets/disaster_fd_<T1|T2>_noise_l<p>_m<q>_s<seed>/csv/log_monitorNN_Fleet_*.csv
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/ereno-Adaptativo"))
from fd.trust_level import _csv_files, load_trace  # noqa: E402

TRACES = {"T1": "~/datasets/disaster_fd_run1", "T2": "~/datasets/disaster_fd_scenario2"}
LEVELS = np.array([0, 15, 30, 50, 65, 80])


def timeouts(df):
    return df["is_timeout"].astype(str).str.lower().eq("true").to_numpy()


def recompute_tl(dev, to):
    last, out = {}, []
    for d, t in zip(dev, to):
        last[d] = not t
        out.append(sum((50 if k == 0 else 15) for k, ok in last.items() if ok))
    return np.array(out)


def main(trace, loss, misreport, seed):
    src = os.path.expanduser(TRACES[trace])
    _, (t0, t0_src) = load_trace(src)
    files = _csv_files(src)
    agree = total = 0
    for f in files:
        df = pd.read_csv(f).sort_values(["ts_arrival", "msg_id"], kind="stable")
        tl = recompute_tl(df["dev_id"].to_numpy(int), timeouts(df))
        agree += int((tl == df["trust_level"].to_numpy()).sum()); total += len(df)
    print(f"{trace}: recomputed trust level equals the recorded one in {agree:,} of {total:,} rows "
          f"({100 * agree / total:.2f} %)")
    tag = f"{trace}_noise_l{loss:g}_m{misreport:g}_s{seed}"
    out = os.path.expanduser(f"~/datasets/disaster_fd_{tag}")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    os.makedirs(os.path.join(out, "csv")); os.makedirs(os.path.join(out, "state"))
    json.dump({"t0_actual": t0, "source": f"original trace {TRACES[trace]} ({t0_src})"},
              open(os.path.join(out, "state", "t0.json"), "w"))
    rng = np.random.default_rng(seed)
    lost = mis = 0
    for f in files:
        df = pd.read_csv(f).sort_values(["ts_arrival", "msg_id"], kind="stable").reset_index(drop=True)
        to = timeouts(df)
        if loss > 0:
            drop = (~to) & (rng.random(len(df)) < loss)
            lost += int(drop.sum())
            to = to | drop
            df["is_timeout"] = to
            df["trust_level"] = recompute_tl(df["dev_id"].to_numpy(int), to)
        if misreport > 0:
            m = rng.random(len(df)) < misreport
            mis += int(m.sum())
            cur = df["trust_level"].to_numpy()
            new = cur.copy()
            for i in np.flatnonzero(m):
                new[i] = rng.choice(LEVELS[LEVELS != cur[i]])
            df["trust_level"] = new
        mid = int(re.search(r"monitor(\d+)", os.path.basename(f)).group(1))
        df.to_csv(os.path.join(out, "csv", f"log_monitor{mid:02d}_Fleet_noise.csv"), index=False)
    open(os.path.join(out, "NOISE.md"), "w").write(
        f"Constructed noisy variant of {TRACES[trace]} (scripts/build_noisy_trace.py, ereno-Adaptativo): "
        f"probe loss {loss}, misreport {misreport}, seed {seed}. Not a recorded run.\n")
    print(f"{tag}: {lost:,} probes lost, {mis:,} trust levels misreported -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", choices=list(TRACES), required=True)
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--misreport", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    main(a.trace, a.loss, a.misreport, a.seed)
