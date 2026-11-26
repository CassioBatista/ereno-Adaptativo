#!/usr/bin/env python3
"""EXPLORATORY (protocol change log 2026-11-26): a regional-partition variant of trace T2, built
to test the case that aggregator re-election cannot serve, a region cut off from the rest of the
network but still connected inside. Constructed, not recorded: it is derived from the Disaster-FD
run of scenario 2 by changing only the probes between ring neighbours inside the region.

T2 (recorded): at 45 min clients 4 and 5 are lost, at 60 min clients 2, 3, 6 and 7, each client
cut from every other one (firewall rules per client). Regional variant (constructed): the region
loses its links to the outside but not its internal links:
  - from 45 min, region {4, 5}: links 3-4 and 5-6 down (as recorded), link 4-5 up;
  - from 60 min, region {2, ..., 7}: links 1-2 and 7-8 down (as recorded), links 2-3, 3-4, 4-5,
    5-6 and 6-7 up;
  - the aggregator is lost by everyone at 30 min and returns at 75 min for clients 1 and 8-14
    only, as recorded.
A probe between two clients whose link is up in the variant is turned into an answered probe; the
trust level of each monitor of clients 2-7 is then recomputed from 45 min on as the sum of the
weights of the targets whose last probe was answered (50 aggregator, 15 per ring neighbour), the
model of Disaster-FD. The recomputation is checked first on the unchanged trace (share of rows
where it reproduces the recorded trust level).

  python scripts/build_regional_trace.py [--out ~/datasets/disaster_fd_scenario2_regional]
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

SRC = os.path.expanduser("~/datasets/disaster_fd_scenario2")
N = 14
W_AGG, W_PEER = 50, 15


def internal_up(a, b, t):
    """Whether the ring link between clients a and b (1-based) is up inside the region at t
    (seconds since run_t0) in the variant; None when the recorded probe is kept."""
    pair = tuple(sorted((a, b)))
    if t >= 3600 and pair in {(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)}:
        return True
    if t >= 2700 and pair == (4, 5):
        return True
    return None


def recompute_tl(df):
    """Trust level after each row: weights of the targets whose last probe (row order) was
    answered."""
    last, out = {}, []
    for d, to in zip(df["dev_id"].to_numpy(int), df["is_timeout"].astype(str).str.lower().eq("true")):
        last[d] = not to
        out.append(sum((W_AGG if k == 0 else W_PEER) for k, ok in last.items() if ok))
    return np.array(out)


def main(out_dir):
    t0 = float(json.load(open(os.path.join(SRC, "state", "t0.json")))["t0_actual"])
    files = sorted(glob.glob(os.path.join(SRC, "csv", "log_monitor*_Fleet_*.csv")))
    # 1. check: recomputed trust level against the recorded one, unchanged trace
    agree = total = 0
    for f in files:
        df = pd.read_csv(f).sort_values(["ts_arrival", "msg_id"], kind="stable")
        tl = recompute_tl(df)
        agree += int((tl == df["trust_level"].to_numpy()).sum()); total += len(df)
    print(f"check on the recorded trace: recomputed trust level equals the recorded one in "
          f"{agree:,} of {total:,} rows ({100 * agree / total:.2f} %)")
    # 2. build the variant
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} exists; remove it or choose another --out")
    shutil.copytree(SRC, out_dir, ignore=shutil.ignore_patterns("csv"))
    os.makedirs(os.path.join(out_dir, "csv"))
    changed = 0
    for f in files:
        mid = int(re.search(r"monitor(\d+)", os.path.basename(f)).group(1))
        df = pd.read_csv(f).sort_values(["ts_arrival", "msg_id"], kind="stable").reset_index(drop=True)
        t = df["ts_arrival"].to_numpy() / 1e6 - t0
        if 2 <= mid <= 7:
            flip = np.array([(d != 0) and bool(internal_up(mid, int(d), tt)) for d, tt in zip(df["dev_id"], t)])
            was = df["is_timeout"].astype(str).str.lower().eq("true").to_numpy()
            changed += int((flip & was).sum())
            df.loc[flip, "is_timeout"] = False
            tl = recompute_tl(df)
            late = t >= 2700
            df.loc[late, "trust_level"] = tl[late]
        df.to_csv(os.path.join(out_dir, "csv", os.path.basename(f)), index=False)
    open(os.path.join(out_dir, "REGIONAL_VARIANT.md"), "w").write(
        "Constructed regional-partition variant of scenario 2 (scripts/build_regional_trace.py, "
        "ereno-Adaptativo). Not a recorded run.\n")
    print(f"variant written to {out_dir}: {changed:,} probes turned into answered probes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/datasets/disaster_fd_scenario2_regional"))
    main(os.path.expanduser(ap.parse_args().out))
