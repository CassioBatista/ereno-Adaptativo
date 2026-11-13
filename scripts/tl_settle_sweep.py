#!/usr/bin/env python3
"""Settle-time sweep of the API 1.9.0 agent policy over a Disaster-FD run (docs/TRUST_LEVEL.md
§3.2, Table tab:settle of the paper).

For each settle time S, the policy of scripts/tl_policy_replay.py (unchanged, imported) is
replayed on every client and the script reports
  * switches per client (min-max);
  * transients: a mode entered and left again within 60 s (phases last 900 s), per client,
    in total, and their durations;
  * latency after each phase boundary (last switch - boundary), median / max, with the
    boundaries taken from the estimated run_t0 (first jump to TL 80 minus 900 s), as in
    tl_policy_replay.py.

  python scripts/tl_settle_sweep.py ~/datasets/disaster_fd_run1 [--settle 0 3 5 10] [--stale 15]
Out: results/tl_settle_sweep.txt
"""
import argparse
import glob
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tl_policy_replay as R  # noqa: E402

T_END = 4980.0
TRANSIENT = 60.0                                    # s; phases last 900 s


def main(d, settles, F, gl, fl):
    files = sorted(glob.glob(os.path.join(d, "runs", "*", "*", "csv", "log_monitor*_Fleet_*.csv")))
    data = {int(re.search(r"monitor(\d+)", os.path.basename(f)).group(1)):
            pd.read_csv(f).sort_values(["ts_arrival", "msg_id"], kind="stable") for f in files}
    t0 = min(df.loc[df["trust_level"] >= 80, "ts_arrival"].min() for df in data.values()) - 900e6
    print(f"run: {d}\nmonitors: {len(data)}  run_t0 (estimated) {R.iso(t0)}")
    print(f"policy: bands gl={gl} fl={fl}; stale after F={F} s; transient = mode left within "
          f"{TRANSIENT:g} s\n")
    for S in settles:
        nsw, ntr, durs, lat = [], [], [], []
        for mid, df in sorted(data.items()):
            t = (df["ts_arrival"].values - t0) / 1e6
            keep = (t >= 0) & (t < T_END)
            sw = R.policy(t[keep], df["trust_level"].values[keep], S, F, gl, fl, T_END)
            tc = [s[0] for s in sw] + [T_END]
            d_ = [tc[k + 1] - tc[k] for k in range(len(sw)) if tc[k + 1] - tc[k] < TRANSIENT]
            nsw.append(len(sw)); ntr.append(len(d_)); durs += d_
            tca = np.array(tc[:-1])
            for B in R.BOUNDS:
                after = tca[(tca >= B - 10) & (tca < B + 60)]
                lat.append(after.max() - B if len(after) else np.nan)
        lat = np.array(lat)
        dur = f"{min(durs):.1f}-{max(durs):.1f} s" if durs else "-"
        print(f"S={S:>4g} s: switches/client {min(nsw)}-{max(nsw)}; transients/client "
              f"{min(ntr)}-{max(ntr)}, total {sum(ntr)}, durations {dur}; latency median "
              f"{np.nanmedian(lat):.1f} s, max {np.nanmax(lat):.1f} s, "
              f"missing {int(np.isnan(lat).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="~/datasets/disaster_fd_run1")
    ap.add_argument("--settle", type=float, nargs="+", default=[0.0, 3.0, 5.0, 10.0])
    ap.add_argument("--stale", type=float, default=15.0)
    ap.add_argument("--gl", type=float, default=30.0)
    ap.add_argument("--fl", type=float, default=50.0)
    a = ap.parse_args()
    run = os.path.expanduser(a.run)
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(run, a.settle, a.stale, a.gl, a.fl)
    print(buf.getvalue(), end="")
    os.makedirs("results", exist_ok=True)
    with open("results/tl_settle_sweep.txt", "w") as f:
        f.write(buf.getvalue())
