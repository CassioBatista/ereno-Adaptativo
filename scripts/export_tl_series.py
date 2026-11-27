#!/usr/bin/env python3
"""Minimal export of the two Disaster-FD runs for publication, pending the agreement of the
Disaster-FD team: per monitor, only the fields the replay reads (ts_arrival, msg_id, dev_id,
is_timeout, trust_level), plus the run start (state/t0.json) where the run records it. Location,
voting, timing-model and error fields are dropped. Written OUTSIDE the repository, then checked:
the replay from the export must give, in every round of T1 and T2, the same modes and participants
as the replay from the original traces, for ReSIDS, fixed FL and fixed GL (S = 5 and 10 s).

  python scripts/export_tl_series.py [--out ~/datasets/disaster_fd_tl_series]
"""
import argparse
import glob
import io
import os
import shutil
import sys
from contextlib import redirect_stdout

import pandas as pd

ROOT = os.path.expanduser("~/ereno-Adaptativo")
sys.path.insert(0, ROOT)
from fd.arch_manager import TrustLevelArchManager as M  # noqa: E402

RUNS = {"T1": ("~/datasets/disaster_fd_run1", 4980), "T2": ("~/datasets/disaster_fd_scenario2", 5880)}
COLS = ["ts_arrival", "msg_id", "dev_id", "is_timeout", "trust_level"]


def files(run):
    f = glob.glob(os.path.join(run, "csv", "log_monitor*_Fleet_*.csv"))
    return sorted(f or glob.glob(os.path.join(run, "runs", "*", "*", "csv", "log_monitor*_Fleet_*.csv")))


def export(src, dst):
    os.makedirs(os.path.join(dst, "csv"), exist_ok=True)
    rows = 0
    for f in files(src):
        df = pd.read_csv(f, usecols=COLS)[COLS]
        mid = os.path.basename(f).split("_")[1]                     # monitorNN
        df.to_csv(os.path.join(dst, "csv", f"log_{mid}_Fleet_export.csv"), index=False)
        rows += len(df)
    t0 = os.path.join(src, "state", "t0.json")
    if os.path.exists(t0):
        os.makedirs(os.path.join(dst, "state"), exist_ok=True)
        shutil.copy(t0, os.path.join(dst, "state", "t0.json"))
    return rows


def schedule(run, tend, policy, settle):
    with redirect_stdout(io.StringIO()):
        m = M(run=run, t_end=tend, delta_s=60, links="scenario", round_offset_s=30, settle=settle,
              gl_peers="both", policy=policy)
        return [(m.get_mode(k), tuple(m.get_active_clients(k)), tuple(m.node_modes(k)))
                for k in range(1, m.n_rounds + 1)]


def main(out):
    out = os.path.expanduser(out)
    for name, (src, tend) in RUNS.items():
        src = os.path.expanduser(src)
        dst = os.path.join(out, name)
        rows = export(src, dst)
        same = all(schedule(src, tend, p, s) == schedule(dst, tend, p, s)
                   for p in ("switching", "fixed_federated", "fixed_gossip") for s in (5.0, 10.0))
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dst) for f in fs)
        print(f"{name}: {rows} probe records, {size / 1e6:.1f} MB -> {dst}; "
              f"replay identical to the original: {same}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/datasets/disaster_fd_tl_series")
    main(ap.parse_args().out)
