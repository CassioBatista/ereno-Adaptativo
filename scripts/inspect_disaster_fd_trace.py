#!/usr/bin/env python3
"""Validate the Disaster-FD telemetry of run 20261008-154706 (14-client pilot, Algorithm 1)
against the checklist of the team's guide.md (section 10), and derive the per-client mode
timeline the ReSIDS agent would follow under the TL bands of API 1.9.0:
    TL < 30 -> local (LL),  30 <= TL < 50 -> gossip (GL),  TL >= 50 -> federated (FL).

run_t0 is not in the run directory; as the guide prescribes when the printed "Scenario t0"
line is missing, it is ESTIMATED from the 15-min all-clear: the first jump to TL = 80,
minus 900 s (the earliest such jump over the 14 monitors).

  python scripts/inspect_disaster_fd_trace.py ~/datasets/disaster_fd_run1
Out: results/disaster_fd_run1_check.txt, results/disaster_fd_run1_modes.csv
"""
import glob
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

PHASES = [(0, 900, "isolated", {"all": [0]}), (900, 1800, "healthy", {"all": [80]}),
          (1800, 2700, "outage", {"M01": [0], "M14": [0], "other": [15]}),
          (2700, 3600, "partial recovery", {"all": [30]}), (3600, 4980, "full recovery", {"all": [80]})]
TOL = 10            # s, the guide's tolerance around phase boundaries


def mode(tl):
    return np.where(tl >= 50, "federated", np.where(tl >= 30, "gossip", "local"))


def main(d):
    files = sorted(glob.glob(os.path.join(d, "runs", "*", "*", "csv", "log_monitor*_Fleet_*.csv")))
    print(f"monitor files: {len(files)}")
    data = {}
    for f in files:
        mid = int(re.search(r"monitor(\d+)", os.path.basename(f)).group(1))
        df = pd.read_csv(f).sort_values(["ts_arrival", "msg_id"], kind="stable")
        data[mid] = df
    # run_t0 estimate: earliest first jump to TL == 80, minus 900 s
    jumps = [df.loc[df["trust_level"] >= 80, "ts_arrival"].min() for df in data.values()]
    t0 = min(j for j in jumps if pd.notna(j)) - 900e6
    print(f"run_t0 (ESTIMATED from the 15-min all-clear): {pd.to_datetime(t0, unit='us')} UTC "
          f"(spread of the 14 jumps: {(max(jumps) - min(jumps)) / 1e6:.1f} s)")
    print("\n=== checklist (guide.md section 10) ===")
    for mid, df in data.items():
        tg = sorted(df["dev_id"].unique())
        pred, succ = ((mid - 2) % 14) + 1, (mid % 14) + 1
        ok = tg == sorted([0, pred, succ])
        trusted = df.groupby("dev_id")["trusted"]
        rows = len(df)
        print(f"  M{mid:02d}: rows {rows:,}  targets {tg} {'OK' if ok else 'MISMATCH'}  "
              f"cadence {df.groupby('dev_id')['ts_arrival'].apply(lambda s: np.median(np.diff(s)) / 1e6).round(1).to_dict()} s")
    allc = pd.concat([df.assign(_m=m) for m, df in data.items()])
    appear = allc.groupby("dev_id")["_m"].nunique()
    print(f"  server in {appear.get(0, 0)} files; clients appear in {sorted(set(appear.drop(0).values))} files each")
    print("\n=== TL per phase (rows with |offset - boundary| > 10 s) ===")
    timeline = []
    for mid, df in data.items():
        off = (df["ts_arrival"].values - t0) / 1e6
        df = df.assign(off=off)
        df = df[(df["off"] >= 0) & (df["off"] < 4980)]
        for a, b, nm, exp in PHASES:
            seg = df[(df["off"] >= a + TOL) & (df["off"] < b - TOL)]
            vc = seg["trust_level"].value_counts().to_dict()
            want = exp.get(f"M{mid:02d}", exp.get("other", exp.get("all")))
            share = sum(v for k, v in vc.items() if k in want) / max(1, len(seg))
            timeline.append({"monitor": mid, "phase": nm, "rows": len(seg),
                             "expected_TL": want, "TL_counts": vc, "share_expected": round(share, 4)})
    T = pd.DataFrame(timeline)
    for nm in [p[2] for p in PHASES]:
        s = T[T["phase"] == nm]
        print(f"  {nm:<17} share of rows at the expected TL: min {s['share_expected'].min():.3f}, "
              f"mean {s['share_expected'].mean():.3f}")
        bad = s[s["share_expected"] < 1]
        for _, r in bad.iterrows():
            print(f"      M{r['monitor']:02d} expected {r['expected_TL']} got {r['TL_counts']}")
    # per-client mode timeline at 1-s resolution (latest TL at each second), and flips
    print("\n=== mode timeline under the API 1.9.0 bands (1-s resolution) ===")
    rows = []
    for mid, df in data.items():
        off = (df["ts_arrival"].values - t0) / 1e6
        s = pd.Series(df["trust_level"].values, index=off).groupby(level=0).last().sort_index()
        sec = np.arange(0, 4980)
        idx = np.searchsorted(s.index.values, sec + 1, side="right") - 1
        tl = np.where(idx >= 0, s.values[np.clip(idx, 0, None)], np.nan)
        m = mode(np.nan_to_num(tl, nan=-1))
        m[np.isnan(tl)] = "none"
        flips = int((m[1:] != m[:-1]).sum())
        gl = (sec >= 2700 + TOL) & (sec < 3600 - TOL)
        print(f"  M{mid:02d}: mode changes {flips:>3}; in the GL phase {100 * np.mean(m[gl] == 'gossip'):5.1f}% "
              f"gossip, {100 * np.mean(m[gl] == 'local'):4.1f}% local (transient flips to TL 15)")
        for s_, tl_, m_ in zip(sec, tl, m):
            rows.append((mid, s_, tl_, m_))
    pd.DataFrame(rows, columns=["client", "second", "trust_level", "mode"]).to_csv(
        "results/disaster_fd_run1_modes.csv", index=False)


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/disaster_fd_run1")
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(d)
    open("results/disaster_fd_run1_check.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
