#!/usr/bin/env python3
"""Replay the Disaster-FD telemetry of a run through the API 1.9.0 agent policy.

From 1.9.0 Disaster-FD detects failures and publishes a trust level (TL) per monitor; each
ReSIDS client decides its own learning mode from the TL of its colocated monitor
(docs/TRUST_LEVEL.md):

    band(TL) = local if TL < gl,  gossip if gl <= TL < fl,  federated if TL >= fl
    a new band is committed once it has held for the settle time S;
    no observation for longer than F (stale TL) -> local.

Each CSV row becomes a trust-level observation (schemas/trust_level.schema.json); each
committed switch an architecture_change event (decided_by: agent), one stream per client,
checked by scripts/validate_events.py. round = the index of the 5-s probe cycle since run_t0
(estimated as in scripts/inspect_disaster_fd_trace.py).

  python scripts/tl_policy_replay.py ~/datasets/disaster_fd_run1 [--settle 5] [--stale 15]
Out: results/tl_policy/events_client<NN>.jsonl, results/tl_policy/tl_monitor07.jsonl
     (sample observation stream), results/tl_policy_replay.txt
"""
import argparse
import glob
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_events  # noqa: E402

N_CLIENTS, CYCLE = 14, 5.0
IF = {"server": 50, "predecessor": 15, "successor": 15}
BOUNDS = [900, 1800, 2700, 3600]                    # phase boundaries (s since run_t0)
EXPECTED = [(0, 900, "local"), (900, 1800, "federated"), (1800, 2700, "local"),
            (2700, 3600, "gossip"), (3600, 4980, "federated")]


def iso(us):
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc).isoformat()


def band(tl, gl, fl):
    return "federated" if tl >= fl else "gossip" if tl >= gl else "local"


def observations(df, mid):
    pred, succ = ((mid - 2) % N_CLIENTS) + 1, (mid % N_CLIENTS) + 1
    role = {0: "server", pred: "predecessor", succ: "successor"}
    for r in df.itertuples(index=False):
        rl = role[int(r.dev_id)]
        yield {"monitor": f"monitor{mid:02d}", "client": mid, "msg_id": int(r.msg_id),
               "ts": iso(int(r.ts_arrival)),
               "target": {"dev_id": int(r.dev_id), "role": rl, "impact_factor": IF[rl],
                          "trusted": bool(r.trusted), "is_timeout": bool(r.is_timeout)},
               "trust_level": float(r.trust_level), "threshold": float(r.threshold),
               "trusted_system": bool(r.trusted_system)}


def policy(t, tl, S, F, gl, fl, t_end):
    """t: observation times (s, sorted), tl: TL after each. Returns the committed switches
    [(t_commit, from, to, reason, tl_used, t_obs_used)]. Starts in local (no TL yet)."""
    mode, out = "local", []
    cand, t_cand = None, None
    for i in range(len(t)):
        # stale check over the gap before this observation
        if i > 0 and t[i] - t[i - 1] > F and mode != "local":
            out.append((t[i - 1] + F, mode, "local", "tl_stale", None, None))
            mode, cand = "local", None
        b = band(tl[i], gl, fl)
        if b == mode:
            cand = None
        elif b != cand:
            cand, t_cand = b, t[i]
        t_next = t[i + 1] if i + 1 < len(t) else t_end
        horizon = t_next if t_next - t[i] <= F else t[i] + F       # TL valid until here
        if cand is not None and horizon - t_cand >= S:
            # the band held for S before it changed or went stale: commit at t_cand + S
            tc = t_cand + S
            j = np.searchsorted(t, tc, side="right") - 1
            out.append((tc, mode, cand, "trust_level", float(tl[j]), float(t[j])))
            mode, cand = out[-1][2], None
    return out


def main(d, S, F, gl, fl):
    files = sorted(glob.glob(os.path.join(d, "runs", "*", "*", "csv", "log_monitor*_Fleet_*.csv")))
    data = {int(re.search(r"monitor(\d+)", os.path.basename(f)).group(1)):
            pd.read_csv(f).sort_values(["ts_arrival", "msg_id"], kind="stable") for f in files}
    jumps = [df.loc[df["trust_level"] >= 80, "ts_arrival"].min() for df in data.values()]
    t0 = min(jumps) - 900e6
    print(f"run: {d}\nmonitors: {len(data)}  run_t0 (estimated) {iso(t0)}")
    print(f"policy: bands gl={gl} fl={fl}; settle S={S} s; stale after F={F} s -> local\n")
    tl_v = Draft202012Validator(json.load(open("schemas/trust_level.schema.json")))
    os.makedirs("results/tl_policy", exist_ok=True)
    n_obs = n_bad = 0
    lat, rows = [], []
    for mid, df in sorted(data.items()):
        for o in observations(df, mid):
            n_obs += 1
            n_bad += not tl_v.is_valid(o)
            if mid == 7:
                rows.append(o)
        t = (df["ts_arrival"].values - t0) / 1e6
        keep = (t >= 0) & (t < 4980)
        t, tl, us = t[keep], df["trust_level"].values[keep], df["ts_arrival"].values[keep]
        gaps = np.diff(t)
        sw = policy(t, tl, S, F, gl, fl, 4980.0)
        evs = []
        for k, (tc, a, b, reason, tlu, tou) in enumerate(sw, 1):
            e = {"seq": k, "ts": iso(t0 + tc * 1e6), "type": "architecture_change",
                 "round": int(tc // CYCLE), "fed_round": None, "from_mode": a, "to_mode": b,
                 "mode": b, "reason": reason, "decided_by": "agent",
                 "trust_level": tlu, "tl_bands": {"gl": gl, "fl": fl},
                 "tl_ts": iso(t0 + tou * 1e6) if tou is not None else None,
                 "fd_monitor": f"monitor{mid:02d}",
                 "detail": f"band held {S:g} s" if reason == "trust_level" else f"no TL for {F:g} s"}
            evs.append(e)
        p = f"results/tl_policy/events_client{mid:02d}.jsonl"
        with open(p, "w") as f:
            for e in evs:
                f.write(json.dumps(e) + "\n")
        # time in the expected mode per phase (1-s grid), latency after each boundary
        grid = np.arange(0, 4980)
        tc = np.array([s[0] for s in sw]); to = [s[2] for s in sw]
        idx = np.searchsorted(tc, grid, side="right") - 1
        m = np.array(["local" if i < 0 else to[i] for i in idx])
        share = [np.mean(m[a:b] == want) for a, b, want in EXPECTED]
        for B in BOUNDS:
            after = tc[(tc >= B - 10) & (tc < B + 60)]
            lat.append(after.max() - B if len(after) else np.nan)
        print(f"  client {mid:02d}: {len(evs)} switches "
              f"[{' '.join(f'{s[1][0]}>{s[2][0]}@{s[0]:.0f}' for s in sw)}]  "
              f"max obs gap {gaps.max():.1f} s  in expected mode per phase "
              f"{' '.join(f'{100 * x:.1f}' for x in share)} %")
    with open("results/tl_policy/tl_monitor07.jsonl", "w") as f:
        for o in rows:
            f.write(json.dumps(o) + "\n")
    lat = np.array(lat)
    print(f"\ntrust-level observations: {n_obs:,}, schema-valid {n_obs - n_bad:,}")
    print(f"switch latency after a phase boundary (last switch - boundary): median "
          f"{np.nanmedian(lat):.1f} s, max {np.nanmax(lat):.1f} s, missing {int(np.isnan(lat).sum())}")
    print("\n=== conformance (scripts/validate_events.py) ===")
    bad = 0
    for mid in sorted(data):
        p = f"results/tl_policy/events_client{mid:02d}.jsonl"
        with redirect_stdout(io.StringIO()):
            errs = validate_events.check_events(p)
        bad += bool(errs)
        if errs:
            print(f"  {p}: FAIL {errs[:3]}")
    print(f"  {len(data) - bad}/{len(data)} client streams CONFORMANT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="~/datasets/disaster_fd_run1")
    ap.add_argument("--settle", type=float, default=5.0)
    ap.add_argument("--stale", type=float, default=15.0)
    ap.add_argument("--gl", type=float, default=30.0)
    ap.add_argument("--fl", type=float, default=50.0)
    a = ap.parse_args()
    run = os.path.expanduser(a.run)
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(run, a.settle, a.stale, a.gl, a.fl)
    open("results/tl_policy_replay.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
