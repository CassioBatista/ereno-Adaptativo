#!/usr/bin/env python3
"""Window triage on Electra Modbus (label set electra-4), with the policy used for ERENO and
the corrected CICIDS2017: network-wide volume/fraction rule (margin x1.25) OR per-specialist
counts (T_c = max(ceil(1.25 * max benign) + 1, 2)).

Same specialists and split as cross_domain_electra.py (clean4: the path-labelled types
MITM_UNALTERED, REPLAY_ATTACK and READ_ATTACK removed from the stream; content features;
global time order, first 70 % train). Electra has no attack-free period, so thresholds are
calibrated on the attack-free windows of the training period and applied to the test
period (calibrate on the past, apply to the future). Time is in microseconds.

  python scripts/_electra_window_triage.py
Out: results/electra_window_triage.txt
"""
import io
import math
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402
from cross_domain_electra import CLEAN4, CONTENT  # noqa: E402

WLENS = (1, 10, 60)
FLOOR = 2


def window_stats(sec, fused, V, cats, wlen):
    w = np.floor(sec / wlen).astype(np.int64)
    ncat = len(cats)
    per = np.column_stack([V[:, [c, c + ncat]].max(1) for c in range(ncat)]) * fused[:, None]
    d = pd.DataFrame(per, columns=cats)
    d["_n"] = 1
    d["_f"] = fused
    return d.groupby(w).sum()


def main():
    df = pd.read_csv(os.path.expanduser("~/datasets/electra/electra_modbus.csv"),
                     usecols=["Time"] + CONTENT + ["label"], dtype={"label": "category"})
    df = df[df["label"].isin(CLEAN4 + ["NORMAL"])].sort_values("Time", kind="stable")
    X = df[CONTENT].to_numpy(np.float32)
    lab = df["label"].astype(str).values
    sec = df["Time"].to_numpy(np.float64) / 1e6
    del df
    n = len(X); cut = int(0.7 * n); tr = np.arange(n) < cut
    cats = CLEAN4
    N = 2 * len(cats)
    rng = np.random.default_rng(C.SEED)
    ben = np.where(lab[tr] == "NORMAL")[0]
    shares = np.array_split(rng.permutation(ben), N)
    halves = {c: np.array_split(rng.permutation(np.where(lab[tr] == c)[0]), 2) for c in cats}
    models = []
    for i in range(N):
        c = cats[i % len(cats)]
        pos, neg = halves[c][i // len(cats)], shares[i]
        rows = np.r_[pos, neg]
        p = dict(C.XGB); p["scale_pos_weight"] = len(neg) / max(len(pos), 1)
        models.append(xgb.train(p, xgb.DMatrix(X[tr][rows], label=np.r_[np.ones(len(pos)), np.zeros(len(neg))]),
                                num_boost_round=10))
    V = np.column_stack([(m.predict(xgb.DMatrix(X)) >= 0.5).astype(np.int8) for m in models])
    fused = (V.sum(1) >= 2).astype(np.int8)
    atk = lab != "NORMAL"
    print(f"packets {n:,} (train {int(tr.sum()):,}); flagged: train {int(fused[tr].sum()):,} "
          f"(on normal {int((fused[tr] & ~atk[tr]).sum()):,}), test {int(fused[~tr].sum()):,} "
          f"(on normal {int((fused[~tr] & ~atk[~tr]).sum()):,})")
    print(f"capture span {(sec.max() - sec.min()) / 3600:.1f} h; train ends at "
          f"{(sec[cut] - sec.min()) / 3600:.1f} h")
    for wlen in WLENS:
        W = window_stats(sec, fused, V, cats, wlen)
        wa = pd.Series(atk.astype(int)).groupby(np.floor(sec / wlen).astype(np.int64)).sum()
        wc = pd.Series(lab).groupby(np.floor(sec / wlen).astype(np.int64)).agg(
            lambda s: s[s != "NORMAL"].mode().iloc[0] if (s != "NORMAL").any() else "NORMAL")
        W["atk"] = wa.reindex(W.index).values > 0
        W["cls"] = wc.reindex(W.index).values
        wtr = W.index < math.floor(sec[cut] / wlen)
        cal = W[wtr & ~W["atk"].values]
        ev = W[~wtr]
        T = int(math.ceil(1.25 * cal["_f"].max())) + 1
        f = 1.25 * (cal["_f"] / cal["_n"]).max()
        Tc = np.maximum(np.ceil(1.25 * cal[cats].max().values) + 1, FLOOR)
        glob = (ev["_f"] >= T) | ((ev["_f"] / ev["_n"]) > f)
        spec = (ev[cats].values >= Tc).any(1)
        a = ev["atk"].values
        print(f"\n=== window {wlen} s: calibration {len(cal):,} attack-free windows (train period); "
              f"test {int(a.sum()):,} attack / {int((~a).sum()):,} benign windows")
        print(f"  thresholds: T={T}, f={f:.4f}; per specialist " +
              ", ".join(f"{c.split('_')[0]}={int(t)}" for c, t in zip(cats, Tc)))
        for name, r in (("network-wide (ERENO policy)", glob.values), ("per specialist", spec),
                        ("network-wide OR per specialist", glob.values | spec)):
            print(f"  {name:<31} attack windows {int(r[a].sum())}/{int(a.sum())} ({100 * r[a].mean():.1f}%)   "
                  f"benign false alarms {int(r[~a].sum())}/{int((~a).sum())} ({100 * r[~a].mean():.3f}%)")
        comb = glob.values | spec
        print("  per category (network-wide | OR per specialist):",
              {c.split("_")[0]: f"{100 * glob.values[(ev['cls'] == c).values].mean():.0f}% | "
                                f"{100 * comb[(ev['cls'] == c).values].mean():.0f}% of {int((ev['cls'] == c).sum())}"
               for c in cats if (ev["cls"] == c).any()})


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/electra_window_triage.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
