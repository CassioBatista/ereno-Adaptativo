#!/usr/bin/env python3
"""Does window triage survive a cadence that Disaster-FD changes under delay?

The event window is whatever the agent scored between two local Disaster-FD ticks, and
the detector may stretch its tick when the network degrades. Thresholds, however, are
calibrated once. This sweep calibrates the triage at the 1-s cadence -- the margin policy
validated out of sample (max benign x 1.25) -- and applies those FIXED thresholds while the
local cadence varies:

  time-based ticks   0.25, 0.5, 1, 2, 5, 10 s           (traffic with time, IEC 61850)
  count-based ticks  500, 1000, 4759, 20000 samples      (data without time, CICIoT2023-like)

Rules:
  volume    n_flags >= T                      grows with the window if errors were uniform
  fraction  n_flags / window_samples > f      would be length-invariant IF errors were i.i.d.
Neither premise holds here: the trace arrives in bursts (a 10-s tick still holds one
~4,760-sample burst) and false positives cluster in time (a short window landing on a
cluster has a high fraction). What the sweep measures is how the two behave regardless.
A window is an attack window if it holds any attack sample. Fusion k>=2, full pool,
model of the scenario streams (benign cap 500k).

  python scripts/cadence_sweep.py
Out: results/cadence_sweep_ereno.{txt,csv,png,pdf}
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from scored_cache import get_scored

K = 2
MARGIN = 0.25
TIME_TICKS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
COUNT_TICKS = [500, 1000, 4759, 20000]
MIN_SAMPLES = 50
OUT = "results/cadence_sweep_ereno"


def windows(ts, flag, votes, is_atk, tick):
    kind, val = tick
    if kind == "time":
        t0 = float(ts[0])
        win = np.floor((ts - t0) / val).astype(np.int64)
        bnd = np.searchsorted(win, np.arange(int(win[-1]) + 2))
    else:
        n = len(ts)
        bnd = np.minimum(np.arange((n + val - 1) // val + 1) * val, n)
    lo, hi = bnd[:-1], bnd[1:]
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    cs = np.r_[0, np.cumsum(flag)]
    ca = np.r_[0, np.cumsum(is_atk)]
    nf = cs[hi] - cs[lo]
    n = hi - lo
    atk = (ca[hi] - ca[lo]) > 0
    masked = np.where(flag, votes, 0)
    kv = np.maximum.reduceat(masked, lo)
    return {"nf": nf, "n": n, "frac": nf / n, "atk": atk, "kv": kv}


def main():
    ts, fired, y, nc, cv, spec, _ = get_scored()
    votes = fired.sum(axis=0)
    flag = votes >= K
    is_atk = y != nc

    # calibrate ONCE, at the 1-s cadence, on benign windows only
    w1 = windows(ts, flag, votes, is_atk, ("time", 1.0))
    b1 = ~w1["atk"]
    T = int(math.ceil(w1["nf"][b1].max() * (1 + MARGIN))) + 1
    F = float(w1["frac"][b1].max() * (1 + MARGIN))
    lines, rows = [], []

    def out(s=""):
        print(s, flush=True); lines.append(s)

    out("ERENO cadence sweep — thresholds calibrated ONCE at the 1-s cadence, then held fixed")
    out(f"calibration (benign windows, 1 s, margin x{1 + MARGIN}): volume T = {T}, fraction f = {F:.4f}")
    out()
    out(f"{'local tick':>16s} {'windows':>8s} {'atk':>5s} {'med n':>7s} | "
        f"{'volume TP':>10s} {'volume FP':>10s} | {'fraction TP':>12s} {'fraction FP':>12s} | "
        f"{'benign max frac':>15s} {f'(n>={MIN_SAMPLES})':>10s}")
    for kind, vals in (("time", TIME_TICKS), ("count", COUNT_TICKS)):
        for v in vals:
            w = windows(ts, flag, votes, is_atk, (kind, v))
            A, B = w["atk"], ~w["atk"]
            vol = w["nf"] >= T
            fra = w["frac"] > F
            both = vol | fra
            big = B & (w["n"] >= MIN_SAMPLES)
            # recalibrated AT this cadence with the same margin policy (in-sample, so its
            # FP is zero by construction -- it shows what recalibration can recover)
            Tc = int(math.ceil(w["nf"][B].max() * (1 + MARGIN))) + 1
            Fc = float(w["frac"][B].max() * (1 + MARGIN))
            rec = (w["nf"] >= Tc) | (w["frac"] > Fc)
            r = {"tick": f"{v:g} s" if kind == "time" else f"{v} samples", "kind": kind,
                 "value": v, "windows": int(len(w["n"])), "attack_windows": int(A.sum()),
                 "benign_windows": int(B.sum()), "median_samples": float(np.median(w["n"])),
                 "vol_tp": int((vol & A).sum()), "vol_fp": int((vol & B).sum()),
                 "frac_tp": int((fra & A).sum()), "frac_fp": int((fra & B).sum()),
                 "both_tp": int((both & A).sum()), "both_fp": int((both & B).sum()),
                 "recal_T": Tc, "recal_f": Fc, "recal_tp": int((rec & A).sum()),
                 "benign_max_frac": float(w["frac"][B].max()) if B.any() else 0.0,
                 "benign_max_frac_big": float(w["frac"][big].max()) if big.any() else 0.0}
            rows.append(r)
            out(f"{r['tick']:>16s} {r['windows']:>8d} {r['attack_windows']:>5d} "
                f"{r['median_samples']:>7.0f} | "
                f"{r['vol_tp']:>4d}/{r['attack_windows']:<5d} {r['vol_fp']:>4d}/{r['benign_windows']:<5d} | "
                f"{r['frac_tp']:>5d}/{r['attack_windows']:<6d} {r['frac_fp']:>5d}/{r['benign_windows']:<6d} | "
                f"{r['benign_max_frac']:>15.4f} {r['benign_max_frac_big']:>10.4f}")
    out()
    out("combined rule (volume OR fraction), thresholds fixed at 1 s  |  recalibrated at each "
        "cadence (in-sample, FP 0 by construction)")
    out(f"{'local tick':>16s} {'both TP':>10s} {'both FP':>10s} | {'recal T':>8s} {'recal f':>8s} "
        f"{'recal TP':>10s}")
    for r in rows:
        out(f"{r['tick']:>16s} {r['both_tp']:>4d}/{r['attack_windows']:<5d} "
            f"{r['both_fp']:>4d}/{r['benign_windows']:<5d} | {r['recal_T']:>8d} {r['recal_f']:>8.4f} "
            f"{r['recal_tp']:>4d}/{r['attack_windows']:<5d}")
    out()
    out("volume TP/FP = attack/benign windows alarmed by n_flags >= T; fraction likewise by "
        f"n_flags/window_samples > f. 'benign max frac' is the largest flagged fraction in any "
        f"benign window at that cadence; the last column restricts it to windows of at least "
        f"{MIN_SAMPLES} samples.")

    os.makedirs("results", exist_ok=True)
    with open(f"{OUT}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    with open(f"{OUT}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, kind, title in ((ax[0], "time", "Time-based local tick (s)"),
                           (ax[1], "count", "Count-based local tick (samples)")):
        sel = [r for r in rows if r["kind"] == kind]
        x = [r["value"] for r in sel]
        a.plot(x, [r["vol_tp"] / r["attack_windows"] for r in sel], "o-", color="#d62728",
               label="volume — attack windows caught")
        a.plot(x, [r["vol_fp"] / max(r["benign_windows"], 1) for r in sel], "o--", color="#d62728",
               label="volume — benign windows alarmed")
        a.plot(x, [r["frac_tp"] / r["attack_windows"] for r in sel], "s-", color="#2ca02c",
               label="fraction — attack windows caught")
        a.plot(x, [r["frac_fp"] / max(r["benign_windows"], 1) for r in sel], "s--", color="#2ca02c",
               label="fraction — benign windows alarmed")
        a.axvline(1.0 if kind == "time" else 4759, color="grey", ls=":", lw=1)
        a.set_xscale("log"); a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3)
        a.set_title(title); a.set_xlabel("local tick (calibrated at the dotted line)")
    ax[0].set_ylabel("share of windows")
    ax[1].legend(fontsize=8, loc="center right")
    fig.suptitle("Triage calibrated once at 1 s, applied while the Disaster-FD cadence varies",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=180); fig.savefig(f"{OUT}.pdf")
    print(f"\n-> {OUT}.{{txt,csv,png,pdf}}")


if __name__ == "__main__":
    main()
