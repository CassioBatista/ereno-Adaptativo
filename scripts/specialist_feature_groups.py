#!/usr/bin/env python3
"""Which features do the specialists rely on? Gain share by feature group.

Bears on whether a specialist trained at one substation can be shared with another: fields
that IEC 61850 standardizes behave alike everywhere, electrical quantities depend on the
feeder, and some fields are set by each substation's configuration.
For the ten specialists of Table 7 (N = 10, GRASP-24, seed 42; the models trained by
scripts/local_baseline.py), the total gain of each booster is split over three groups:
  protocol behaviour        SqNum, StNum, cbStatus, sqDiff, timestampDiff,
                            timeFromLastChange, delay
  configuration-dependent   gooseTimeAllowedToLive, gooseLen, confRev, frameLengthDiff
  electrical                currents, voltages, RMS values and trapezoidal areas
The assignment of a feature to a group is the authors' reading of the ERENO attributes.

  python scripts/specialist_feature_groups.py        (runs local_baseline.py first if needed)
Out: results/specialist_feature_groups.txt
"""
import io
import json
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xgboost as xgb

import local_baseline as lb

NAME = {2: "isbA", 4: "isbC", 5: "ismA", 14: "isbARmsValue", 15: "isbBRmsValue", 17: "ismARmsValue",
        19: "ismCRmsValue", 20: "vsbARmsValue", 21: "vsbBRmsValue", 23: "vsmARmsValue",
        25: "vsmCRmsValue", 27: "isbBTrapAreaSum", 35: "vsmATrapAreaSum", 40: "SqNum", 41: "StNum",
        42: "cbStatus", 44: "gooseTimeAllowedToLive", 45: "gooseLen", 46: "confRev", 50: "sqDiff",
        54: "frameLengthDiff", 55: "timestampDiff", 57: "timeFromLastChange", 58: "delay"}
PROTOCOL, CONFIG = (40, 41, 42, 50, 55, 57, 58), (44, 45, 46, 54)
GROUP = {f: "protocol" if f in PROTOCOL else "configuration" if f in CONFIG else "electrical"
         for f in NAME}
ATTACK = {1: "random_replay", 2: "inverse_replay", 3: "masquerade_fake_fault",
          4: "masquerade_fake_normal", 5: "injection", 6: "high_StNum", 7: "poisoned_high_rate"}


def main():
    kf = lb.kf
    if not os.path.exists(f"{kf.MODELS}/meta.json"):
        kf.phase_train()
    feats = sorted(kf.FEATURES)
    assert set(feats) == set(NAME), "feature names out of step with the GRASP-24 set"
    meta = json.load(open(f"{kf.MODELS}/meta.json"))
    mean = defaultdict(float)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"gain share by feature group, {kf.N} specialists (N = {kf.N}, GRASP-24, seed 42)\n")
        print(f"{'node':>4} {'attack':<24} {'protocol':>9} {'config.':>9} {'electrical':>11}   top features")
        for i, c in enumerate(meta["spec"]):
            b = xgb.Booster()
            b.load_model(f"{kf.MODELS}/spec{i:02d}.json")
            g = b.get_score(importance_type="total_gain")
            s = sum(g.values())
            by, feat = defaultdict(float), {}
            for k, v in g.items():
                f = feats[int(k[1:])]
                by[GROUP[f]] += v / s
                feat[NAME[f]] = v / s
            for grp in ("protocol", "configuration", "electrical"):
                mean[grp] += by[grp] / len(meta["spec"])
            top = ", ".join(f"{n} {100 * v:.0f}%" for n, v in sorted(feat.items(), key=lambda x: -x[1])[:3])
            print(f"{i:>4} {ATTACK[c]:<24} {100 * by['protocol']:8.1f}% {100 * by['configuration']:8.1f}% "
                  f"{100 * by['electrical']:10.1f}%   {top}")
        print("\nmean over the specialists: " +
              ", ".join(f"{g} {100 * mean[g]:.1f}%" for g in ("protocol", "configuration", "electrical")))
    open("results/specialist_feature_groups.txt", "w").write(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
