#!/usr/bin/env python3
"""Regression self-test for fd/spec_detector.py (Tier-2 zero-day, VALUE+RATE).

Data-backed (loads the real ERENO train/test, ~a couple of minutes). Fits the
SpecDetector on benign only, using the fields available in the GRASP-24 model vector
(VALUE F40,41,42,44,57 + RATE F55; F56 is not in GRASP), and asserts the per-attack
held-out recall reproduces scripts/zeroday_rules_final.py within tolerance, at low FPR.
Rodar: ~/venv-ereno314/bin/python scripts/spec_detector_selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import python.util as util
from fd.spec_detector import SpecDetector

# GRASP-available spec fields as FULL-58 0-based columns (F_n -> n-1):
FIELDS = {"VALUE": [39, 40, 41, 43, 56],   # F40 SqNum, F41 StNum, F42 cbStatus, F44 TTL, F57 timeFromLastChange
          "RATE":  [54]}                    # F55 (rate); F56 not in GRASP-24
DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"

ok = True
def check(cond, msg):
    global ok
    print(("  OK  " if cond else " FAIL ") + msg); ok = ok and cond

print("[spectest] carregando treino (benigno p/ fit)...")
Xtr, ytr, cvals = util.load_arff(f"{DATASET}.csv")
nc = util.normal_class
det = SpecDetector(FIELDS).fit(Xtr[ytr == nc])
del Xtr, ytr
print("[spectest] regras aprendidas:")
for c, (k, s) in sorted(det.rules.items()):
    spec = ("{" + ",".join(f"{v:g}" for v in sorted(s)) + "}") if k == "set" else f"[{s[0]:.4g},{s[1]:.4g}]"
    print(f"    col{c} (F{c+1}): {spec}")

print("[spectest] carregando teste...")
Xte, yte, _ = util.load_arff(f"{TESTFILE}.csv")
util.normal_class = nc
flagged = det.flag(Xte)

fpr = 100.0 * float(flagged[yte == nc].mean())
rec = {}
for c in range(len(cvals)):
    if c == nc:
        continue
    m = (yte == c)
    rec[cvals[c]] = 100.0 * float(flagged[m].mean()) if m.any() else float("nan")
print(f"\n[spectest] benign FPR = {fpr:.3f}%")
for a, r in rec.items():
    print(f"[spectest]   {a:>22}: recall={r:6.2f}%")

# ---- regression asserts (tolerant; GRASP-available fields, F55-only rate) ----
check(fpr <= 1.5, f"benign FPR low ({fpr:.2f}% <= 1.5%)")
check(rec["injection"] >= 99.0, f"injection ~100% ({rec['injection']:.1f})")
check(rec["high_StNum"] >= 99.0, f"high_StNum ~100% ({rec['high_StNum']:.1f})")
check(rec["poisoned_high_rate"] >= 95.0, f"poisoned_high_rate caught by RATE F55 ({rec['poisoned_high_rate']:.1f})")
check(rec["random_replay"] >= 75.0, f"random_replay caught (value+rate) ({rec['random_replay']:.1f})")
check(rec["masquerade_fake_normal"] <= 10.0, f"masquerade_fake_normal irreducible/low ({rec['masquerade_fake_normal']:.1f})")

print("\n" + ("SPEC DETECTOR SELFTEST OK" if ok else "SPEC DETECTOR SELFTEST FAILED"))
sys.exit(0 if ok else 1)
