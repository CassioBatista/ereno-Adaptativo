# Fusion rule selection (pre-registered)

Status: **pre-registered on 2026-11-03 (tag `fusion-v1`), committed before the selection in
`scripts/fusion_rule_selection.py` was run** (only its calibration phase, which scores the
held-out rows and computes no rule, ran before). The rule chosen here is then applied unchanged to every evaluated domain; only its
thresholds are recalibrated in each domain, by the same procedure, on that domain's own
calibration data.

## 1. Why

The exploratory fusion audit (`scripts/k_fusion_audit*.py`, `results/k_fusion_*.txt`) found
that on ERENO k >= 2 acts as a veto rather than as corroboration: 99.6 % of its false
positives are the two specialists of one attack firing together; under node loss it loses
the attacks left with one specialist; a silent specialist vetoes its class. Its thresholds
were matched in sample, so no rule could be preferred. This selection uses out-of-sample
calibration and fixed criteria.

## 2. Development data (ERENO only)

- Specialists: the 14 of the paper's operating point (two per attack, 24 features, seed 42,
  benign cap 500k), `results/.cache/k2audit_models/` (trained by `k_fusion_audit.py train`).
- Calibration set: the 20 % of each specialist's partition held out by `train_test_split`
  (test_size 0.2, seed 42) when it was trained. These rows were never used to train any
  specialist (partitions are disjoint).
- Test set: the ERENO test file (2,955,648 samples), used once, after calibration.

## 3. Candidate rules

| Id | Rule | Threshold |
|---|---|---|
| R0 | OR: alarm if any specialist's score > 0.5 | fixed 0.5 |
| R1 | k >= 2: alarm if at least two specialists' scores > 0.5 (current) | fixed 0.5 |
| R2 | max score: alarm if the highest score > t | one t, calibrated |
| R3 | per-specialist OR: specialist i alarms if its score > t_i; alarm if any does | one t_i per specialist, calibrated to FPR target / (number of specialists present) |
| R4 | pair agreement: an attack alarms if every present specialist of that attack scores > t (a single present specialist alarms alone); alarm if any attack does | one t, calibrated |

Calibration: thresholds are set on the benign samples of the calibration set so that the
fused alarm rate on them equals the **target false-positive rate of 0.1 %** (R3: each
specialist at 0.1 % divided by the number of specialists present).

## 4. Scenarios (test set)

- **Full:** all 14 specialists.
- **Absence:** one specialist per attack (the first of each pair); calibrated rules are
  recalibrated on the 7 present, since an absent specialist is known.
- **Silent fault:** all 14 present, the second specialist of each pair outputs 0; thresholds
  stay as calibrated for Full, since a silent fault is not known.

## 5. Criteria and decision

Hard constraints (a rule must meet all three):

- **H1, absence:** in Absence, every attack keeps at least 90 % of its Full recall under the
  same rule.
- **H2, silent fault:** in Silent fault, every attack keeps at least 90 % of its Full
  recall under the same rule.
- **H3, out-of-sample false positives:** in Full, the test false-positive rate is at most
  twice the target (0.2 %).

Decision: among the rules that meet H1-H3, the one with the highest overall test recall in
Full. Rules within 0.5 percentage points of the best are tied, and the tie goes to the
simpler rule, in the order R2, R3, R4. If no rule meets all three constraints, the rule
meeting the most constraints is chosen, then by recall, then by simplicity.

Everything else (F1-score, precision, per-attack recall, all scenarios for every rule) is
reported for the comparison in the paper (RQ3), not used for the decision.

## Change log

- (none)
