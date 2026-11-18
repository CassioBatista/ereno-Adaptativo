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
| R0 | OR: alarm if any specialist's score >= 0.5 *(corrected from "> 0.5", see the change log)* | fixed 0.5 |
| R1 | k >= 2: alarm if at least two specialists' scores >= 0.5 (current) *(corrected from "> 0.5", see the change log)* | fixed 0.5 |
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
reported for the comparison in the paper (in the description of the fusion rule, in the
methodology) *(amended 2026-11-17, see the change log; was "(RQ3)")*, not used for the
decision.

## Change log

- **2026-11-17, wording, section 5: where the comparison is reported.** The second paper no
  longer has research questions (the results are organised by the gaps of its introduction),
  and the fusion rule is described in its methodology, since it was fixed before any
  evaluation run. "(RQ3)" is replaced by the methodology. Nothing in the selection, its
  criteria or its results changes.

- **2026-11-15, correction (criterion), section 5 (H1, H2): an attack with zero Full recall
  is not kept.** Defect: H1 and H2 compare each attack's recall in a scenario with 90 % of its
  Full recall under the same rule; for an attack the rule does not detect at all (Full recall
  0), 0 >= 0.9 x 0 holds, so a rule passes H1 and H2 on that attack by missing it. Found in
  the repetition of this selection on the evaluation-v1 ERENO selection, on the calibration
  set only (`docs/EVALUATION_PROTOCOL.md`, change log of 2026-11-15;
  `scripts/fusion_rule_selection_ev1.py`): R3 detects masquerade_fake_normal on 0.0 % and
  masquerade_fake_fault on 5.0 % of its calibration rows in all three scenarios, and passed
  H1-H3 there (the only rule to do so), while in the selection above it failed H1 and H2 with
  the same per-attack recall (83.06 % overall, 3.8 % and 0.0 % on the two masquerade types,
  against 82.79 %, 5.0 % and 0.0 %). Its per-attack recalls there are equal, to 0.1 pp, in the
  three scenarios, so the attack that failed differs by less than 0.1 pp and must have a Full
  recall under 1 %: masquerade_fake_normal, with a Full recall above zero and under 0.05 %,
  lower in Absence and Silent. The outcome of the criterion turned on noise near zero, not on
  the rules. The correction was made after the
  calibration result was seen (the ERENO test file was not read), and it is applied because
  the uncorrected criterion selects, on that ground alone, a rule that misses two of the seven
  attack types. Correction: in H1 and H2 an attack with zero Full recall counts as not kept.
  Effect: (1) on the evaluation-v1 selection, R3 fails H1 and H2, no rule meets H1-H3, and R0
  alone meets two constraints and is chosen (`results/fusion_rule_selection_ev1.txt`; the run
  under the original criterion is kept as `results/fusion_rule_selection_ev1_h12original.txt`).
  (2) The selection above is unchanged: the correction can only turn a pass of H1 or H2 into a
  failure; R0, the only rule meeting two constraints there, has Full recall above zero on
  every attack (at least 99.6 %), so it keeps H1 and H2, and no other rule can gain a
  constraint. `scripts/fusion_rule_selection.py` is left unchanged as the record of that run.

- **2026-11-12, wording correction, section 3 (R0, R1): the fixed threshold is inclusive,
  score >= 0.5.** The table stated "> 0.5"; ReSIDS flags a message when a specialist's
  probability is at least 0.5 (`main_dist.py`, k-of-n prediction), and the evaluation applies
  the rule as implemented. The selection script (`scripts/fusion_rule_selection.py`, R0 and
  R1) used "> 0.5" and is left unchanged as the record of the run. The two forms differ only
  for a score of exactly 0.5, and no such score occurs in the selection data: 0 of the
  41,379,072 test scores (`results/.cache/ereno_probs_n14_k2audit.npz`, 2,955,648 rows × 14
  specialists) and 0 of the 1,949,738 calibration scores
  (`results/.cache/ereno_probs_n14_calib.npz`, 139,267 rows × 14), none within 10^-4 of 0.5.
  The results in `results/fusion_rule_selection.txt` and the choice of R0 are therefore the
  same under either form. The calibrated rules R2-R4 (score > t) are not affected.
