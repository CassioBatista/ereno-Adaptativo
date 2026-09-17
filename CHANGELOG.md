# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses simple
`vMAJOR.MINOR` tags.

This is the **v1.x (Paper 1 / JPDC) maintenance lineage**, kept separate from the
v2 adaptive line. See the [`v2.0.0`](https://github.com/CassioBatista/ereno-Adaptativo/releases/tag/v2.0.0)
release for the autonomous, decentralized FL↔GL switch and the external monitor.

## [v1.2] — 2026-09-17

Adds a **two-tier detection** capability. Alongside the Tier-1 per-attack XGBoost
specialists (k-of-n fusion, runtime FL⇄GL), a **Tier-2 protocol-specification
detector** — learned from benign traffic only, with zero attack examples — flags
novel attacks that violate IEC 61850 invariants. Also incorporates the Paper 1
reviewer response on FL as the normal mode of operation.

### Added
- **`fd/spec_detector.py`** — `SpecDetector` (VALUE + RATE rule families): benign-only
  fit, allowed-set rules for near-constant fields (e.g. `TTL ∈ {11000}`) and robust
  ranges for continuous ones. It models the shared benign baseline, so it is
  **monolithic / replicated → mode-independent**: it runs identically in FL and GL,
  with no aggregation.
- **Tier-2 wired into the live eval** ([main_dist.py](main_dist.py)) — opt-in via
  `detection.tier2.enabled`; per-round `ROUND-T2;<round>;<mode>;spec_fpr=..;novel_on_missed=..`
  line. Verified in both modes ([conf/experiments/ereno_tier2_demo.yaml](conf/experiments/ereno_tier2_demo.yaml),
  `results/tier2_demo.log`): the line appears in FL (rounds 1–6) and GL (7–12) with an
  identical `spec_fpr=0.4972%`, confirming mode-independence.
- **`scripts/spec_detector_selftest.py`** — data-backed regression test
  (FPR 0.52%, injection/high_StNum 100%, poisoned 99.56%).
- **Zero-day experiment sweep** — `scripts/zeroday_{anomaly,spec,rules,rules_full,rules_final}.py`.
  Efficacy at ~0.57% benign FPR: injection/high_StNum/poisoned_high_rate 100%,
  random_replay 98%, inverse_replay 57% (partial), masquerade irreducible.
- **Docs** — [docs/zeroday_two_tier.md](docs/zeroday_two_tier.md) (the negative→positive
  experimental arc), [docs/v1_2_integration_plan.md](docs/v1_2_integration_plan.md),
  [docs/refs_zeroday.bib](docs/refs_zeroday.bib).
- **Reviewer response — FL as the normal mode** ([docs/reviewer_response_fl_normal_mode.md](docs/reviewer_response_fl_normal_mode.md)):
  reframed on a single dissemination-work axis (Θ(N)) with the hub as the discriminator;
  `scripts/new_attack_scaling.py` (N=10/20/50/100 latency scaling) and
  `scripts/new_attack_convergence.py` (novel-attack propagation, FL vs GL, k≥1/k≥2).
- Project metadata: `CITATION.cff`, this `CHANGELOG.md`.

### Notes
- Stateful **SEQUENCE** rules (StNum/SqNum monotonicity over an ordered per-source
  stream, to close `inverse_replay`), the graduation loop (novel → labelled → new
  Tier-1 specialist), and monitor CoAP `intrusion_detected` notification (a v2
  backport) are **deferred to future work**.

## [v1.1] — 2026-08-27

Reproducibility artifact and feature-selection / robustness scripts.

### Added
- **`reproduce.py`** — single-command min-run: verify input hashes, train the 10
  specialists under fixed seeds, evaluate the full ERENO test, emit table, figure,
  and TP/FP/TN/FN audit, check against `MANIFEST.json`. Validated: F1 95.72 / #FP
  17,881 (OR) and 87.69 / #FP 49 (k≥2).
- **`MANIFEST.json`** — environment, seeds, input SHA-256 hashes, expected min-run
  numbers, and a provenance map (paper table → result file → hash → script).
- **`requirements.lock.txt`** — pinned environment (Python 3.14, XGBoost 3.3,
  Flower 1.32, scikit-learn 1.9, NumPy 2.5); **`REPRODUCE.md`** clean-environment recipe.
- Review-response scripts: `feature_ladder` (all-58 vs GRASP ablation),
  `lambda_combined_eval` / `lambda_multiseed` (5-seed λ sensitivity with 95% CIs),
  operational-metrics and inference-latency scripts, factorial+Holm analysis, and
  the raw result CSVs backing the paper tables.

## [v1.0] — 2026-08-24

Initial ReSIDS / ereno-Adaptativo (JCSA): centralized / federated / gossip in one
run with a **static** `FixedArchManager` schedule; `GlowStrategy` gossip
(ring/chain/star), GRASP feature selection, XGBoost per-attack specialists, OR /
k-of-n fusion, the GL=FL equivalence study, and node-reduction resilience.
