# Evaluation protocol (pre-registered)

Status: **pre-registered on 2026-11-03 (tag `evaluation-v1`), committed before any
evaluation run.**
After that, only error corrections, each recorded in the change log with its date.

## 1. Domains and data

Selected by `docs/DATASET_PROTOCOL.md` (tag `protocol-v1`, `results/protocol_v1/summary.md`):

| Domain | Release | Attack types | Order | Window |
|---|---|---|---|---|
| IEC 61850 substation | ERENO | 7 | time | 1 s |
| enterprise network | CICIDS2017, improved | 26 (the qualified `Label` values, "Attempted" labels included) | time | 10 s |
| industrial control | Morris-4 (gas pipeline, 2015) *(amended, see the change log; was Morris-3)* | 30 specific attacks (5 excluded by Q2a) | time | 100 s |
| IoT | CICIoT2023, the 10 part files screened | 30 (three excluded by Q2b) | none (rows shuffled) | no window |

Results on the types excluded by the screening are reported separately (protocol-v1,
section 8). Every row of a qualified type and of the benign class is used, except ERENO
benign training rows, capped at 500,000 (memory), as in the earlier runs.

## 2. From raw data to distributed training

The steps between the raw release and distributed training are the same for the four
datasets: **(1) dataset preparation, (2) feature selection, (3) partitioning.**

### 2.1 Dataset preparation

Common rule: rows of the attack types excluded by the screening are removed; misaligned
rows (Q1a) are removed; candidate features are the content columns, i.e. every column
except **identifiers**, the fields that locate a row in a capture or name an endpoint and
that no attack manipulates (time, addresses, ports, frame numbers, checksums, session ids);
the split of section 4 is applied before any feature selection.

| Dataset | Raw release | Label | Candidate features | Identifiers (removed) |
|---|---|---|---|---|
| ERENO | `all_in_one_ereno_{train,test}.csv` (converted release) | class (7 attacks + normal) | 58 numeric columns minus absolute time | `Time`, `t`, `GooseTimestamp`. `SqNum` and `StNum` are GOOSE message fields that the attacks manipulate (`high_StNum`, injection, replay), so they are content |
| CICIDS2017, improved | `CICIDS2017_improved.zip` (version of 27 April 2023), five days | `Label` (26 qualified values) | numeric flow statistics | `id`, `Flow ID`, `Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Timestamp` |
| Morris-4 *(amended)* | `IanArffDataset.arff` (Web Archive copy, checksums in `results/morris_ics_md5sums.txt`) | `specific result` (30 qualified attacks + normal) | Modbus and process fields | `time`, `crc rate` |
| CICIoT2023 | 10 part files of the CSV release | the 30 qualified labels | 46 flow features | none (the release has no identifier column) |

### 2.2 Feature selection: GRASP, same recipe in every domain

The recipe of the pipeline (`scripts/run_grasp_cross_domain.sh`), run on the **training split
only**: (a) binary GRASP per attack type, attack vs benign (GraspVND, XGBoost, 5-fold
cross-validation, lambda = 0.05, sample 60,000, benign cap 60,000, attack rows capped at the
benign count, stop after 5 iterations without improvement *(amended, see the change log;
was 15)*); (b) penalized multiclass GRASP over all types (lambda = 0.05, sample 150,000,
stop after 5 iterations without improvement *(amended, see the change log; was 15)*; on a
class-balanced copy when benign is under 10 % of the training rows,
i.e. CICIoT2023); (c) the selected set is the union of (a) and (b). Types with fewer than
10 training rows get no per-type run and are covered by (b). No wall-clock limit applies to
(a) or (b): a run stops only on its iteration rule *(correction, see the change log of
2026-11-12)*.

- ERENO: the existing selection of this recipe, `features/all_in_one_ereno_train_combined.json`
  (24 features), unchanged (the fusion rule was selected on it).
- CICIDS2017, Morris-4, CICIoT2023: new runs, per type, on the splits of section 4.

### 2.3 Partitioning

- **One node per attack type** (N = 7, 26, 30, 30; Morris-4 amended). Node i holds all training rows of type i
  and the i-th of N disjoint random slices of the training benign rows (seed 42); ERENO
  benign training rows capped at 500,000 before slicing.
- Each node trains **one XGBoost specialist** on its partition, on the selected features
  (max_depth 4, eta 0.1, 10 rounds, scale_pos_weight = benign / attack).

## 3. Fusion

- **Fusion: OR at 0.5** (rule R0, `docs/FUSION_RULE.md`, tag `fusion-v1`): a sample is
  flagged when any specialist scores above 0.5. No threshold is calibrated.
- Seed 42 for the reported results; seeds 43-46 repeat the per-sample results to report
  their dispersion.

## 4. Training and test split

- **ERENO:** the release's training and test files.
- **CICIDS2017, improved:** *(corrected 2026-11-03, see the change log)* per label, its
  first 70 % of flows in time train and the rest test; benign likewise, day by day. Windows
  are built over the test flows in time order.
- **Morris-4** *(amended; Morris-3 had the same rule in file order)*: for each label (benign
  included), its first 70 % of samples in time train, the rest test.
- **CICIoT2023:** by file: the first 7 part files (name order) train, the last 3 test.

## 5. Node loss

N = number of attack types (one specialist per node). Nodes are removed in a fixed order (a
permutation drawn with seed 42), from N down to 3:

- **GL, retention:** every node keeps the specialists it received; the fused detector of a
  surviving node is unchanged.
- **FL, re-formed aggregate:** the aggregate holds only the specialists of the nodes still
  present.

Reported: overall and per-type recall, and false-positive rate, at every N.

## 6. Window alarms (ERENO, CICIDS2017, Morris-4)

- Per window: network-wide flagged count and flagged fraction, and per specialist S_t = the
  flagged samples on which the specialist of type t fired.
- Alarm: network-wide volume OR fraction OR any S_t >= T_t, with thresholds = ceil(1.25 x
  maximum over benign windows) + 1, and T_t >= 2.
- **Out-of-sample calibration, blocked-5** (the ERENO procedure, now for every windowed
  domain): the test period, in order, is cut into 5 contiguous blocks; each block is scored
  with thresholds calibrated on the attack-free windows of the other four; results are
  pooled over the five blocks.
- Windows: ERENO 1 s; CICIDS2017 10 s; Morris-4 100 s *(amended: Morris-4 has a clock; at
  about one sample per second over 76.6 h, 100 s keeps the order of magnitude of the
  100-sample window registered for Morris-3)*. Sensitivity reported for Morris-4 10 s and
  1,000 s and for CICIDS2017 1 s and 60 s; the primary values above are the reported
  results.
- CICIoT2023: no window (rows shuffled; at 2.4 % benign almost every window would hold an
  attack).
- **Profile ablation** *(amended, see the change log)*: CICIDS2017 is also evaluated under
  a profile without time or addresses: same specialists, same test flows in arrival order,
  windows of M consecutive flows, with M = the mean number of test flows per occupied 10-s
  window (computed on the test split), attribution none; same alarm rule and blocked-5
  calibration. Only the profile changes.

## 7. Metrics

Per sample: F1-score, recall, precision, false-positive rate (at the prevalence of the test
split), per-type recall; F1-score also at the ERENO test prevalence (6.8 %) for comparison
across domains; precision also at operational prevalences of 1 % and 0.1 % *(amended, see
the change log)*, derived from the measured recall R and false-positive rate FPR as
P(π) = Rπ / (Rπ + FPR (1 − π)); when no false positive occurs, FPR is replaced by its 95 %
upper bound, 3/n for n benign test samples. Windows: attack windows alarmed, benign windows alarmed (false alarms),
per type. Types with fewer than 100 test samples are reported with exact 95 % intervals.

## 8. Differences from the earlier runs (declared)

- One specialist per type instead of two per category; OR instead of k >= 2.
- CICIDS2017: per-label specialists (26) instead of 7 categories; "Attempted" flows are
  their own types instead of benign; features by GRASP per type on the new split (the
  initial TCP window sizes are candidates like any content column; no feature is removed
  on the basis of the exploratory probe); window thresholds by blocked-5 instead of Monday.
- Morris-4 (amended, replacing Morris-3) and CICIoT2023 are new domains; Electra and UNSW
  are no longer evaluated.

## Change log

- **2026-11-12, correction (implementation), section 2.2: no wall-clock limit on GRASP; the
  CICIDS2017 step (b) resumed from a checkpoint.** The GRASP implementation carried a
  wall-clock limit of 24 h (`max_time` in `python/grasp/base.py`), not stated in the
  protocol, checked at the end of each iteration and inside IWSS; the clock includes
  standby. It could end a run before the registered rule (5 iterations without improvement)
  was met. Of the 86 GRASP runs completed for evaluation-v1 (the 84 selections in use, the
  replaced Morris-4 `specific_19`, and one Morris-3 run superseded when Morris-4 replaced
  Morris-3), one reached it: the first Morris-4 `specific_19` (86,889 s, 13 iterations under
  the then-registered rule of 15), which establishes the cause left open in the entry of
  2026-11-07; that run had already been replaced by a re-run under the amended rule. The 84
  selections in use all stopped on the iteration rule (longest: CICIoT2023 step (b),
  77,719 s), so none is affected. The limit is
  removed for steps (a) and (b) (`--max-time-h 0` in `scripts/run_ev1_grasp.sh`; the default
  of the library is unchanged for other callers), which brings the implementation in line
  with the registered rule. Only the CICIDS2017 step (b), still running, runs under the
  correction. That step had been interrupted three times (a restart by Windows Update, a
  shutdown of the WSL virtual machine, and one of unrecorded cause), each time restarting from
  the first iteration. It is resumed from an evaluation checkpoint
  (`python/grasp/checkpoint.py`): GRASP runs again from the start with the same seed, and
  each evaluation already made is served from the checkpoint instead of recomputed, so that
  the search reaches the point of interruption with the same random-number state as an
  uninterrupted run; any difference in the requested subsets or in the hash of that state
  aborts the run. The checkpoint was seeded with the 2,276 evaluations of the longest
  interrupted run (`scripts/grasp_ckpt_from_log.py`); the four logged runs are identical
  over their common evaluations, and each F1-score is rebuilt exactly from the logged fold
  counts. On a reduced Morris-4 problem, a run killed half-way and resumed, and a run seeded
  from a truncated log, returned the same selection, evaluation sequence and random-number
  states as an uninterrupted run. No CICIDS2017 selection existed and no CICIDS2017
  evaluation model had been trained.

- **2026-11-10, amendment (addition), section 7: precision at operational prevalence.** The
  test splits are attack-rich (attack share 6.8 % in ERENO, 24.6 % in CICIDS2017, 19.4 % in
  Morris-4, 97.6 % in CICIoT2023), far above the prevalence of attacks in operation. Recall
  and the false-positive rate do not depend on prevalence; precision and F1-score do, and
  the F1-score at the ERENO prevalence already registered aligns the domains but not with
  operation. Precision is therefore also reported at prevalences of 1 % and 0.1 %, derived
  from the measured recall and false-positive rate; no run is added. When no false positive
  occurs, the 95 % upper bound of the false-positive rate (3/n) is used, so that precision
  is not reported as 1. Added before any evaluation run; no evaluation model had been
  trained and no test result had been seen.

- **2026-11-07, amendment (compute budget), section 2.2: GRASP stops after 5 iterations
  without improvement instead of 15, in steps (a) and (b) and in every domain.** Reason:
  compute time, not results. Run alone, a Morris-4 iteration takes about 0.7 min per type in
  step (a) and about 84 min in the multiclass step (b), so the registered rule (at least 16
  iterations per run) costs about 22 h for step (b) of Morris-4 alone, and CICIoT2023 and
  CICIDS2017 are larger. The change is not expected to alter the selection: in 29 of the 30
  Morris-4 per-type runs the best subset was found in the first iteration (`iterations` −
  `no_improvement` = 1 in `features/por_ataque/grasppen005_ev1_morris4_*.json`), so stopping
  after 5 iterations returns the same subset; in step (b) the best subset was also found in
  the first iteration, followed by 8 iterations without improvement. This evidence comes
  from the cross-validation on the training split only; no evaluation model had been trained
  and no test result had been seen. Consequences: the 29 Morris-4 per-type selections are
  kept as computed (identical under the amended rule); `specific_19`, whose output records 13
  iterations (fewer than the rule implies, cause not established; the run overlapped the
  parallel period and a standby), is re-run under the amended rule, and the Morris-4 superset
  is rebuilt; step (b) of Morris-4, interrupted after 9 iterations, is re-run under the
  amended rule; CICIoT2023 and CICIDS2017 run under it from the start. The ERENO selection
  is unchanged, as registered in section 2.2.

- **2026-11-03, amendment (addition), section 6: profile ablation on CICIDS2017.** The
  controlled ablation of the earlier runs (`scripts/_cicids_profile_ablation.py`, profile
  `it-flow-blind`) is kept: CICIDS2017 is re-run with count windows (M = mean test flows per
  occupied 10-s window) and no attribution, the specialists and test flows unchanged, so
  that only the profile changes. Added before any evaluation run; after the Morris-4
  amendment, CICIoT2023 is the only domain without time, and the ablation isolates the
  effect of the profile on the same traffic.

- **2026-11-03, amendment following the screening protocol's amendment (industrial control
  domain).** Morris-4 replaces Morris-3: its authors released it to correct the unintended
  patterns they report in the 2014 releases (see `docs/DATASET_PROTOCOL.md`, change log).
  Consequences, fixed before any model of the evaluation was trained: 30 specific attacks
  (N = 30); Morris-4 has a clock, so its split is per label in time and its window is 100 s
  (about 100 samples, as the window registered for Morris-3); sensitivity at 10 s and
  1,000 s. The partial GRASP run on Morris-3 (one attack type) is discarded.

- **2026-11-03, error correction, section 4 (CICIDS2017 split).** The registered rule ("on
  each attack day, the flows from the 70 % time quantile of each label to its last flow are
  test, all labels included", "as in the earlier runs") left attack types without training
  rows once specialists are per type: the test interval of small labels spread over a day
  (the "Attempted" ones) covered other attacks whole (DDoS 0 training rows, PortScan 344 of
  159,066). Computing the intervals per attack category, as the earlier runs did, was tried
  first and failed the same way (DDoS and DoS GoldenEye with 0 training rows), because one
  specialist per category had hidden it. The rule is replaced by the one already registered
  for Morris-3: per label, its first 70 % in time train, the rest test; benign likewise, day
  by day. Every type now has training rows (`scripts/build_ev1_datasets.py`). No model was
  trained and no result was seen before the correction.
