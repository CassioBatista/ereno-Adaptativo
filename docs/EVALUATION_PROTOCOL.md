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
| industrial control | Morris-3 (water tank) | 6 (CMRI excluded by Q2a) | file order (no clock) | M = 100 rows |
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
| Morris-3 | `water_final.arff` (Web Archive copy, checksums in `results/morris_ics_md5sums.txt`) | `result` (6 qualified attack categories + normal) | Modbus and process fields | `time`, `crc_rate` |
| CICIoT2023 | 10 part files of the CSV release | the 30 qualified labels | 46 flow features | none (the release has no identifier column) |

### 2.2 Feature selection: GRASP, same recipe in every domain

The recipe of the pipeline (`scripts/run_grasp_cross_domain.sh`), run on the **training split
only**: (a) binary GRASP per attack type, attack vs benign (GraspVND, XGBoost, 5-fold
cross-validation, lambda = 0.05, sample 60,000, benign cap 60,000, attack rows capped at the
benign count, stop after 15 iterations without improvement); (b) penalized multiclass
GRASP over all types (lambda = 0.05, sample 150,000, stop after 15 iterations without
improvement; on a class-balanced copy when benign is under 10 % of the training rows,
i.e. CICIoT2023); (c) the selected set is the union of (a) and (b). Types with fewer than
10 training rows get no per-type run and are covered by (b).

- ERENO: the existing selection of this recipe, `features/all_in_one_ereno_train_combined.json`
  (24 features), unchanged (the fusion rule was selected on it).
- CICIDS2017, Morris-3, CICIoT2023: new runs, per type, on the splits of section 4.

### 2.3 Partitioning

- **One node per attack type** (N = 7, 26, 6, 30). Node i holds all training rows of type i
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
- **CICIDS2017, improved:** as in the earlier runs: on each attack day, the flows from the
  70 % time quantile of each label to its last flow are test, all labels included; Monday
  after its 70 % time quantile is test; the rest is training.
- **Morris-3:** file order; for each label (benign included), its first 70 % of rows train,
  the rest test.
- **CICIoT2023:** by file: the first 7 part files (name order) train, the last 3 test.

## 5. Node loss

N = number of attack types (one specialist per node). Nodes are removed in a fixed order (a
permutation drawn with seed 42), from N down to 3:

- **GL, retention:** every node keeps the specialists it received; the fused detector of a
  surviving node is unchanged.
- **FL, re-formed aggregate:** the aggregate holds only the specialists of the nodes still
  present.

Reported: overall and per-type recall, and false-positive rate, at every N.

## 6. Window alarms (ERENO, CICIDS2017, Morris-3)

- Per window: network-wide flagged count and flagged fraction, and per specialist S_t = the
  flagged samples on which the specialist of type t fired.
- Alarm: network-wide volume OR fraction OR any S_t >= T_t, with thresholds = ceil(1.25 x
  maximum over benign windows) + 1, and T_t >= 2.
- **Out-of-sample calibration, blocked-5** (the ERENO procedure, now for every windowed
  domain): the test period, in order, is cut into 5 contiguous blocks; each block is scored
  with thresholds calibrated on the attack-free windows of the other four; results are
  pooled over the five blocks.
- Windows: ERENO 1 s; CICIDS2017 10 s; Morris-3 M = 100 consecutive rows in file order
  (no clock: the order of the file, declared as such). Sensitivity reported for Morris-3
  M = 10 and 1,000 and for CICIDS2017 1 s and 60 s; the primary values above are the
  reported results.
- CICIoT2023: no window (rows shuffled; at 2.4 % benign almost every window would hold an
  attack).

## 7. Metrics

Per sample: F1-score, recall, precision, false-positive rate (at the prevalence of the test
split), per-type recall; F1-score also at the ERENO test prevalence (6.8 %) for comparison
across domains. Windows: attack windows alarmed, benign windows alarmed (false alarms),
per type. Types with fewer than 100 test samples are reported with exact 95 % intervals.

## 8. Differences from the earlier runs (declared)

- One specialist per type instead of two per category; OR instead of k >= 2.
- CICIDS2017: per-label specialists (26) instead of 7 categories; "Attempted" flows are
  their own types instead of benign; features by GRASP per type on the new split (the
  initial TCP window sizes are candidates like any content column; no feature is removed
  on the basis of the exploratory probe); window thresholds by blocked-5 instead of Monday.
- Morris-3 and CICIoT2023 are new domains; Electra and UNSW are no longer evaluated.

## Change log

- (none)
