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
| industrial control | Morris-4 (gas pipeline, 2015) *(amended, see the change log; was Morris-3)* | 35 specific attacks (none excluded; Q2d applied before Q2a) *(amended 2026-11-17, see the change log; was 34, 1 excluded by Q2a; before 2026-11-16, 30, 5 excluded by Q2a)* | time | 100 s |
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
| Morris-4 *(amended)* | `IanArffDataset.arff` (Web Archive copy, checksums in `results/morris_ics_md5sums.txt`) | `specific result` (35 qualified attacks + normal; the responses of 18 attack types excluded by Q2d, see below) | Modbus and process fields, `crc rate` included *(amended 2026-11-17: attack 18, "Bad CRC", manipulates it)* | `time` *(amended 2026-11-17; was `time`, `crc rate`)* |
| CICIoT2023 | 10 part files of the CSV release | the 30 qualified labels | 46 flow features | none (the release has no identifier column) |

- **Morris-4, Q2d before Q2a** *(amended 2026-11-16, see the change log)*: the release labels
  a whole command/response transaction, so the controller's response to an injected command
  carries the attack label. For each attack type, its responses (`command response` = 0) are
  excluded when every one of them has a feature vector that also occurs as benign (the identity
  test of Q2a); a type with any response not found among benign rows (forged responses) keeps
  them all. Q2a is then applied to the rows that remain. Implemented in
  `scripts/build_ev1_datasets.py` (`q2d_transaction_responses`, `q2a_after_q2d`), before the
  split of section 4.

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
2026-11-12)*. An iteration counts as an improvement only when its result beats the best
subset found so far, which is the subset the run returns *(correction (implementation), see
the change log of 2026-11-12, GRASP incumbent)*: the best-subset retention rule. The rule
has no minimum margin: a subset that beats the best one by any amount is an improvement and
resets the count, so a run can be lengthened by differences smaller than the spread of its
own cross-validation folds. Such a difference does not show that one subset is better than
the other; the returned subset is used as returned, and a feature that displaced another by
such a difference is not read as more relevant than the one it displaced *(clarification,
see the change log of 2026-11-18)*. Each run
of (a) is seeded on its own: Python's `random` with the GRASP seed at its start, and its
benign and attack row draws with a generator seeded by (GRASP seed, type), so that its result
depends neither on the process nor on the other types *(correction (implementation), see the
change log of 2026-11-14)*.

- All four domains: runs of this recipe under the best-subset retention rule and the per-type
  seeding, every run of (a) from the start, (b) resumed from the evaluations of its earlier
  run (`scripts/run_ev1_grasp_bsr.sh`, outputs tagged `bsr005`) *(amended, see the change log
  of 2026-11-14)*. For ERENO this replaces the selection published with the earlier work
  (`features/all_in_one_ereno_train_combined.json`, 24 features, which stays as published
  there) *(amended, see the change log of 2026-11-13)*. Morris-4: the same recipe on the data
  rebuilt with Q2d before Q2a and `crc rate` as content (section 2.1), every run from the
  start, (b) included (outputs tagged `bsr005crc`) *(amended 2026-11-17, see the change log;
  was `bsr005q2d`, stopped in step (b) and not used)*.

### 2.3 Partitioning

- **N = 14 nodes in every domain** *(amended, see the change log of 2026-11-13; was one node
  per attack type, N = 7, 26, 30, 30)*, the clients of the Disaster-FD run of section 5;
  monitor NN of that run is colocated with node NN.
- **Specialists:** one per attack type; the types with fewer than 100 training rows share
  one specialist, trained on their rows together (CICIDS2017: 10 types, 220 rows;
  CICIoT2023: 2 types, 117 rows). Specialists: ERENO 14 (each of the 7 types on two nodes,
  its rows split into two disjoint random halves, seed 42), CICIDS2017 17, Morris-4 35
  *(amended 2026-11-17; was 34; before 2026-11-16, 30)*, CICIoT2023 29.
- **Placement:** specialists in decreasing order of training rows (ties: order of the class
  declaration), in contiguous blocks: with S specialists, q = S div 14 and r = S mod 14,
  nodes 1 to 14 − r take q consecutive specialists and the last r nodes take q + 1, so that
  large specialists share a node only with large ones and small ones only with small ones
  (CICIDS2017: nodes 1-11 one each, 12-14 two each; Morris-4: 1-7 two, 8-14 three *(amended 2026-11-17; was 1-8 two, 9-14 three; before 2026-11-16, 1-12 two, 13-14 three)*;
  CICIoT2023: 1-13 two, 14 three). With fewer specialists than nodes, the extra nodes go to
  the largest specialists, one at a time (the `attack` partitioner of `fd/dataset.py`). Listing: `scripts/ev1_partition_n14.py`,
  `results/ev1_partition_n14.txt`.
- Node i holds the i-th of 14 disjoint random slices of the training benign rows (seed 42);
  ERENO benign training rows capped at 500,000 before slicing. Every specialist on node i
  is **one XGBoost** trained on its own attack rows against that slice, on the selected
  features (max_depth 4, eta 0.1, 10 rounds, scale_pos_weight = benign / attack).
- **References (step 1)** *(amended, see the change log of 2026-11-13)*, on the same
  training rows and features, fused by the rule of section 3 where there is more than one
  model: (i) *Local*: each node with its own specialists only; (ii) *IID*: the training rows
  split into 14 equal random parts (`iid` partitioner, seed 42), one binary XGBoost per node
  with round(10 S / 14) rounds, S = number of specialists (ERENO 10, CICIDS2017 12,
  Morris-4 25 *(amended 2026-11-17; was 24; before 2026-11-16, 21)*, CICIoT2023 21);
  (iii) *Centralized*: one binary XGBoost of 10 S rounds (140, 170, 350 *(amended
  2026-11-17; was 340; before 2026-11-16, 300)*, 290) on the pooled training rows of the 14 nodes, scale_pos_weight =
  benign / attack, an upper reference, not a competitor.

## 3. Fusion

- **Fusion: OR at 0.5** (rule R0, `docs/FUSION_RULE.md`, tag `fusion-v1`): a sample is
  flagged when any specialist scores at least 0.5 *(corrected from "above 0.5", see the change
  log of 2026-11-12)*. No threshold is calibrated.
- Seed 42 for the reported results; seeds 43-46 repeat the per-sample results to report
  their dispersion.
- **Secondary, k = 2** *(added 2026-11-17, see the change log)*: a sample is flagged when at
  least two specialists score at least 0.5 (rule R1 of `docs/FUSION_RULE.md`), on the same
  scores as OR, with no model trained for it. It is reported in the appendix, in the four
  domains, with the per-sample metrics of section 7 and per-type recall, under the
  attack-based partition with all nodes present. It is not used for any decision and is not
  applied to node loss, the trust-level replay or the windows.

## 4. Training and test split

- **ERENO:** the release's training and test files.
- **CICIDS2017, improved:** *(corrected 2026-11-03, see the change log)* per label, its
  first 70 % of flows in time train and the rest test; benign likewise, day by day. Windows
  are built over the test flows in time order.
- **Morris-4** *(amended; Morris-3 had the same rule in file order)*: for each label (benign
  included), its first 70 % of samples in time train, the rest test.
- **CICIoT2023:** by file: the first 7 part files (name order) train, the last 3 test.

## 5. Node loss

N = 14 *(amended, see the change log of 2026-11-13; was the number of attack types)*; a node
removed takes its specialists with it. Nodes are removed in a fixed order (a permutation
drawn with seed 42), from N down to 3:

- **GL, retention:** every node keeps the specialists it received; the fused detector of a
  surviving node is unchanged.
- **FL, re-formed aggregate:** the aggregate holds only the specialists of the nodes still
  present.

Reported: overall and per-type recall, and false-positive rate, at every N.

## 6. Window alarms (ERENO, CICIDS2017, Morris-4)

- Per window: network-wide flagged count and flagged fraction, and per specialist S_s = the
  flagged samples on which specialist s fired *(amended, see the change log of 2026-11-13;
  was "the specialist of type t": a specialist may now be shared by small types, and an
  ERENO type has two)*.
- Alarm: network-wide volume OR fraction OR any S_s >= T_s, with thresholds = ceil(1.25 x
  maximum over benign windows) + 1, and T_s >= 2.
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

**Morris-4, reporting by category** *(added 2026-11-17, see the change log)*:

- **Grouping.** Per-type recall is reported grouped by the attack category of the release
  (`categorized result`): NMRI (1), CMRI (2), MSCI (3), MPCI (4), MFCI (5), DoS (6) and
  Recon (7) *(amended 2026-11-17: DoS, specific 18, is kept once `crc rate` is content)*.
- **Benign-identical share.** Each type is reported with the fraction of its test rows
  whose vector of the 16 candidate features also occurs among the benign rows of the
  training or test split. A per-sample detector cannot flag these rows without also flagging
  the benign rows identical to them.
  - The share is computed from the data alone, with no model, by the identity test of Q2a.
  - It extends to every type the bound already stated for specific 13, 14 and 15 (change log
    of 2026-11-16).
- **Not a detection metric: the multiclass GRASP score.** The score of step (b) of section
  2.2 counts an attack assigned to another attack type as a false positive
  (`python/evaluation.py`). Several Morris-4 types share their feature vectors with other
  types, so this score is not reported as a measure of detection; detection under OR only
  needs some specialist to flag the sample.

## 8. Differences from the earlier runs (declared)

- One specialist per type instead of two per category (ERENO: two per type, on 14 nodes;
  small types share one; amended 2026-11-13); OR instead of k >= 2.
- CICIDS2017: per-label specialists (26) instead of 7 categories; "Attempted" flows are
  their own types instead of benign; features by GRASP per type on the new split (the
  initial TCP window sizes are candidates like any content column; no feature is removed
  on the basis of the exploratory probe); window thresholds by blocked-5 instead of Monday.
- Morris-4 (amended, replacing Morris-3) and CICIoT2023 are new domains; Electra and UNSW
  are no longer evaluated.

## 9. Trust-level replay with detection *(added 2026-11-17, see the change log)*

- **Traces.**
  - **T1:** Disaster-FD run 1 (`20261008-154706`, `scenario.md`), 83 min: local 0-15,
    federated 15-30, local 30-45, gossip 45-60, federated 60-83.
  - **T2:** the Disaster-FD run of `scenario2.md`, 98 min, requested from the Disaster-FD
    team: local 0-15, federated 15-30, aggregator lost at 30, clients {4, 5} lost at 45 and
    {2, 3, 6, 7} at 60, aggregator restored at 75. The lost clients stay unreachable to the
    end.
  - If the T2 run is not delivered, T2 is replayed from the trust levels the scenario
    expects: its phase table, with the link states following the lost clients and the
    aggregator, and no transient flips. It is then reported and labelled as synthetic.
- **Mode of each node.** Monitor NN drives node NN (section 2.3). The mode is derived from
  the trust level of each 5-s probe cycle by `docs/TRUST_LEVEL.md` (T_GL = 30, T_FL = 50,
  S = 5 s, F = 15 s). Whether a link is up in a cycle is read from the per-target rows of the
  telemetry (`dev_id`, `is_timeout`), not from the scenario (except in the synthetic T2).
- **Unit of knowledge: the block of a node**, i.e. its specialists (section 2.3).
  - **Schedule**, the same in every domain and in both traces: the blocks of nodes
    {1, 2, 5, 6, 7, 8, 13, 14} exist at 0 min, those of {3, 4, 11, 12} arrive at 30 min and
    those of {9, 10} at 60 min. That is 8, 12 and 14 nodes, the proportions 4/7, 6/7 and 7/7
    of the scenarios.
  - A block arrives at its own node.
  - **ERENO**, whose types sit on two adjacent nodes: 4 types at 0 min (random replay, high
    StNum, inverse replay, masquerade as fault), injection and masquerade as normal at
    30 min, high-rate poisoning at 60 min.
  - **Choice of nodes**, by their position in the T2 ring, before any run:
    - 3 and 4 inside the region lost in T2 (4 at 45 min, 3 at 60);
    - 11 and 12 outside it;
    - 9 and 10 in the part of the ring still in gossip at 60 min.

    The same sets are used in T1.
- **Learning rounds** of period **Δ** in both learning modes, counted from `run_t0`.
  **Δ = 60 s**; sensitivity at 5, 15, 60, 180, 300 and 600 s. The values at 60 s are the
  reported results.
  - **Gossip round:** two ring neighbours exchange the blocks they hold when both are in
    gossip and each observes the other as up in that cycle; one hop per round.
  - **Federated round:** the nodes in federated mode that observe the aggregator as up take
    part.
    - *Retention:* the aggregate is the union of the blocks held by the participants, and
      each participant keeps what it holds plus the aggregate.
    - *Re-formed:* the aggregate is the participants' own blocks, and each participant holds
      the aggregate only.
  - **Local:** no exchange. An unreachable node keeps what it holds and keeps detecting: the
    traces model a communication outage, not a crash.
- **Policies compared on the same trace:**
  1. local only;
  2. gossip fixed, where every node exchanges with each ring neighbour whose link is up,
     whatever its trust level;
  3. federated fixed, retention;
  4. federated fixed, re-formed;
  5. switching on the trust level, retention (ReSIDS);
  6. switching on the trust level, re-formed.

  The fixed policies ignore the trust level only for the choice of mode. They still need
  the link: a federated round needs the aggregator, and a gossip exchange needs the
  neighbour.
- **Detector of node i at time t:** the specialists it holds at t, fused by OR at 0.5
  (section 3). These are the specialists of section 2.3 (seed 42) on the features of
  section 2.2, never retrained.
- **Traffic: one stream per domain and trace**, recomposed by phase from the test split
  (the method of `docs/disaster_fd_trace.md`, generalised).
  - **Content:**
    - only the types of blocks that have already arrived;
    - no row is used twice;
    - the attack rows of a type are spread over the phases in which it is present, in
      proportion to their durations;
    - they are taken in contiguous chunks in their original order (ERENO, CICIDS2017,
      Morris-4) or drawn at random (CICIoT2023);
    - rows are placed at uniform random times within a phase, each chunk keeping its order;
    - seed 42; seeds 43-46 for the dispersion of the recomposition.
  - **Domains with windows (ERENO, CICIDS2017, Morris-4):**
    - Every benign test row is used. Benign rows per trace minute are b = B / (T + 10), where
      B is the number of benign test rows, T the trace length in minutes, and 10 min a benign
      calibration block outside the trace.
    - Attack rows are added up to a share of 6.8 % of a phase (the ERENO test prevalence,
      section 7). A phase whose types do not reach that share keeps fewer attack rows; the
      generator reports the share of each phase.
    - Windows are M consecutive samples, where M is the mean number of test samples per
      occupied window of the domain's registered length (section 6), computed on the test
      split. These are count windows, as in the profile ablation, because the time axis is
      synthetic.
    - The thresholds follow the rule of section 6, calibrated on the calibration block.
  - **CICIoT2023 (no windows):** the stream holds every test row of the present types, with
    no cap on the attack share. Precision and F1-score are reported at 6.8 % from the
    recall and false-positive rate (section 7): with 16,665 benign test rows, a stream at
    6.8 % would hold about 1,200 attack rows for 30 types.
  - **Declared simplification:** every node scores the same stream; no node sees traffic of
    its own.

  Resulting sizes:

  | Domain | B (benign test rows) | b, T1 (83 min) | b, T2 (98 min) | Attack rows / min at 6.8 %, T2 |
  |---|---:|---:|---:|---:|
  | ERENO | 2,755,139 | 29,625 | 25,510 | ≤ 1,861 |
  | CICIDS2017 | 474,772 | 5,105 | 4,396 | ≤ 321 |
  | Morris-4 | 64,374 | 692 | 596 | ≤ 43 |
  | CICIoT2023 | 16,665 | 201 | 170 | all rows of the present types |

- **Reported:**
  - **Diffusion:** for each block, the number of nodes holding it over time. Its latency is
    the time from its arrival until every node reachable from its own node holds it.
  - **Detection, per policy, phase and node:**
    - recall;
    - recall on the types of the blocks that arrived at 30 and 60 min;
    - false-positive rate;
    - precision and F1-score at 6.8 %;
    - in the windowed domains, the share of attack windows alarmed and the false alarms.
  - **Network level:** the mean over nodes, reported separately for reachable and
    unreachable nodes.
  - **Sensitivity:** the values of Δ.

## Change log

- **2026-11-18, step-1 script; the design is kept after a smoke test on training rows.**
  - **Script.** `scripts/ev1_step1.py` implements step 1 as registered in sections 2.3, 3, 4
    and 7: specialists, IID, Centralized and Local; OR, with k = 2 on the same scores; the
    per-sample metrics of section 7. The ERENO partition reproduces the `attack` partitioner
    of `fd/dataset.py` draw for draw: benign cap, benign slices, class shuffles.
  - **Missing values.** The Morris-4 features contain NaN (fields absent from a message
    type). They are passed to XGBoost as missing values, as in the GRASP runs.
  - **Smoke test.** Run on Morris-4 with the training file only: 30 % of the training rows
    (random, seed 42) stand in for the test split. The test file was not opened. This is not
    the temporal split and not a result.

    | Detector | Recall | FPR |
    |---|---|---|
    | OR over the 35 specialists | 99.9 % | 41.0 % |
    | IID, OR | 99.8 % | 40.9 % |
    | Centralized | 86.5 % | 3.9 % |

    The false positives concentrate on the nodes that hold the weak types (30, 35, 13, 15, 2,
    9, 10), with a local FPR of 13 % to 19 %.
  - **Decision (2026-11-18).** The design is unchanged: `scale_pos_weight`, threshold 0.5,
    OR. Recorded because any change made after this point would be informed by these
    numbers. None was made.
  - **Status.** Written before step 1 read any test split.

- **2026-11-18, clarification, section 2.2: marginal improvements under the best-subset
  retention rule; no change to the procedure.**
  - **Observed.** In the Morris-4 `bsr005crc` run, step (b), iteration 6 (the fifth after
    the last improvement), evaluation 837 replaces F3 with F11 in the 13-feature subset of
    evaluation 834. Same size, so the penalty is equal. Cross-validated F1 77.218 against
    77.206 (+0.012 pp).
  - **Size of the difference.** The folds of evaluation 834 range from 76.69 to 77.50. On the
    same folds, the differences between the two subsets range from −0.31 to +0.24 pp.
  - **Under the rule.** This is an improvement: the count resets, and the run makes at least
    five more iterations (about 1.7 h each).
  - **Not changed.** A minimum margin is not added. It would be decided during the run,
    after this evaluation was seen, and it would change the subset the run returns (that of
    evaluation 834). The registered rule is kept.
  - **Reporting.** The returned subset is used as returned. The choice between features that
    differ only within the fold spread is not interpreted as relevance.
  - **Other runs (checked from the logs, `results/grasp_bsr005.log` and
    `results/grasp_bsr005crc.log`).** The rule was replayed over the logged evaluations. An
    improvement is counted as marginal when it raises the penalized F1 by less than 0.1 pp
    without reducing the number of features. Gains from dropping features at equal F1 are the
    intended effect of λ and are not counted.

    | Domain | Improvements | Marginal | Returned subset from a marginal improvement |
    |---|---|---|---|
    | ERENO | 10 | 2 | global (+0.020 pp, 4 of 10 features swapped) |
    | CICIDS2017 | 30 | 6 | DoS Slowhttptest (+0.042), Infiltration Portscan (+0.013) |
    | CICIoT2023 | 40 | 6 | Recon HostDiscovery (+0.039) |
    | Morris-4 (`bsr005crc`, per type) | 36 | 5 | specific 18, 28, 29, 35 |

    Where the folds are in the log, every one of these is smaller than the fold range of the
    subset it replaced or of the new subset. The ERENO global step was resumed by replay, so
    its folds are not in the log. In Morris-4, `crc rate` entered the subsets of specific 29 and 35 through a
    marginal improvement, and those of 26 and 28 with gains of 0.11 pp. Its presence in those
    types is not read as relevance.
  - **Status.** Written while the run was in progress, before it returned its subset.

- **2026-11-17, amendment, sections 1, 2.1, 2.2, 2.3 and 7: in Morris-4, `crc rate` is a
  content feature, not an identifier.**
  - **The inconsistency.** Section 2.1 defines identifiers as fields that locate a row or
    name an endpoint *and that no attack manipulates*; that is why ERENO keeps `SqNum` and
    `StNum`. In Morris-4, `crc rate` was removed as an identifier (the screening loader drops
    every column whose name contains "crc"). The dataset paper (Morris, Thornton and
    Turnipseed 2015, Table 2) describes attack 18, "Bad CRC" (DoS), which sends Modbus
    packets with incorrect CRC values, so the field is manipulated by an attack and is
    content under the registered definition. Found while reading that paper, before any
    evaluation model.
  - **What the field is.** Checked on the raw release, labels and this column only:
    - benign rows take 256 distinct values (12,336 to 17,990), the same set for commands and
      responses;
    - the field is not a clock: its Spearman correlation with time is −0.005 in benign
      rows, and its mean is flat over time deciles;
    - no attack category other than DoS has a value outside the benign set (0.0 %);
    - the commands of specific 18 have values outside it in 99.6 % of rows (21 to 65,492),
      and its responses in 0 %.

    The field therefore carries the signal of the one attack that manipulates it and no
    time or tool artefact visible as out-of-set values.
  - **Effect.**
    - With `crc rate` (16 candidate features), Q2d still excludes the responses of the same
      18 types (15,244 rows), specific 18 included: its 1,088 responses are identical to
      benign.
    - Q2a then excludes no type: the commands of specific 18 are 0.0 % identical to benign,
      against 62.1 % without the field.
    - Morris-4 has 35 attack types (259,384 rows; train 181,553, test 77,831), hence 35
      specialists (nodes 1-7 two, 8-14 three; `results/ev1_partition_n14.txt` regenerated),
      IID 25 rounds per node and Centralized 350.
    - The other types keep their rows, so their split is unchanged.
    - The DoS category enters the reporting of section 7.
  - **Code.** Implemented in `scripts/build_ev1_datasets.py` (`build_morris4` moves `crc
    rate` from the identifiers to the features, as F16, so F1-F15 keep their meaning). The
    screening loader (`scripts/audit_loaders.py`, protocol-v1) is not changed; it stays as
    the record of the screening.
  - **GRASP.** The Morris-4 GRASP is re-run on the rebuilt data from the start
    (`TAG=bsr005crc`). The `bsr005q2d` run, stopped in its global step after 554
    evaluations, is kept and not used, as is `bsr005`. The earlier Morris-4 files are kept
    outside the repository (`~/.trash/ereno-Adaptativo/2026-11-17/`).
  - **Diagnostics of the entry below.** `results/morris4_q2d_*` were computed on the 15
    features and 34 types; they are recomputed on the rebuilt data, and the numbers of that
    entry are replaced by the recomputed ones in a later entry.
  - **Other domains.** Not affected: ERENO, CICIDS2017 and CICIoT2023 have no checksum
    column among their identifiers. Whether an attack in those domains manipulates one of
    their identifiers is not re-examined here.
  - **Status.** No evaluation model had been trained and no test result had been seen.

- **2026-11-17, amendment (addition), section 7: Morris-4 reported by attack category, with
  the benign-identical share of each type.**
  - **What prompted it.** A diagnostic on the Morris-4 training split, made before any
    evaluation model, the test file not read (`results/morris4_q2d_training_diagnostic.txt`).
    The per-type GRASP runs (`bsr005q2d`, cross-validated F1-score on the training split)
    are below 90 in 10 of the 34 types: specific 30 (NMRI, 22.0), 35 (CMRI, 27.1), 10
    (MPCI, 68.0), 33 (CMRI, 68.9), 34 (CMRI, 76.8), 13 (MSCI, 78.6), 15 (MSCI, 79.4), 2
    (MPCI, 83.8), 14 (MSCI, 85.0), 6 (MPCI, 88.6). The other 24 types are between 92.0
    and 100.
  - **The low values are not a labelling artefact.** In 30, 35, 33, 34 and in the three
    MPCI types, at most 5.3 % of the training rows have a feature vector that also occurs
    as benign (0 % in the MPCI types), so Q2a and Q2d do not apply.
  - **They are consistent with the purpose of the release.** The report of the earlier
    releases (`MSU_SCADA_Final_Report.pdf`, pp. 10-14) lists the patterns that gave the
    attacks away there, among them CMRI measurements always in one narrow range and NMRI
    measurements always grossly out of bounds. The Morris-4 release was made to remove such
    patterns (`docs/DATASET_PROTOCOL.md`, change log): a forged measurement inside the
    normal range is not separable from a genuine one sample by sample.
  - **The MSCI bound was already registered.** In 13, 14 and 15, 27.3 %, 22.0 % and 11.9 % of
    the training rows are identical to benign (the bound of the entry of 2026-11-16).
  - **Why the multiclass score is excluded.** The Morris-4 multiclass GRASP run (best
    penalized score 73.4 so far) is low for a reason unrelated to detection. In 5 types (19,
    21, 22, 31, 32) every training feature vector also occurs in another attack type, and in
    4 more (25, 26, 28, 29) between 65 % and 75 % do, and the score counts the resulting
    confusion between attack types as false positives (section 7).
  - **Reporting rule.** Fixed now so that the Morris-4 results are reported against
    properties of the data established before the test, not explained after it:
    - per-type recall grouped by category;
    - the benign-identical share of each type, computed on the test rows from the data alone;
    - the multiclass GRASP score not used as a measure of detection.
  - **Scope.** Nothing in the selection, the specialists, the fusion or the metrics changes.
  - **Status.** No evaluation model had been trained and no test result had been seen.

- **2026-11-17, amendment (addition), section 3: k = 2 reported as a secondary result.** The
  fusion rule (OR at 0.5) was selected on ERENO only (`docs/FUSION_RULE.md`), the one domain
  with two specialists per attack type. In CICIDS2017, Morris-4 and CICIoT2023 each type has
  exactly one specialist, so under k = 2 a sample of a type is flagged only when a
  specialist of another type also fires on it. The argument for OR in those domains is
  therefore structural, not measured. k = 2 is added as a secondary result, in the four
  domains, so that this is measured. It uses the scores already computed for OR and needs
  no training. It does not change the rule evaluated (section 3 is otherwise unchanged) and
  is not used for any decision. No evaluation model had been trained and no test result had
  been seen.

- **2026-11-17, amendment (addition), section 9: trust-level replay with detection,
  specified before any run.** This carries out the entry of 2026-11-13, which left the replay
  of the trust levels with detection, and the dissemination of a specialist while the server
  is unavailable, to a later entry made before they are run.
  - **Two traces.**
    - T1, Disaster-FD run 1, is the dissemination scenario.
    - T2, `scenario2.md`, is the node-loss scenario. The aggregator is lost, then a
      contiguous region of six clients is lost in two steps, and the aggregator returns
      without them.
    - In T2 the static sweep of section 5 (nodes removed from 14 to 3) becomes a dynamic
      one, with the losses driven by the failure detector. Section 5 is unchanged.
    - T2 has been requested from the Disaster-FD team and not yet run. If it is not
      delivered, it is replayed from the trust levels the scenario expects and labelled as
      synthetic. The expected trust levels fix which nodes are reachable and when, which is
      what the node-loss comparison depends on; a real run adds transient flips and probe
      latency.
  - **Unit of knowledge.** The unit is the block of a node, not the attack type, so that the
    rule is the same in the four domains. Their numbers of specialists differ: ERENO 14,
    CICIDS2017 17, Morris-4 35, CICIoT2023 29. The 4/+2/+1 types of the scenarios become
    8/+4/+2 blocks.
  - **Choice of nodes.** The nodes whose blocks arrive late were chosen by their position in
    the T2 ring, as stated in section 9. With them, knowledge created inside the region
    lost later, and outside it, must leave or reach that region before the losses.
  - **Policies.** The six policies isolate three effects:
    - learning against no learning (policy 1);
    - the mode (policies 2 and 3);
    - the aggregation rule under loss (policies 3 against 4, and 5 against 6).
  - **Round period.**
    - Neither the implementation nor the traces fix a learning round in wall-clock time.
      The 5-s cycle of the traces is the failure detector's probe cycle, not a learning
      round, so Δ = 60 s is taken as the primary value and the sensitivity runs from 5 s to
      600 s.
    - With retention, the T2 losses cost knowledge only when Δ exceeds about 7.5 min: two
      gossip rounds between 30 and 45 min take the block of node 4 to node 2, and node 2
      stays in gossip with node 1 until 60 min.
    - The sensitivity therefore includes a value, 600 s, at which knowledge can be trapped
      in the lost region.
  - **Expected by construction, reported as measured.** In T2, between 60 and 75 min, nodes
    1 and 8 have a trust level of 15.
    - Under switching (policy 5) they are in local mode, although each still has one live
      link, to node 14 and to node 9.
    - Under gossip fixed (policy 2), the blocks of nodes 9 and 10 reach node 8 within one
      round and node 1 within five.
    - Under switching, they reach them only at 75 min, by federated learning.

    This is the cost of the cutoff T_GL = 30, which requires both ring neighbours.
  - **Differences from the ERENO trace of the earlier work** (`docs/disaster_fd_trace.md`,
    kept as it is):
    - the specialists and features of this evaluation instead of the 24-feature
      specialists;
    - benign rows from the test split only, instead of also the training rows beyond the
      500,000 cap, so that one rule holds in the four domains (the other domains have no
      unused training rows);
    - therefore a benign rate 0.77 (T2) and 0.89 (T1) of the original;
    - count windows instead of 1-s windows on the synthetic time axis;
    - the late types fixed by the node rule. They were an open choice there, and differ:
      injection and masquerade as normal at 30 min, high-rate poisoning at 60.
  - **What was read.** The test split was read for its label counts only, to size the
    streams; no model has scored it.
  - **Status.** No evaluation model had been trained and no test result had been seen.

- **2026-11-16, amendment, sections 1, 2.1, 2.2 and 2.3: Morris-4 rebuilt with Q2d before
  Q2a.** Found in the analysis of the Morris-4 selection (`bsr005`), after the GRASP runs and
  before any evaluation model: in 13 of its 30 types (specific 1, 3-12, 16, 17) the rows are
  half commands and half responses of the controller, and every response has a feature vector
  that also occurs as benign, while no command does. The release labels the whole
  command/response transaction: the response to an injected command carries the attack label
  although it is an ordinary response. Any per-sample detector is then bounded at a recall of
  50 % on these types, an F1-score of 66.7 at a precision of 1, which is where the per-type GRASP
  runs stopped (66.3 to 66.7); an XGBoost on all 15 features reached the same values (the GRASP
  subsets equal or exceed it on the 11 types checked), so the bound is in the labels, not in
  the search. The same mechanism excluded 5 types under Q2a (specific 2, 13, 14, 15, 18, with
  50.1 % to 81.1 % of rows identical to benign, all their responses included); specific 2 was
  excluded at 50.07 %, against the bound of 50 %. Amendment: Q2d (label scope: samples
  labelled by a rule regardless of their content are excluded when they can be identified) is
  applied to these responses, before Q2a, as stated in section 2.1. Result: the responses of
  18 types are excluded (15,244 rows; every other type has at most 4.6 % of its rows identical
  to benign, its forged responses kept); Q2a on the rows that remain excludes specific 18 only
  (62.1 %), and specific 2, 13, 14 and 15 are kept (0.1 %, 26.6 %, 23.2 %, 16.4 %). Morris-4
  has 34 attack types (258,296 rows; train 180,792, test 77,504 by the split of section 4);
  34 specialists, placed as in section 2.3 (nodes 1-8 two, 9-14 three; listing updated);
  IID 24 rounds per node, Centralized 340. Consequences: (1) the Morris-4 GRASP selection
  (`bsr005`) is replaced by a run of the recipe on the rebuilt data, every step from the start
  (`TAG=bsr005q2d`, `scripts/run_ev1_grasp_bsr.sh`); the `bsr005` Morris-4 outputs are kept
  and not used. (2) The other domains are not affected: no type of ERENO, CICIDS2017 or
  CICIoT2023 was excluded by Q2a, and none of their releases labels transactions. (3) The
  per-sample results of specific 13, 14 and 15 remain bounded by the share of their commands
  identical to benign (26.6 %, 23.2 %, 16.4 %); they are reported with that bound. The earlier
  Morris-4 files are kept outside the repository (`~/.trash/ereno-Adaptativo/2026-11-16/`).
  No evaluation model had been trained and no test result had been seen.

- **2026-11-15, result and correction (criterion), section 3: fusion rule on the new ERENO
  selection.** The selection specified in the next entry was run
  (`results/fusion_rule_selection_ev1.txt`). Under the criteria as registered it returned R3,
  the only rule meeting H1-H3, with a recall of 82.79 %: it detects masquerade_fake_normal on
  0.0 % and masquerade_fake_fault on 5.0 % of their calibration rows, and it passed H1 and H2
  because an attack with zero Full recall satisfies "at least 90 % of the Full recall". This is
  a defect of H1 and H2, corrected in `docs/FUSION_RULE.md` (change log of 2026-11-15) after
  this result was seen, with its evidence there; the test file was not read. Under the
  corrected criterion no rule meets H1-H3 and R0 alone meets two constraints (H1 and H2; its
  out-of-fold false-positive rate on the calibration benign is 0.63 %, against the bound of
  0.2 %, as on the test set in the first selection, 0.69 %). **R0 is chosen; section 3 is
  unchanged.** The run under the criteria as registered is kept
  (`results/fusion_rule_selection_ev1_h12original.txt`).

- **2026-11-15, specification (before the run), section 3: fusion rule selected again on
  the new ERENO selection, calibration set only.** Carries out the entry of 2026-11-13,
  consequence (1) (`scripts/fusion_rule_selection_ev1.py`). Procedure of
  `docs/FUSION_RULE.md` unchanged (candidates R0-R4, target fused false-positive rate 0.1 %,
  scenarios Full, Absence and Silent fault, constraints H1-H3, decision rule, tie of 0.5 pp,
  order of simplicity), with two differences. (1) Features: the ERENO selection of section 2.2
  (`features/all_in_one_ev1_ereno_train_combined.json`, 15 features); the 14 specialists are
  those of `docs/FUSION_RULE.md` section 2 (the `attack` partitioner, seed 42, benign cap
  500,000, the partition of section 2.3), each trained on 80 % of its partition
  (train_test_split, test_size 0.2, seed 42); the other 20 % of every partition is the
  calibration set. (2) No test set: the ERENO test file is not read. The calibration set is
  split into two folds (StratifiedKFold on the class, shuffled, seed 42); the thresholds of a
  fold are calibrated on the benign rows of the other fold and applied to it, and every metric
  of the scenarios is computed on the pooled out-of-fold predictions, so that H3 still bounds
  an out-of-sample false-positive rate (calibrating and measuring on the same benign rows would
  meet the target by construction). R0 and R1 have no threshold and are applied directly. The
  rule returned is the one evaluated (section 3 is updated if it is not R0). No evaluation
  model had been trained and no test result had been seen.

- **2026-11-14, correction (implementation) and amendment, section 2.2: per-type seeding;
  every GRASP selection re-run under the best-subset retention rule.** (1) Test of the
  resumption required by the entry of 2026-11-13, consequence (2), on a reduced Morris-4
  multiclass problem (sample 2,000): the run under the earlier rule (457 evaluations), the
  run under the best-subset retention rule from the start, and the same run resumed from the
  checkpoint of the first returned the same subset and F1-score; the last two have the same
  811 evaluations in the same order, and their first 457 equal those of the first run, whose
  RNG-state hashes were all verified on the replay. Passed. (2) Defect found while preparing
  the re-run: the per-type runs (step (a), `scripts/grasp_por_ataque.py`) never seeded Python's
  `random`, which drives the construction, the neighbourhoods and the BitFlip draws; its state
  came from the operating system at process start (two processes start from different
  states), and the benign and attack row draws shared one generator across the types, so a
  type's rows depended on the types run before it in the same process. The multiclass runs
  (step (b), `main_grasp.py`) seed `random` and are not affected. The per-type runs made so
  far (the 82 of the three new domains and the 7 ERENO runs of 2026-11-13) are valid runs of
  the recipe but cannot be reproduced, and those that would not yet have stopped under the
  best-subset retention rule cannot be continued from their logs; the re-run planned in the
  entry of 2026-11-13, consequence (2), is therefore not possible for them. The selection of
  the earlier work was produced by the same code. Correction: each per-type run seeds
  `random` with the GRASP seed at its start and draws its rows with a generator seeded by
  (GRASP seed, type id); per-type evaluation checkpoints, with a fingerprint of the rows, are
  added. Test on reduced Morris-4 per-type problems (sample 2,000): two processes from the
  start gave the same output and evaluation sequence; a run killed after 282 of its 480
  evaluations and resumed from its checkpoint gave the same output and evaluation sequence
  (282 RNG-state hashes verified); two types run in one process gave the same results as each
  type run alone. (3) Re-run (`scripts/run_ev1_grasp_bsr.sh`, outputs tagged `bsr005`, one job
  at a time, ERENO, Morris-4, CICIoT2023, CICIDS2017 in this order): every per-type run of
  the four domains from the start; each multiclass run resumed from the evaluations of its
  earlier run (ERENO and CICIDS2017 from their checkpoints, Morris-4 and CICIoT2023 from
  checkpoints seeded from their logs), whose evaluation sequence does not depend on the
  incumbent; then the union. The combined files `features/all_in_one_ev1_<domain>_train_combined.json`
  are replaced by the outputs of this re-run. No evaluation model had been trained and no
  test result had been seen.

- **2026-11-13, amendment, sections 2.3, 5 and 6: 14 nodes in every domain; references of
  the first evaluation step.** Reason: the runtime switching is evaluated by replaying the
  trust levels of the Disaster-FD run, which has 14 clients (`docs/TRUST_LEVEL.md`, section
  6); with one node per attack type (N = 7, 26, 30, 30) no domain matched it, and the
  specialists would have had to be mapped onto the clients by a rule chosen for the
  replay alone. With N = 14 everywhere, monitor NN drives node NN, and node loss runs from
  14 to 3 in every domain. Changes: (1) N = 14. (2) Specialists stay one per attack type,
  so each type is covered by exactly one specialist (ERENO: by two, on two nodes, its rows
  split in halves, which is the setting of the development data of `docs/FUSION_RULE.md`).
  The types with fewer than 100 training rows share one specialist, because a specialist
  trained on a few rows against a benign slice of about 79,000 (CICIDS2017; 3 rows for the
  smallest type) carries a weight of the attack class of up to about 26,000, and under OR
  its false positives reach the fused detector directly; 100 is the size below which this
  protocol already treats a type as small (section 7). Assigning several types to one
  specialist per node was considered and rejected: placed by size, every large type would
  have shared a booster of 10 trees with a type of 0.04 % to 2 % of its rows. (3) Placement
  by decreasing training rows in contiguous blocks over the 14 nodes (section 2.3), with the
  pipeline's rule for fewer specialists than nodes; the listing is
  `results/ev1_partition_n14.txt`. Dealing the specialists round-robin was considered and
  rejected: it puts the last, smallest specialists on the nodes of the largest ones (in
  CICIDS2017 the shared specialist of the 10 small types would sit with DDoS), so that every
  node mixing sizes would lose a rare type together with a high-volume one under node loss;
  in blocks, a node holds specialists of similar size, and the loss of high-volume and of
  rare types appears separately. (4) First evaluation step, references added: Local
  (each node alone), IID (the same rows in 14 random parts, one model per node) and
  Centralized (one model on the pooled rows); IID and Centralized have the same number of
  trees as the specialists together (10 per specialist; IID rounded to whole trees per
  node), so that the comparison varies only how the data are divided. (5) Window alarms
  count per specialist instead of per type (section 6). Consequences: the fusion-rule
  selection repeated on the new ERENO selection (entry below, consequence (1)) uses this
  partition; the trust-level replay with detection and the dissemination of a specialist
  while the server is unavailable are specified in a later entry, before they are run.
  The GRASP selection (section 2.2) does not depend on the partition and is unchanged. No
  evaluation model had been trained and no test result had been seen.

- **2026-11-13, amendment, section 2.2: ERENO selection re-run under the corrected
  implementation.** The earlier work is not changed: its selection and its criteria stay as
  published there. This evaluation registers the correction of 2026-11-12 and re-runs GRASP
  under it in every domain, ERENO included, with the recipe of section 2.2 (steps (a), (b),
  (c)); the published ERENO selection is no longer used here. The comparison planned in the
  entry of 2026-11-12, consequence (4), is dropped: the re-run of the 7 ERENO per-type runs
  with the parameters of the earlier work was stopped during the first type (random_replay),
  because replaying the code as it was on the new log does not reproduce the stored
  selection (`[9, 44, 55, 57]`, 4,762 evaluations, against `[19, 21, 57]`, 4,334): the search
  code is unchanged since that selection (commit `28f66cb`), but the software environment of
  those runs was not recorded, and the current one differs from it at least in the pandas
  version. From this entry on, each GRASP run records its environment (Python and package
  versions) next to its log. Partial log kept in
  `results/grasp_ereno_fixaudit_py314_stopped.log`; it is not used. The ERENO run
  (`scripts/run_ev1_grasp_ereno.sh`, outputs named `ev1_ereno`, log
  `results/grasp_ev1_ereno.log`) started on 2026-11-13. Consequences: (1) the fusion rule
  (section 3, `docs/FUSION_RULE.md`) was selected on the published ERENO selection; its
  selection procedure is repeated on the new ERENO selection, on the calibration set only,
  and the rule it returns is the one evaluated. (2) The 84 selections of the other domains
  are re-run under the corrected implementation, seeded from their logs: the evaluations a
  run makes do not depend on the incumbent, and those logs were produced in the current
  environment, so each logged evaluation is served from an evaluation checkpoint and only the
  evaluations beyond it are computed; the result is the one of a run from the start. This is
  conditional on a test of the resumption (a reduced run resumed from a checkpoint equals the
  same run from the start, in output and in evaluation sequence). No evaluation model had
  been trained and no test result had been seen.

- **2026-11-12, correction (implementation), section 2.2: GRASP incumbent and stopping
  rule.** Defect: the VND local search (`do_vnd` in `python/grasp/local_searches.py`)
  replaced the best subset found so far (the incumbent, which a run returns and against which
  an iteration is judged an improvement) by any subset better than the starting point of the
  current iteration, without comparing it with the incumbent. Two effects: (i) a run could
  return a subset worse, on the penalized criterion, than one it had already found; (ii) the
  result of an iteration had already become the incumbent when it was compared with it, so
  it never counted as an improvement, the counter of the registered rule (5 iterations
  without improvement) never reset, and every run made exactly the minimum number of
  iterations. Correction: the local search returns its result and leaves the incumbent to the
  caller (`GraspVND.run`), which replaces it only by a better subset (`do_vnd`; also
  `do_rvnd`, not used in evaluation-v1). This brings the implementation in line with the
  registered rule. Audit (`scripts/audit_grasp_incumbent.py`): the evaluations a run makes do
  not depend on the incumbent (each iteration rebuilds from the subset of the first
  iteration, and the local searches never read the incumbent), so each of the 84 selections
  in use was replayed from its log under both versions. The replay of the code as run
  reproduces all 84 stored outputs (subset, F1-score, iterations); in all 84 the counter
  never reset. Under the corrected rule, 19 selections are identical; 16 differ within the
  iterations already run (3 of them by a tie: another subset of the same penalized score);
  49 runs would not yet have stopped (Morris-4 10 of 31, CICIoT2023 23 of 31, CICIDS2017 16
  of 22), and in 37 of them the incumbent at that point already differs. Consequences: (1)
  the evidence of the entry of 2026-11-07 ("in 29 of the 30 Morris-4 per-type runs the best
  subset was found in the first iteration", from `iterations` − `no_improvement` = 1) is an
  artifact of the defect, since that difference was 1 in every run; the amendment to 5
  iterations, motivated by compute time, stands, but its expectation that the selection
  would not change and the decision to keep the 29 Morris-4 per-type selections computed
  under 15 do not. Under the corrected rule with 5 iterations, the 30 Morris-4 per-type runs
  all stop within their logged iterations (19 identical, 11 different). (2) "Stopped on the
  iteration rule" in the entry below refers to the rule as implemented. (3) The CICIDS2017
  step (b) ran under the defect and finished on 2026-11-13 after 7 iterations with
  `[25, 29, 42, 69, 71, 82]` (F1-score 98.684); the replay reproduces it. Its subset is the
  best one evaluated, but under the corrected rule the counter stood at 4, not 5, so the run
  would have gone on for at least one more iteration; its evaluations are valid and stay in
  its checkpoint. It is kept as it is for now; whether it is resumed under the corrected rule
  is decided after the ERENO audit. (4) The ERENO selection (section 2.2), published with the
  earlier work, was produced by the same implementation but no log of its runs exists. Its 7
  per-type runs are re-run with the corrected code and the same parameters (seed, sample,
  benign cap, 15 iterations without improvement, λ = 0.05); the comparison is valid only if
  replaying the code as it was on the new log reproduces each stored selection. Criterion,
  fixed before the run: the published selection is kept if, for every type, the corrected
  selection exceeds it on the penalized cross-validation F1-score by less than 0.05 points
  (the value of one feature); otherwise the OR fusion of the two unions is compared on the
  calibration set. The test split is not used. The outcome is recorded in a later entry. How the
  selections are brought under the corrected implementation is recorded in a later entry.
  The audit uses the cross-validation on the training split only; no evaluation model had
  been trained and no test result had been seen.

- **2026-11-12, wording correction, section 3: OR fires at a score of at least 0.5.** The
  section said "above 0.5"; the implementation flags a sample when a specialist's
  probability is >= 0.5 (`main_dist.py`), and the evaluation applies it as implemented. The
  same correction is made, with its evidence, in the change log of `docs/FUSION_RULE.md`:
  the two forms differ only for a score of exactly 0.5, none occurs in the ERENO selection
  data, so the selection of R0 is the same under either. No run is affected.

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
