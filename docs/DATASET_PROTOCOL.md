# Dataset screening protocol (frozen)

Status: **frozen on 2026-11-03 (tag `protocol-v1`)**, committed before the protocol was
applied to any release. After that, only error corrections are
allowed, each recorded in the change log at the end of this file with its date and reason.
Criteria, thresholds and the selection rule are not changed after the results are seen.

## 1. Purpose

Decide which public releases can be used to evaluate ReSIDS, with criteria that any
intrusion-detection evaluation needs, independent of the ReSIDS architecture and of any
detection result. Whether a release carries time or source addresses is **not** a criterion:
it is an entry of the release's domain profile.

## 2. History (declared)

An exploratory audit (`docs/DATASET_AUDIT.md`, `scripts/inspect_*.py`) was run before this
protocol was written. It examined 22 releases with checks that changed during the work and
with a shortcut probe whose results depend on its parameters. This protocol replaces it for
the selection. The exploratory reports remain in the artifact as descriptive material.

## 3. Candidate pool

The 22 releases examined by the exploratory audit:

ERENO; CICIDS2017, improved (Liu et al., version of 27 April 2023); Electra Modbus; UNSW IoT
attack traces; CICIoT2023; ToN_IoT (network part); IoT-23; X-IIoTID; Kitsune; Edge-IIoTset;
UAV-NIDD; PDNet-IDS26; Hassler et al. (UAV); CIC-BCCC-NRC-IoMT-2024; CIC IIoT 2025
(DataSense); Morris-1; Morris-2; Morris-3; Morris-4; PowerDuck; Westermo; HAI 23.05.

No release is added or removed from the pool after approval.

## 4. Unit of decision

The decision is taken per **attack type** (the finest label the release provides, or the
category used by its authors when types are not given), then per release. A release
qualifies if, after the per-type decisions, it keeps at least **two attack types** and its
benign class meets Q3.

## 5. Criteria

All checks are deterministic counts on the files; no model is trained.

### Q1 — Well-formed files

- **Q1a, alignment.** A data line whose number of fields differs from the header, or whose
  value kinds contradict the column names (an IP address in a time column, a protocol
  number in an address column), is *misaligned*. Misaligned lines are dropped. An attack
  type with more than **1 %** of its lines misaligned is excluded.
- **Q1b, duplicated files.** Two data files with the same checksum filed under different
  labels: every attack type involved is excluded.

### Q2 — Labels describe the attack

- **Q2a, separability from benign.** Identifier columns are removed (time, addresses, ports,
  frame and sequence numbers, checksums, session and subject ids). An attack type is
  excluded if **more than 50 %** of its samples have a feature vector that also occurs as
  benign: most of its samples cannot be told from benign traffic by any detector.
- **Q2b, protocol consistency** (when the release has protocol indicators). An attack type
  named after a protocol (ARP, ICMP, DNS, HTTP, MQTT, Modbus, UDP, TCP SYN) is excluded if
  **50 % or fewer** of its samples carry that protocol.
- **Q2c, annotation visible in the data** (when labels come from annotated intervals rather
  than per sample). An attack type is excluded if the counters of the protocol it uses rise
  above the preceding hour (ratio > 2) in **50 % or fewer** of its annotated intervals.
- **Q2d, label scope.** Samples labelled by a rule that marks every sample of a period or
  capture regardless of its content (for example "every flow during the event", "every
  window of an attack session including empty ones"): when the release offers another
  labelling that does not, that one is used; otherwise the samples concerned are excluded
  if they can be identified (empty windows), and the attack type is excluded if they
  cannot.

### Q3 — Benign traffic for false positives

- The benign class shares the feature space of the attack types (same columns, same
  extraction).
- At least **10,000** benign samples, so that a held-out 30 % (3,000 samples) bounds a
  false-positive rate of 0.1 % at 95 % confidence when no false positive occurs (rule of
  three).

## 6. Domain profile (recorded, not a criterion)

For each qualified release: time (clock / sample order only / none); source address
(points at the emitter / at the victim / none); sample unit (packet, flow, window,
device-minute, process reading); classes recorded interleaved or in separate sessions.

## 7. Selection among qualified releases

- ERENO is the case study of this work and is included if it qualifies.
- Domain classes: the four kinds of system that motivate this work (introduction):
  - **IEC 61850 substation:** ERENO, PowerDuck;
  - **enterprise network:** CICIDS2017, improved;
  - **industrial control (network or process):** Electra Modbus, Morris-2, Morris-3,
    Morris-4, Westermo, HAI 23.05;
  - **IoT (consumer and industrial):** UNSW IoT attack traces, CICIoT2023, ToN_IoT, IoT-23,
    X-IIoTID, Kitsune, Edge-IIoTset, CIC IIoT 2025.

  The other releases of the pool (CIC-BCCC-NRC-IoMT-2024 and PDNet-IDS26, medical IoT;
  UAV-NIDD and Hassler et al., UAV networks; Morris-1, power-system measurements) are
  screened and reported but are outside the scope of the evaluation.
- Citations are applied **only after Q1-Q3**: they rank the releases of a domain class that
  already qualified, and never qualify or exclude a release. For each domain class with at
  least one qualified release, one release is evaluated: the one whose **requested
  reference** received the most citations in **2025 and 2026**. The window counts current
  use rather than accumulated age, so that older releases are not favoured. The count is a
  tie-breaker, not a measure of quality.
- Requested reference, in order of precedence:
  1. the reference that the documentation of the release (dataset page, README, record)
     asks users to cite; when it lists several, the one that introduces the part used;
     when it names a preprint, its published version; when it asks to cite the data
     record itself (Zenodo, GitHub, UCI), that record;
  2. when no reference is requested, the publication the documentation names as
     describing the release;
  3. otherwise, the publication that introduces the release.
- Citations are counted once, in OpenAlex, on a single date, before the protocol is
  applied: the citations the reference received in 2025 and in 2026 up to that date
  (OpenAlex `counts_by_year`), found by DOI when the reference has one, otherwise by exact
  title (`scripts/dataset_citations.py`, report `results/dataset_citations.txt`). A
  reference that OpenAlex does not index has no count and ranks last in its class. The
  counts and the date are recorded in the table below and not updated afterwards.
- Ties: when one reference introduces two releases (Morris-2 and Morris-3), the release
  with more benign samples; otherwise the more recent release.
- Every other qualified release is listed in the supplementary material as qualified and
  not evaluated.

### Citation counts (filled before application)

Requested references checked on 2026-11-03 (rule 1, 2 or 3 above). Citations counted on
2026-11-03 (OpenAlex): received in 2025-2026 (the criterion), with the all-time total in
parentheses. n.i. = not indexed by OpenAlex.

| Release | Requested reference | Rule | Basis | Citations 2025-2026 (total) |
|---|---|---|---|---|
| ERENO | Quincozes et al., "ERENO: A Framework for Generating Realistic IEC-61850 Intrusion Detection Datasets for Smart Grids", IEEE TDSC, 2023 | 3 | repository README requests no reference | 38 (46) |
| CICIDS2017, improved | Liu et al., "Error Prevalence in NIDS Datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018", IEEE CNS, 2022 | 1 | release page gives the BibTeX of this paper | 84 (127) |
| Electra Modbus | Perales Gómez et al., "On the Generation of Anomaly Detection Datasets in Industrial Control Systems", IEEE Access, 2019 | 3 | release page (archived 2022) requests no reference | 38 (120) |
| UNSW IoT attack traces | Hamza et al., "Detecting Volumetric Attacks on IoT Devices via SDN-Based Monitoring of MUD Activity", ACM SOSR, 2019 | 2 | page: "Data collected for our paper at ACM SOSR 2019"; "Cite our data" section empty | 38 (193) |
| CICIoT2023 | Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment", Sensors, 2023 | 1 | release page, Citation section | 982 (1,248) |
| ToN_IoT (network) | Moustafa, "A New Distributed Architecture for Evaluating AI-based Security Systems at the Edge: Network TON_IoT Datasets", Sustainable Cities and Society, 2021 | 1 | page requests eight papers; this one introduces the network part | 418 (747) |
| IoT-23 | Garcia, Parmisano, Erquiaga, "IoT-23: A labeled dataset with malicious and benign IoT network traffic", Zenodo, 2020 (doi:10.5281/zenodo.4743746) | 1 | release page requests the data record | 75 (240) |
| X-IIoTID | Al-Hawawreh et al., "X-IIoTID: A Connectivity-Agnostic and Device-Agnostic Intrusion Data Set for Industrial Internet of Things", IEEE IoT Journal, 2022 | 3 | author's Kaggle page has no description | 153 (285) |
| Kitsune | Mirsky et al., "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", NDSS, 2018 | 1 | author's repository: "If you use ... the datasets ... please cite" (the UCI copy gives its own data citation; the audited copy is the author's upload) | 20 (212) |
| Edge-IIoTset | Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning", IEEE Access, 2022 | 1 | release Readme requests the TechRxiv preprint; published version counted | 778 (1,218) |
| UAV-NIDD | Hadi et al., "UAV-NIDD: A Dynamic Dataset for Cybersecurity and Intrusion Detection in UAV Networks", IEEE TNSE, 2025 | 1 | release README, Citation section | 45 (45) |
| PDNet-IDS26 | Lui et al., "PDNet-IDS26: A Healthcare 5.0 Multiclass Intrusion Detection Dataset with Biomedical and Network Features", SBCAS, 2026 | 3 | release page not found | 0 (0) |
| Hassler et al. (UAV) | Hassler, Mughal, Ismail, "Cyber-Physical Intrusion Detection System for Unmanned Aerial Vehicles", IEEE T-ITS | 1 | release README, "Citing this work" | 82 (90) |
| CIC-BCCC-NRC-IoMT-2024 (tabular) | Sasi et al., "An Efficient Self Attention-Based 1D-CNN-LSTM Network for IoT Attack Detection and Identification Using Network Traffic", Journal of Information and Intelligence, 2024 | 1 | page of the tabular release (the original CICIoMT2024 captures request Dadkhah et al., Internet of Things, 2024) | 58 (58) |
| CIC IIoT 2025 | Firouzi et al., "DataSense: A Real-Time Sensor-Based Benchmark Dataset for Attack Analysis in IIoT with Multi-Objective Feature Selection", Electronics, 2025 | 1 | release page, Citation section | 32 (32) |
| Morris-1 | Pan, Morris, Adhikari, "Developing a Hybrid Intrusion Detection System Using Data Mining for Power Systems", IEEE TSG, 2015 | 2 | page lists four works that used the sets, this one first; no request | 56 (457) |
| Morris-2, Morris-3 | Morris, Gao, "Industrial Control System Traffic Data Sets for Intrusion Detection Research", Critical Infrastructure Protection VIII, 2014 | 2 | page: attacks "described in" this work; no request | 49 (241) |
| Morris-4 | Morris, Thornton, Turnipseed, "Industrial Control System Simulation and Data Logging for Intrusion Detection System Research", 7th Annual Southeastern Cyber Security Summit, 2015 | 1 | page: "Please cite the following paper when using these datasets" | n.i. |
| PowerDuck | Zemanek et al., "PowerDuck: A GOOSE Data Set of Cyberattacks in Substations", CSET, 2022 | 2 | Zenodo record: data set "as described in" this paper; Citation section empty | 14 (25) |
| Westermo | Strandberg et al., "The Westermo network traffic data set", GitHub, 2023 | 1 | README "Suggested attribution" names the data set itself | n.i. (the data paper of the same title, Data in Brief 2023: 9 (18)) |
| HAI 23.05 | "HAI security datasets", GitHub, 2023 | 1 | README requests this entry for HAI 22.04 and 23.05 | n.i. |

## 8. Reporting

- For every release: Q1-Q3 counts, per-type decisions, profile entries, final status.
- For every qualified and selected release: results on all its qualified attack types.
  Results on excluded attack types are also reported when the release is evaluated, marked
  as excluded by Qx.
- The shortcut probe of the exploratory audit may be reported in the supplementary
  material as description; it takes no part in any decision.

## Change log

- **2026-11-03, amendment (not an error correction), section 7, after the screening and
  before any evaluation run:** *a qualified release that its authors superseded with a
  corrected release is replaced by that release, when the corrected release also
  qualifies.* The authors of Morris-2 and Morris-3 state on the dataset page that the 2014
  releases "contain some unintended patterns" and released Morris-4 (2015) to fix them.
  The citation rule had selected Morris-3 (49 citations of Morris and Gao 2014) because the
  reference requested by Morris-4 (Morris, Thornton and Turnipseed 2015) is not indexed by
  OpenAlex. Morris-4 qualifies (30 of 35 types) and replaces Morris-3 in the industrial
  control class. Decided before any model of the evaluation was trained; the alternative
  of excluding superseded releases, which would have selected Electra, was not taken.

- **2026-11-03, operational definitions fixed while writing `scripts/apply_protocol.py`**
  (no criterion, threshold or selection rule changed; listed so that every choice made
  after the freeze is visible):
  - *Unit (section 4):* the finest label of each release is: CICIDS2017, improved, the
    `Label` column (the "Attempted" labels are attack types of their own); CICIoT2023, the
    34 labels; IoT-23, the detailed label; X-IIoTID, `class1`; CIC-BCCC-NRC-IoMT-2024, the
    file (one per class); CIC IIoT 2025, `label4` (category and attack type;
    `label_full` adds the target of each window); Morris-4, `specific result`; PowerDuck, the attack recording; UNSW, the attack
    name without rate and origin; HAI 23.05, the target control loop of the manual; the
    label column as given for the others.
  - *Releases with several tables* (UAV-NIDD cases, Hassler et al. schemas, CIC IIoT 2025
    window lengths, Westermo labellings): each table is screened with its own benign class;
    the release qualifies through its best table.
  - *Q1a:* checked columns are those named after an address, a port or a time. A row is
    misaligned when its value is not of the named kind while at least half of the
    non-empty values of that column are; a column holding another representation
    throughout (addresses encoded as integers) marks no row. Lists of addresses ("a/b")
    and missing values pass. Rows without a label are misaligned.
  - *Q1b:* per-label files by MD5; for UAV-NIDD, the listing of the raw release (CRC32 and
    size, `results/uav_nidd_figshare_listing.txt`), folders mapped to labels by their first
    four letters.
  - *Q2b:* type names are split into words (separators and camel case); the protocol words
    are ARP, ICMP, DNS, HTTP/HTTPS, MQTT, Modbus/MBTCP, UDP, SYN. A sample carries the
    protocol when a column named after it is non-zero, or when an IP protocol field holds
    its number or name; with an IP protocol field, a sample does not carry ARP.
  - *Q2d:* applied when the labelling rule is documented (Westermo timing labels) or when
    attack-labelled samples have no content (CIC IIoT 2025 windows with no traffic and no
    sensor reading).
