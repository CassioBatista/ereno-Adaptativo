# Dataset audit — index

Every public release considered for the evaluation was audited with the same protocol
before use. Of the **22 releases audited, 4 are retained** (two of them restricted to a
label subset), 16 are rejected and 2 are not used. This file indexes, for each release, the
outcome, the script that runs the audit, its report and the checksums of the audited files.
The full discussion is in the paper (dataset audit subsection).

The datasets themselves are not redistributed; each script header states where the audited
copy was obtained and the command that runs it.

## Protocol

Six checks are applied to every release.

| Check | What is verified |
|---|---|
| **S** — structure | fields per line, embedded header lines, agreement between the kind of value a column holds (epoch time, radio frequency, signal power, IPv4 address) and its header name |
| **L** — labels | exact duplicates, duplicates once identifiers are removed, feature vectors carrying more than one label, identical files filed under different labels |
| **T** — time | presence and completeness of timestamps; classes interleaved or recorded in separate sessions |
| **A** — attribution | whether source addresses exist and point to the emitter |
| **P** — prevalence | class balance and benign volume |
| **X** — shortcut probe | gradient-boosted classifier on deduplicated rows, with and without the columns that locate a row in a capture (timestamps, frame and sequence numbers, checksums, ports, addresses), on a random split and, where available, on independent data |

Three probe outcomes indicate that labels are predictable from the capture rather than from
the attack: separability carried by one such column, a collapse once they are removed, or
separability that holds on a random split but not on independent data.

## Outcome

| Dataset | Domain | Failed checks | Main finding | Decision |
|---|---|---|---|---|
| ERENO | IEC 61850 substation | A | attacks impersonate the single publisher | retained |
| CICIDS2017, improved (Liu et al., version of 27 Apr 2023) | enterprise network | L, X | scans duplicated across two labels; initial TCP window acts as an OS fingerprint | retained, two scan labels merged, initial window excluded |
| Electra Modbus | industrial control network | L | three labels mark the man-in-the-middle path, not the packet | retained, `electra-4` (4 of 7 attack types) |
| UNSW IoT attack traces | consumer IoT devices | L, A, X | TCP/UDP floods invisible in minute counters | retained, `unsw-iot-3` (3 of 7 categories), no attribution |
| CICIoT2023 | IoT network | L, T, A, P | four of seven categories labelled by capture session | rejected |
| ToN_IoT | IoT/IIoT network | S, T, P, X | narrow normal class; 18% of normal flows flagged | rejected |
| IoT-23 | IoT network (malware) | L, T, P, X | "benign" is the infected host's unlabelled traffic | rejected |
| X-IIoTID | industrial IoT | T, A, X | host statistics act as a session clock; 13% of normal flagged | rejected |
| Kitsune | IoT and IP cameras | S, L, T, A, X | timestamps in features; normal traffic identifies the capture | rejected |
| Edge-IIoTset | industrial IoT | S, L, T, A, X | two classes shifted by one column | rejected |
| UAV-NIDD | UAV network | S, L, T, A, P, X | captures reused across scenarios and labels | rejected |
| PDNet-IDS26 | medical IoT | P, X | class identified by the TCP connection | rejected |
| Hassler et al. (UAV) | UAV, cyber-physical | S, T, X | two attacks without benign data in their schema | rejected |
| CIC-BCCC-NRC-IoMT-2024 | medical IoT | L, T, A, P, X | ICMP, UDP and ARP classes hold only TCP flows | rejected |
| CIC IIoT 2025 (DataSense) | industrial IoT, sensors | L, T, A, X | empty windows labelled attack; sensors carry no attack signal | rejected |
| Morris-1 | power system (PMU) | T, A, P | 1% random sample: no time order; 71% attacks | rejected |
| Morris-2 | gas pipeline (Modbus) | L, T, A | no clock; 85% of one attack identical to normal | rejected |
| Morris-3 | water tank (Modbus) | L, T, A, X | no clock; one attack identical to normal | rejected |
| Morris-4 | gas pipeline (Modbus) | L, A, X | 27% of normal packets identical to attacks; 65% flagged | rejected |
| PowerDuck | IEC 61850 substation | A, P, X | floods are 92.5% of packets; no transfer across recordings | rejected |
| Westermo | ICS network | L | timing labels mark flows not involving the attacker; adds no property beyond the improved CICIDS2017 | not used |
| HAI 23.05 | ICS process (HIL) | A, X | sound labels; supervised detection fails on unseen attacks (one-class design needed) | not used |

### Conditions of use of the retained datasets

- **ERENO** — passes S, L and T; the probe stays near-perfect on the authors' independent
  test file (macro-F1 0.994) without absolute time and the SqNum/StNum counters. Fails A:
  all benign messages and 99.9–100% of the attack messages carry the same publisher
  identity, so source attribution would point at the victim.
- **CICIDS2017, improved** — 126 vectors carry two labels (91,253 rows, almost all scans
  labelled both "Portscan" and "Infiltration – Portscan"): the two labels are merged. The
  probe falls from 0.991 (random) to 0.932 (time-ordered) and to 0.908 without the initial
  TCP window sizes (port scan 0.86 → 0.62): the initial window is excluded from the
  features. Attempted flows (11,979) are labelled benign; Heartbleed (11 flows) is excluded.
- **Electra Modbus** — 99.75% of rows are duplicates on the five content fields and 97% lie
  in vectors with more than one label. `MITM_UNALTERED` (83%), `REPLAY_ATTACK` (97%) and
  `READ_ATTACK` (66%) share the content of normal packets and are told apart only by the
  MAC address (95% of labels from the address alone, against 85% for the majority class).
  They are removed from the stream. The four retained types share almost no vector with
  normal traffic (none for three, 0.18% of the modified responses); probe F1 0.97–1.00 on
  random and time-ordered splits. Retained attacks: 0.41% of the packets.
- **UNSW IoT attack traces** — the counters of the attacked protocol rise in every
  annotated interval of ARP spoofing, ping of death and Smurf, but only in 58–86% of the
  intervals of the TCP/UDP floods and reflections. The probe falls from 0.85 (random) to
  0.37 (time-ordered, intervals kept whole); ARP spoofing holds at 0.91. `unsw-iot-3` keeps
  the three categories visible in every interval; this criterion was applied after a first
  run of the evaluation protocol with all seven categories (5.4% of normal device-minutes
  flagged). Counters belong to the victim device: no attribution.

## Files

Scripts are in `scripts/`, reports in `results/`. Where the audit needed the evaluation
protocol to decide, the protocol script is listed too.

| Dataset | Audit script | Report(s) | Checksums | Audited copy |
|---|---|---|---|---|
| ERENO | `inspect_ereno.py` | `ereno_audit.txt` | — | release used by the pipeline (`build_ereno_dataset.py`) |
| CICIDS2017, improved | `inspect_cicids2017c.py`, `inspect_cicids2017_improved.py` | `cicids2017c_audit.txt`, `cicids2017_improved_inspection.{txt,json}` | — | `CICIDS2017_improved.zip`, authors' release |
| Electra Modbus | `inspect_electra.py`; protocol `cross_domain_electra.py` | `electra_audit.txt`, `cross_domain_electra.txt` | `electra_md5sums.txt` | perception.inf.um.es/ICS-datasets |
| UNSW IoT attack traces | `inspect_unsw_iot.py` (audit and protocol) | `unsw_iot_audit.txt` | `unsw_iot_attack_md5sums.txt` | iotanalytics.unsw.edu.au/attack-data |
| CICIoT2023 | `inspect_ciciot2023.py`; label and protocol checks `_ciciot_label_protocol_check.py`, `_ciciot_threshold_calibration.py`, `_ciciot_volumetric_subset.py` | `ciciot2023_audit.txt` (10 of 169 files); summary in `docs/RESULTADOS_ciciot2023.md` | — | CSV release, authors' site |
| ToN_IoT | `inspect_toniot.py`, `inspect_toniot_followup.py`; protocol `cross_domain_toniot.py` | `toniot_audit.txt`, `toniot_followup.txt`, `cross_domain_toniot.txt`, `cross_domain_toniot_session.txt` | `toniot_md5sums.txt` | Kaggle mirror (see script header) |
| IoT-23 | `inspect_iot23.py` | `iot23_audit.txt` | `iot23_md5sums.txt` | Stratosphere Laboratory archive (labelled flows) |
| X-IIoTID | `inspect_xiiotid.py`; protocol `cross_domain_xiiotid.py` | `xiiotid_audit.txt`, `cross_domain_xiiotid.txt` | `xiiotid_md5sums.txt` | dataset author's Kaggle upload |
| Kitsune | `inspect_kitsune.py` (audit and protocol) | `kitsune_audit.txt` | `kitsune_md5sums.txt` | dataset author's Kaggle upload |
| Edge-IIoTset | `inspect_edge_iiotset.py` | `edge_iiotset_inspection.txt` | — | authors' release (full DNN file and ML file) |
| UAV-NIDD | `inspect_uav_nidd.py`, `list_uav_nidd_figshare.py` | `uav_nidd_inspection.txt`, `uav_nidd_figshare_listing.txt` | listed with the raw files | figshare article 25486462 |
| PDNet-IDS26 | `inspect_pdnet_ids26.py` | `pdnet_ids26_inspection.txt` | — | authors' release |
| Hassler et al. (UAV) | `inspect_uav_hassler.py` | `uav_hassler_inspection.txt` | — | authors' GitHub release (`Dataset_T-ITS.csv`) |
| CIC-BCCC-NRC-IoMT-2024 | `inspect_iomt2024.py` | `iomt2024_audit.txt` | `iomt2024_md5sums.txt` | Kaggle mirror (see script header) |
| CIC IIoT 2025 | `inspect_cic_iiot2025.py` | `cic_iiot2025_audit.txt` | `cic_iiot2025_md5sums.txt` | Kaggle mirror (see script header) |
| Morris-1 | `inspect_cps_survey4.py` | `morris1_audit.txt` | — | Kaggle mirror (original links return 404) |
| Morris-2, Morris-3, Morris-4 | `inspect_morris_ics.py`; protocol on Morris-4 `cross_domain_morris4.py` | `morris_ics_audit.txt`, `cross_domain_morris4.txt` | `morris_ics_md5sums.txt` | Web Archive copies (original page gone) |
| PowerDuck | `inspect_cps_survey4.py` | `powerduck_audit.txt`, `cps_survey4_followup.txt` | — | Zenodo 6974112 |
| Westermo | `inspect_cps_survey4.py` | `westermo_audit.txt`, `cps_survey4_followup.txt` | — | authors' GitHub release |
| HAI 23.05 | `inspect_hai2305.py` | `hai2305_audit.txt` (HAI 20.07: `hai2007_audit.txt`) | SHA-256 checked against GitHub LFS | authors' GitHub/Kaggle release |

## Not audited

Screened out on their published description, from the survey of CPS intrusion datasets
used to shortlist the power-system and ICS candidates: EPIC (no per-sample labels), Biswas
et al. and ICS-ADD (unlabelled packet captures), the Sahu dataset (not public) and
ICS-Flow (no physical data).

Not considered, for reasons documented in the literature: the original CICIDS2017 (label
and flow-feature errors), CSE-CIC-IDS2018 (same generation methodology and error classes),
NSL-KDD (no timestamps or addresses; artefacts of the 1998 DARPA traffic), and
re-extractions of rejected captures (CIC-ToN-IoT; NF-UQ-NIDS, which merges ToN_IoT and
CSE-CIC-IDS2018 among others).

## Running

Each script runs on its own and writes its report to `results/`; the command is in the
script header. Several scripts read releases of millions of rows: run them one at a time
on machines with less than about 16 GB of memory.
