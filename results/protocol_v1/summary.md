# Dataset screening — protocol-v1

Applied by `scripts/apply_protocol.py`; per-release reports in this folder.

| Release | Class | Types kept / total | Benign | Qualifies | Citations 2025-26 | Selected |
|---|---|---|---|---|---|---|
| ereno (ereno) | IEC 61850 substation | 7 / 7 | 5,514,564 | yes | 38 | **yes** |
| cicids2017c (cicids2017c) | enterprise network | 26 / 26 | 1,582,566 | yes | 84 | **yes** |
| electra (electra) | industrial control | 4 / 7 | 13,894,323 | yes | 38 |  |
| unsw_iot (unsw_iot) | IoT | 7 / 7 | 568,160 | yes | 38 |  |
| ciciot2023 (ciciot2023) | IoT | 30 / 33 | 55,859 | yes | 982 | **yes** |
| toniot (toniot) | IoT | 8 / 9 | 387,542 | yes | 418 |  |
| iot23 (iot23) | IoT | 10 / 15 | 183,150 | yes | 75 |  |
| xiiotid (xiiotid) | IoT | 17 / 18 | 420,825 | yes | 153 |  |
| kitsune (kitsune) | IoT | 5 / 5 | 1,323,834 | yes | 20 |  |
| edge_iiotset (edge_iiotset) | IoT | 11 / 14 | 1,615,643 | yes | 778 |  |
| uav_nidd (uav_nidd_2) | out of scope | 5 / 10 | 76,737 | yes | 45 |  |
| pdnet_ids26 (pdnet_ids26) | out of scope | 3 / 3 | 5,388 | no | 0 |  |
| uav_hassler (uav_hassler_1) | out of scope | 2 / 2 | 9,425 | no | 82 |  |
| iomt2024 (iomt2024) | out of scope | 9 / 14 | 32,620 | yes | 58 |  |
| cic_iiot2025 (cic_iiot2025_1) | IoT | 83 / 83 | 136,800 | yes | 32 |  |
| morris1 (morris1) | out of scope | 1 / 1 | 22,714 | no | 56 |  |
| morris2 (morris2) | industrial control | 6 / 7 | 61,156 | yes | 49 |  |
| morris3 (morris3) | industrial control | 6 / 7 | 172,415 | yes | 49 | **yes** |
| morris4 (morris4) | industrial control | 30 / 35 | 214,580 | yes | n.i. |  |
| powerduck (powerduck) | IEC 61850 substation | 6 / 16 | 84,945 | yes | 14 |  |
| westermo (westermo_2) | industrial control | 7 / 7 | 54,028 | yes | n.i. |  |
| hai2305 (hai2305) | industrial control | 6 / 6 | 1,169,416 | yes | n.i. |  |

Excluded types per release:

- **electra**: MITM_UNALTERED (Q2a 83.0% identical to benign); READ_ATTACK (Q2a 65.8% identical to benign); REPLAY_ATTACK (Q2a 97.1% identical to benign)
- **ciciot2023**: MITM-ArpSpoofing (Q2b 0.1% carry ARP); DDoS-UDP_Fragmentation (Q2b 12.0% carry UDP); DNS_Spoofing (Q2b 0.2% carry DNS)
- **toniot**: ransomware (Q2a 90.9% identical to benign)
- **iot23**: PartOfAHorizontalPortScan (Q2a 54.6% identical to benign); Okiru (Q2a 80.0% identical to benign); C&C-HeartBeat (Q2a 56.3% identical to benign); C&C (Q2a 63.5% identical to benign); C&C-Torii (Q2a 66.7% identical to benign)
- **xiiotid**: crypto-ransomware (Q1a 33.84% misaligned)
- **edge_iiotset**: DDoS_UDP (Q1a 99.97% misaligned); DDoS_HTTP (Q2b 1.4% carry HTTP); MITM (Q1a 30.23% misaligned)
- **uav_nidd**: Brute-Force (Q1a 93.79% misaligned); Scanning (Q1a 38.63% misaligned, Q1b raw capture identical to folders ['reco']); Fake Landing (Q1b raw capture identical to folders ['deau']); Ewil Twin (Q1a 51.21% misaligned); MITM (Q1a 47.91% misaligned)
- **iomt2024**: DoS UDP Flood (Q2b 0.0% carry UDP); DDoS UDP Flood (Q2b 0.0% carry UDP); DDoS ICMP Flood (Q2b 0.0% carry ICMP); DoS ICMP Flood (Q2b 0.0% carry ICMP); MITM ARP Spoofing (Q2b 0.0% carry ARP)
- **morris2**: MPCI (Q2a 83.6% identical to benign)
- **morris3**: CMRI (Q2a 99.0% identical to benign)
- **morris4**: specific 18 (Q2a 81.1% identical to benign); specific 14 (Q2a 61.6% identical to benign); specific 13 (Q2a 63.3% identical to benign); specific 15 (Q2a 58.2% identical to benign); specific 2 (Q2a 50.1% identical to benign)
- **powerduck**: 15-flood-repeat (Q2a 100.0% identical to benign); 12-sup-1-1-tbv2 (Q2a 100.0% identical to benign); 11-sup-1-1-tbv1 (Q2a 100.0% identical to benign); 14-sup-1 (Q2a 100.0% identical to benign); 13-sup-2 (Q2a 100.0% identical to benign); 10-sup-1-1-tbv0 (Q2a 100.0% identical to benign); 04-insert-fake-open-w-intermediate (Q2a 100.0% identical to benign); 05-insert-fake-open-only-end (Q2a 100.0% identical to benign); 02-replay-opening-switch-w-context (Q2a 100.0% identical to benign); 01-replay-opening-switch-isolated (Q2a 100.0% identical to benign)
