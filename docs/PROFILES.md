# Domain profiles

ReSIDS v2 is an **adaptive IDS with a domain-neutral core**. The event model, the
command surface, the authority rule, the reconciliation contract and the two Disaster-FD
cadences are the same in every domain. A domain enters through three doors only — the
**data** (which specialists are trained), the **profile** (this document) and the
**failure-detection provider** (Disaster-FD).

A profile fixes what the core deliberately leaves abstract:

| Core notion | Meaning in the core | What the profile decides |
|---|---|---|
| agent | one ReSIDS agent per monitored device | what a monitored device is |
| window | samples scored between two local Disaster-FD ticks | tick period, or count-based tick when the data has no time |
| traffic time | optional metadata, never the window | whether the traffic carries timestamps (`traffic_time: frame \| none`) |
| emitter (`source_node`) | the node whose traffic an alarm is attributed to | where attribution comes from, or that it does not exist |
| attack label (`attack`) | a label from the profile's label set | the label set (`label_set`) |
| transport security | mutual authentication + per-command signature ([`COMMANDS.md`](COMMANDS.md) §6) | whether the domain prescribes a specific transport |

The profile in force is declared in the scenario (`profile` block,
[`SCENARIOS.md`](SCENARIOS.md)), recorded in the provenance manifest, and reported by
`/status` ([`API.md`](API.md)), so a consumer never has to guess which vocabulary an event
speaks.

---

## `iec61850-goose-sv` — power-system substations (ERENO)

The case study of the evaluation.

| | |
|---|---|
| Monitored device | an IED (Intelligent Electronic Device) on the process bus |
| Local tick | time-based, 1 s by default — about 4.7 k Sampled Values at the SV cadence of ~0.21 ms, delivered in bursts |
| Traffic time | `frame`: GOOSE/SV frames carry timestamps; events add `traffic_time_start/end` for correlation with the substation's sequence-of-event records |
| Attribution | the GOOSE/SV publisher: APPID, `gocbRef`, source MAC. **Not implemented in v2** (`source_node` is null outside the synthetic scenarios); a v3 premise. The ERENO audit shows why it cannot come from the data: every message, benign or attack, carries the same publisher identity (one MAC, one `gocbRef`), because the attacks impersonate the legitimate publisher ([`../results/ereno_attribution.txt`](../results/ereno_attribution.txt)) |
| Label set | `ereno-7`: `random_replay`, `inverse_replay`, `masquerade_fake_fault`, `masquerade_fake_normal`, `injection`, `high_StNum`, `poisoned_high_rate` |
| Time-critical traffic | protection messages with a 3 ms transfer-time budget (TT6); the IDS control plane must never contend with them (lower 802.1Q priority) |

**Security alignment — IEC 62351.** IEC 62351 defines no DTLS profile. It delegates the
transport protection of TCP/IP profiles to TLS (62351-3; 62351-4 for MMS), while GOOSE/SV
(62351-6) are protected inside the PDU with group keys (62351-9), because TLS does not serve
multicast within a 3 ms budget. Two consequences:

1. The ReSIDS–Disaster-FD channel is an out-of-band *management* interface, which 62351
   does not prescribe. Choosing DTLS for it is outside the standard's scope rather than a
   deviation from it.
2. Where the utility applies 62351, the natural alignment is TLS: CoAP over TLS (RFC 8323)
   keeps `Observe` and `since=<contiguous>` unchanged
   ([`decentralized_monitoring.md`](decentralized_monitoring.md) §7.1). Credentials should
   reuse the site PKI and cipher policy of 62351-9.

## `iot-flow` — IoT networks (CICIoT2023)

Defined for CICIoT2023, which the audit rejected (labels mark the capture session in four of seven categories: [`../results/ciciot2023_label_protocol_check.txt`](../results/ciciot2023_label_protocol_check.txt); [`RESULTADOS_ciciot2023.md`](RESULTADOS_ciciot2023.md)). The profile shape it stands for, **no traffic time and no attribution**, is exercised in Paper 2 by a controlled ablation instead: the corrected CICIDS2017 run under `it-flow-blind` (count tick, attribution `none`), see [`../results/cicids2017c_profile_ablation.txt`](../results/cicids2017c_profile_ablation.txt). The table below records the conditions that would apply to CICIoT2023.

| | |
|---|---|
| Monitored device | an IoT device or gateway |
| Local tick | **count-based** (`cadence.local.samples`): the CSV release carries no timestamps, so the tick is simulated and must be declared as such. Per the cadence sweep, a count-based tick should not be shorter than the traffic's natural burst |
| Traffic time | `none` |
| Attribution | device address when the capture provides it; the CSV release does not, so `source_node` is null |
| Label set | `ciciot2023-7` as released; only DDoS, DoS and Mirai have labels consistent with their content (Recon, Web, BruteForce, Spoofing mark the capture session: ARP spoofing rows 0.1 % ARP, ping sweep 0 % ICMP) |
| Conditions of use | `IAT` is excluded from the features: it behaves as a capture clock (narrow bands per class), and without it DoS F1 drops from 1.00 to 0.73; the raw prevalence is inverted (2.4 % benign) and must be rebalanced. No duplicates were found in the audited sample (10 of 169 files). No window triage: there is no time |

## `iot-device-minute` — consumer IoT devices (UNSW IoT attack traces)

The IoT domain of Paper 2 (audit and results: [`../results/unsw_iot_audit.txt`](../results/unsw_iot_audit.txt), [`RESULTADOS_unsw_iot.md`](RESULTADOS_unsw_iot.md)).

| | |
|---|---|
| Monitored device | one consumer IoT device (camera, smart plug, sensor); one agent per device |
| Local tick | time-based: one sample per device and minute (MUD-flow counters) |
| Traffic time | `frame` (minute timestamps) |
| Attribution | `none`: the counters belong to the **victim** device, not to the emitter, so an alarm names no source to isolate |
| Label set | `unsw-iot-3`: ArpSpoof, PingOfDeath, Smurf (N = 6). TCP SYN flood/reflection, UDP flood and SSDP/SNMP reflection are excluded: their attack shows in the affected counters in only 58–86 % of the annotated intervals, against 100 % for the three kept |
| Conditions of use | features = counters summed per direction × reach × protocol (41); device id and time feed the profile only; attack intervals kept whole across the split; small counts (180 attack minutes in the test) |

## `ics-modbus` — industrial control network (Electra Modbus)

The industrial domain of Paper 2 (audited: [`../results/electra_audit.txt`](../results/electra_audit.txt); results: [`RESULTADOS_electra.md`](RESULTADOS_electra.md)).

| | |
|---|---|
| Monitored device | a PLC, the SCADA host or a control-network segment |
| Local tick | time-based: packets carry a timestamp (µs) |
| Traffic time | `frame` (packet timestamps) |
| Attribution | source MAC address (`device_address`); every attack packet has the man-in-the-middle host as source or destination |
| Label set | `electra-4`: RECOGNITION, WRITE, FORCE_ERROR, RESPONSE (N = 8). MITM_UNALTERED, REPLAY and READ are excluded: their content is that of normal packets and only the MitM MAC tells them apart |
| Conditions of use | features = packet content only (`request, fc, error, address, data`); MACs, IPs and time feed the tick and attribution; FORCE_ERROR is small (1,129 packets) |

## `it-flow` — enterprise IT (CICIDS2017, corrected)

The attribution-and-time domain of Paper 2: the only one with real timestamps and source
addresses that point at the emitter (audited: [`../results/cicids2017c_audit.txt`](../results/cicids2017c_audit.txt)).

| | |
|---|---|
| Monitored device | a host or a network segment |
| Local tick | time-based: flows carry a timestamp |
| Traffic time | `frame` (flow timestamps) |
| Attribution | source IP of the flow |
| Label set | `cicids2017c-7`: DoS, DDoS, PortScan, BruteForce (FTP/SSH-Patator), Web (three web attacks merged), Bot, Infiltration; Heartbleed (11 flows) excluded |
| Conditions of use | `Infiltration - Portscan` merged into PortScan (91,253 scan flows identical under both labels); `Attempted` flows (11,979) labelled benign; `FWD/Bwd Init Win Bytes` excluded (host fingerprint: port-scan F1 0.86 → 0.62 without them on a time-ordered split); identifiers (IPs, ports, Timestamp, Flow ID) feed the tick and attribution only |

---

## What a new profile must state

1. what a monitored device is;
2. the local tick: a period, or a sample count if the data has no time;
3. whether the traffic carries timestamps;
4. where attribution comes from, or that it does not exist;
5. the label set, with an identifier;
6. whether the domain prescribes a transport, and what alignment follows.

Nothing in the core needs to change to add one.
