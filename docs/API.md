# ReSIDS Monitor API

Observability / notification interface of the ReSIDS adaptation plane
([`fd/monitor.py`](../fd/monitor.py)).

> **Who is "the monitor".** The architecture has two roles only: **Disaster-FD** — one
> monitor process per node and per server, federated by region — and the **ReSIDS agent**.
> "Monitor" in this API always means a Disaster-FD monitor process; there is no external
> monitor or supervisor, and the monitor in this repository is a simulation of it.
>
> **API 1.9.0 changes the authority.** Disaster-FD detects failures and publishes a trust
> level (TL); each agent decides its learning mode (`local`, `gossip`, `federated`) from the
> TL of its colocated monitor, and records the switch with `decided_by: agent`. The TL
> input and the policy are in [`TRUST_LEVEL.md`](TRUST_LEVEL.md). The intrusion events
> (`intrusion_detected`, `node_isolated`) and the command surface below are deferred to
> future work with the Byzantine model. The rest of this document describes 1.7.0, whose
> event surface 1.9.0 keeps.
>
> **This document is the OBSERVATION half.** Disaster-FD also *decides*: under the
> authority rule ([`decentralized_monitoring.md`](decentralized_monitoring.md) §1.1) every
> transition is its call — the fail-fast FL→GL by the node-local instance, everything else
> by the federated one. The agent acts alone only as a watchdog when its own local FD
> process is down. The commands are specified in [`COMMANDS.md`](COMMANDS.md).

Two surfaces share **one event model**:

1. **REST pull** — the Disaster-FD monitor POLLS the agent (`GET /events`, `/status`, `/health`).
   Machine-readable spec: [`openapi.yaml`](openapi.yaml) (OpenAPI **3.1**, which `$ref`s
   the event schema instead of restating it — they drifted apart once, over
   `detector_nodes`, and the `$ref` makes that impossible).
2. **CoAP push** — ReSIDS PUSHES each event to an IoT hub (e.g., Magenta IoT Hub /
   ThingsBoard) as CoAP telemetry/attributes.

Authoritative payload schema: [`schemas/event.schema.json`](../schemas/event.schema.json).
Version: **1.7.0**. Five events are emitted: `architecture_change` (FL↔GL switch),
`node_failure` (a node identified as failing), `node_recovery` (a node back in the
membership), `node_isolated` (a node removed by command) and `intrusion_detected` (an
attack alarm — **notification only**, see §4).

Of these, only `architecture_change` and `node_isolated` are **actions**; the other three
are observations. The distinction is carried by `decided_by`, and §6 records when each
piece arrived.

---

## 1. Event model

| field | type | values | notes |
|---|---|---|---|
| `seq` | int | monotonic, ≥1 | unique; continues across runs |
| `ts` | string / int | ISO-8601 UTC (REST) · epoch-ms (CoAP) | timestamp |
| `type` | enum | `architecture_change`, `node_failure`, `node_recovery`, `node_isolated`, `intrusion_detected` | event kind |
| `round` | int | ≥0 | **local** round: tick of the node-local Disaster-FD (counts `detect_lag`, watchdog D) |
| `fed_round` | int \| null | ≥0 | **federated** round: tick of the regional Disaster-FD (counts `dwell`, `cmd_latency`) |
| `window_start` / `window_end` | string \| null | ISO-8601 UTC | the two local ticks delimiting the window — **not** inferred from traffic; distinct from `ts`, the emission instant |
| `traffic_time_start` / `_end` | string \| null | ISO-8601 UTC | optional, per profile: first and last traffic timestamp in the window (e.g. GOOSE/SV frames in `iec61850-goose-sv`); absent where the data has no time |
| `window_samples` | int \| null | ≥0 | samples scored between the two ticks — denominator for `n_flags` |
| `mode` | enum \| null | `federated`, `gossip` | aggregation mode in effect, so the monitor never infers state |
| `n_active` | int \| null | ≥0 | `len(active_nodes)` |
| `from_mode` / `to_mode` | enum \| null | `federated`, `gossip` | architecture_change only |
| `reason` | enum \| null | `node_failure`, `recovery`, `scheduled`, `intrusion`, `autonomous_fallback` (`commanded` kept for pre-1.5.0 streams) | the **cause**; who decided is in `decided_by`. `autonomous_fallback` marks the agent's watchdog ([`COMMANDS.md`](COMMANDS.md) §4.1) |
| `failed_nodes` | int[] | node indices | authoritative on `node_failure` |
| `recovered_nodes` | int[] | node indices | authoritative on `node_recovery` |
| `active_nodes` | int[] \| null | node indices | null = all active |
| `attack` | string \| null | label | from the **label set of the domain profile** in force (`/status.profile.label_set`; [`PROFILES.md`](PROFILES.md)) — the core fixes no vocabulary |
| `k_votes` | int \| null | k in k-of-n | corroboration **strength** (`intrusion_detected`) |
| `n_flags` | int \| null | ≥0 | alarm **volume**: flagged samples in the window (`intrusion_detected`) |
| `detector_specialists` | int[] | specialist indices | which **boosters** fired — not nodes: every node holds the same diffused union and reaches the same verdict (`intrusion_detected`) |
| `source_node` | int \| null | node index | attributed emitter; where the identity comes from is fixed by the **profile** (`/status.profile.attribution`); **null when unavailable (typical v2)** |
| `confidence` | number \| null | score | optional (`intrusion_detected`) |
| `decided_by` | enum \| null | `monitor`, `autonomous`, `operator` | who took the **action**. Present only on `architecture_change` and `node_isolated`; observations omit it. `monitor` = a Disaster-FD monitor process (node-local for the fail-fast FL→GL, federated otherwise); `autonomous` = the agent's watchdog FL→GL, and nothing else |
| `command_id` | string \| null | UUID | the command that caused the action ([`COMMANDS.md`](COMMANDS.md)). Null on the watchdog action, which no command caused |
| `detail` | string \| null | free text | optional |

**Current state** (served by `/status`, pushed as CoAP attributes):
`current_mode`, `round`, `active_nodes[]`, `failed_nodes[]`, `isolated_nodes[]`,
`switch_pending`, `authority_epoch`, `evidence`, `last_seq`, `last_cmd_seq`. The last five
exist for the command surface: a commander that cannot read `evidence` cannot know whether
a GL→FL return will be accepted, and would be reduced to guess-and-retry. `profile`
(`name`, `label_set`, `attribution`, `traffic_time`) states the domain binding, so a
consumer knows which vocabulary `attack` uses and whether `source_node` can ever be set.

---

## 2. REST pull surface

Base: `http://<host>:<port>` (default `127.0.0.1:8722`). JSON, read-only.

| Method | Path | Purpose | Query | Success |
|---|---|---|---|---|
| GET | `/health` | liveness | — | `200` `{"status":"ok"}` |
| GET | `/status` | current state | — | `200` Status object |
| GET | `/events` | events (incremental) | `since`, `limit`, `type` | `200` Event[] |

`GET /events?since=<seq>` returns only events with `seq > since`. `type` filters by
event kind.

**Poll with the highest CONTIGUOUS `seq`, not the highest one seen.** The contiguous
`seq` is the largest *c* such that every event up to *c* has been received. Using the
highest seq seen is wrong whenever delivery can lose messages (CoAP push over UDP): once
a later notification arrives, the tail is up to date while an earlier event is still
missing, and the hole becomes invisible. Measured on the replay harness
([`scripts/monitor_interaction_sim.py`](../scripts/monitor_interaction_sim.py)) at 20%
notification loss: **33 events permanently lost with the tail test, 0 with the
contiguous one**.

This single request also subsumes liveness: an **empty** answer means *alive*, *no gap*
and *nothing to recover*; a non-empty one reveals the gap and fills it in the same round
trip. Polling `/status` first to compare `last_seq`, then fetching, costs two round trips
for the same result — so `/status` is for **state bootstrap** (monitor start-up, or after
a long outage), not for periodic probing. See
[`decentralized_monitoring.md`](decentralized_monitoring.md) §4.

**Example**
```
GET /events?since=1&type=architecture_change
200 OK
[
  {"seq":2,"ts":"2026-09-04T11:18:12.100000+00:00","type":"architecture_change",
   "round":11,"from_mode":"federated","to_mode":"gossip","reason":"node_failure",
   "failed_nodes":[7],"active_nodes":[0,1,2,3,4,5,6,8,9],"detail":null}
]
```

### 2.1 Behavioural contract

What a consumer may rely on, and what it must handle. These are the guarantees the
machine-readable spec cannot express; the conformance script
([`scripts/validate_events.py`](../scripts/validate_events.py)) checks the ones that are
checkable from a stream.

| Property | Guarantee |
|---|---|
| **Ordering** | events are returned by `seq` **ascending** |
| **`since` semantics** | **exclusive**: returns `seq > since` |
| **`seq`** | monotonic, starts at 1, **no gaps**, and continues across process restarts |
| **Gap detection** | poll with the highest **contiguous** seq (see above); `last_seq` in `/status` tells you how far the emitter has gone |
| **Idempotency** | re-requesting a range returns identical events; duplicates are the consumer's to drop, by `seq` |
| **Pagination** | `limit` caps the answer (max 10000). A full page means **more may exist**: poll again with `since` = last `seq` returned |
| **Retention** | bounded. `/status.retained_from_seq` is the oldest `seq` still served; a `since` below it returns **`410 Gone`**, never a silently short answer |
| **Errors** | `400` malformed parameter · `404` unknown path · `410` beyond retention |
| **Versioning** | `/status.api_version` carries the implemented version, for in-band discovery |
| **Authentication** | **none** on this surface; it binds to `127.0.0.1` by default. Exposing it beyond loopback requires a reverse proxy with TLS and authentication — the server itself does not authenticate. **This stance is specific to reading**: the command surface requires mutual authentication and a per-command signature ([`COMMANDS.md`](COMMANDS.md) §6) |
| **Read-only** | no endpoint *here* mutates state. Actuation lives on the separate, specified-but-unimplemented command surface ([`COMMANDS.md`](COMMANDS.md)) |

**Retention matters more than it looks.** The in-memory store keeps the last
`keep_in_memory` events (10 000 by default) while the JSONL trail keeps everything. A
monitor that was offline long enough will ask for a `since` that is gone: it must get
`410` and resync from `/status`, not a short array it would mistake for "nothing
happened". The complete history lives in the audit trail, not on this surface.

Status codes: `200 OK`, `400 Bad Request`, `404 Not Found` (unknown path), `410 Gone`.

> **Implementation note.** `fd/monitor.py` today returns `200` with a short array instead
> of `410`, and `/status` carries neither `api_version` nor `retained_from_seq`. The
> contract above is what the spec requires; closing that gap is pending work.

---

## 3. CoAP push surface

ReSIDS acts as a **CoAP client / IoT device**, pushing to a ThingsBoard-style hub.
Transport: **UDP**, port **5683** (plain) / **5684** (DTLS). Content-Format:
`application/json` (50) or `application/cbor` (60). Messages are **Confirmable
(CON)** — reliable delivery with retransmission.

**Auth:** access token in path (`/api/v1/$ACCESS_TOKEN/...`) or X.509. Keep the
token in configuration/environment, never in source.

| Method | Path | Purpose | Body | ACK |
|---|---|---|---|---|
| POST | `/api/v1/$TOKEN/telemetry` | push an event (time-series) | `{"ts":<ms>,"values":{...event...}}` | `2.01 Created` |
| POST | `/api/v1/$TOKEN/attributes` | push current state | `{...state...}` | `2.04 Changed` |

`values` carries the flattened event fields; list-valued fields (`failed_nodes`,
`active_nodes`) are sent as JSON strings plus a scalar count (`n_failed`) for easy
dashboards/alarms.

**Example (event as telemetry)**
```
coap-client -m post coap://iothub.magenta.at/api/v1/$TOKEN/telemetry \
  -e '{"ts":1757800000000,
       "values":{"type":"node_failure","round":10,"failed_nodes":"[7]","n_failed":1}}'
→ 2.01 Created
```

**Example (intrusion alarm as telemetry — notification only)**
```
coap-client -m post coap://iothub.magenta.at/api/v1/$TOKEN/telemetry \
  -e '{"ts":1757800020000,
       "values":{"type":"intrusion_detected","round":14,"mode":"gossip","attack":"injection",
                 "window_start":"2026-01-01T00:00:24.000420+00:00",
                 "window_end":"2026-01-01T00:00:25.000420+00:00","window_samples":4758,
                 "k_votes":2,"n_flags":37,"detector_specialists":"[3,4]","source_node":null}}'
→ 2.01 Created
```

**Example (state as attributes)**
```
POST /api/v1/$TOKEN/attributes
{"current_mode":"gossip","round":11,"active_nodes":"[0,1,2,3,4,5,6,8,9]","failed_nodes":"[7]"}
→ 2.04 Changed
```

Status codes: `2.01 Created`, `2.04 Changed`, `4.00 Bad Request`,
`4.01 Unauthorized` (bad token), `4.04 Not Found`.

---

## 4. Semantics

- **When emitted:** `architecture_change` on every committed FL↔GL switch;
  `node_failure` when a node is detected as failing. Detection latency is one round
  (detected at the end of round *r*, switch takes effect at *r+1*).
- **`intrusion_detected`:** emitted when the fused (k-of-n) detector raises an attack
  alarm, **aggregated per round/window** (one alarm per attack type per round, with
  `n_flags` as the volume) — not one message per sample, to avoid flooding.
  `k_votes` is the k-of-n corroboration strength; `source_node` is the attributed
  emitter when the domain profile provides attribution (e.g. the GOOSE/SV publisher in
  `iec61850-goose-sv`; see [`PROFILES.md`](PROFILES.md)) and **null otherwise (typical
  in v2)**. `attack` is a label from the profile's label set.
  This is a **notification** for Disaster-FD — the agent does **NOT** isolate,
  quarantine, or otherwise actuate on it. Isolation is decided by the federated
  Disaster-FD; physical containment is a network action outside the IDS (a
  Byzantine-aware, trust-driven containment loop is deferred to v3,
  see `docs/v3_trust_model.md`).
- **Delivery:** CoAP CON is retried on loss; the local JSONL audit trail
  (`EventStore`) is the durable fallback if the monitor is unreachable.
- **Ordering / idempotency:** `seq` is monotonic; consumers should de-duplicate by
  `seq`. Re-delivered CON messages carry the same `seq`.
- **Read-only:** this surface only observes. Actuation goes through the command surface,
  and neither ever touches the protected process bus.

---

## 5. Configuration

```yaml
monitor:
  enabled: true
  serve: true                 # REST pull endpoint
  host: 127.0.0.1
  port: 8722
  trail: results/monitor_events.jsonl
  coap:                       # optional CoAP push
    host: iothub.magenta.at
    token: ${RESIDS_IOTHUB_TOKEN}   # from env, never hardcoded
    port: 5684
    dtls: true
    confirmable: true
    content_format: json      # json | cbor
```

## 6. Versioning

API version follows this document (**1.7.0**). Breaking changes to the event model
or endpoints bump the major version; additive fields bump the minor. The `type`
and `reason` enums may gain values in minor versions — consumers must ignore
unknown enum values gracefully.

**v1.9.0** splits the roles by competence ([`TRUST_LEVEL.md`](TRUST_LEVEL.md)): Disaster-FD
detects failures and publishes a **trust level** per observation
(`schemas/trust_level.schema.json`, new); the agent decides its mode from the TL — bands
TL < 30 local, 30 ≤ TL < 50 gossip, TL ≥ 50 federated, a settle time of 5 s, and a fallback
to local when no TL arrives for 15 s. Schema changes are additive: `local` in the mode
enums, `reason` `trust_level` and `tl_stale`, `decided_by` `agent`, and the fields
`trust_level`, `tl_bands`, `tl_ts`, `fd_monitor`; every older stream still validates. What
changes in meaning is the decider: in a 1.9.0 stream every transition is `decided_by:
agent`, and `monitor` / `autonomous` do not appear. `intrusion_detected`, `node_isolated`
and the command surface ([`COMMANDS.md`](COMMANDS.md)) are not emitted or served; intrusion
notification and isolation return with the Byzantine-resilient version (future work).
There is no 1.8.0.

**v1.7.0** makes the **domain binding explicit** ([`PROFILES.md`](PROFILES.md)): a `profile`
(name, label set, attribution source, traffic time) in the scenario, the manifest and
`/status`. `attack` is a label from the profile's label set and `source_node` comes from the
profile's attribution source (always null when it is `none`). Additive; the scenario
streams are unchanged. The core documentation no longer assumes IEC 61850; that domain is
the `iec61850-goose-sv` profile, where its IEC 62351 alignment now lives.

**v1.6.0** makes **Disaster-FD the time reference**, in two cadences
([`decentralized_monitoring.md`](decentralized_monitoring.md) §1.2). `round` becomes the
local tick (node-local Disaster-FD) and the new `fed_round` the federated one; the window is
what the agent scored between two local ticks and is no longer derived from traffic
timestamps, which move to the optional, profile-dependent `traffic_time_start/end`. All
additive; with a 1-s local tick and a federated tick every round, every pre-existing field
of every event is unchanged (verified on the three scenario streams). The scenario format
replaces `window_seconds` with `cadence` (the old field is still accepted) and gains
`traffic_time` and `isolation.escalate_fraction`.

**v1.5.0** moves authority to **Disaster-FD** (decision of 2026-10-02). The schema is
unchanged; what changes is **who is recorded as deciding the fail-fast FL→GL**: it is now
commanded by the node-local Disaster-FD monitor (`decided_by: monitor`, `reason:
node_failure`), and `decided_by: autonomous` is reserved for the agent's watchdog
(`reason: autonomous_fallback`) when that local process is down. A consumer that read
`autonomous` as "the fail-fast transition" must update: in normal operation it no longer
appears. `reason: commanded` is no longer emitted (the decider is in `decided_by`). The
scenario format gains `timing.watchdog` and `local_fd_available`.

**v1.4.0** adds the **command surface** ([`COMMANDS.md`](COMMANDS.md)): the `/commands`
endpoints, `schemas/command.schema.json`, `schemas/command_result.schema.json`, the event
field `command_id`, the `reason` values `commanded` and `autonomous_fallback`, and the
`/status` fields `isolated_nodes`, `switch_pending`, `authority_epoch`, `evidence` and
`last_cmd_seq` — all additive. Nothing on the event surface changes meaning.

**v1.3.0** added the `node_isolated` type and `decided_by`, and restricted `decided_by` to
events that *are* an action. That restriction is the machine-checkable form of the
authority rule; [`scripts/validate_events.py`](../scripts/validate_events.py) rejects a
stream that claims otherwise (since 1.5.0: `autonomous` only on the watchdog FL→GL).

**v1.2.0** added the `node_recovery` type; the traffic window (`window_start`,
`window_end`, `window_samples`); explicit state on every event (`mode`, `n_active`); and
**renamed** `detector_nodes` to `detector_specialists`.

**v1.1.0** added the `intrusion_detected` event type and its fields (`attack`,
`k_votes`, `n_flags`, `detector_nodes`, `source_node`, `confidence`) — additive,
backward-compatible.

**v1.2.0** added the `node_recovery` type and the fields `recovered_nodes`, `mode`,
`n_active`, `window_start`, `window_end`, `window_samples` — additive — and **renamed
`detector_nodes` to `detector_specialists`**, which is *not* backward-compatible for
consumers that read that field by name. The rename is deliberate: those indices identify
**boosters**, not nodes, and the old name suggested a per-node corroboration the
architecture does not have.

## 7. Files

- [`schemas/event.schema.json`](../schemas/event.schema.json) — authoritative payload schema (JSON Schema 2020-12).
- [`schemas/trust_level.schema.json`](../schemas/trust_level.schema.json) — Disaster-FD trust-level observation (1.9.0).
- [`TRUST_LEVEL.md`](TRUST_LEVEL.md) — the 1.9.0 input surface and agent policy.
- [`scripts/tl_policy_replay.py`](../scripts/tl_policy_replay.py) — replay of the policy over a Disaster-FD run.
- [`openapi.yaml`](openapi.yaml) — REST endpoints, events and commands (OpenAPI 3.1; 1.9.0, `/commands` deprecated).
- [`COMMANDS.md`](COMMANDS.md) — the command surface: the half that lets the monitor decide.
- [`fd/monitor.py`](../fd/monitor.py) — reference implementation (events only).
- [`scripts/validate_events.py`](../scripts/validate_events.py) — stream conformance, including the authority rule.
