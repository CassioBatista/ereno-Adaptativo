# ReSIDS Monitor API

Observability / notification interface of the ReSIDS adaptation plane
([`fd/monitor.py`](../fd/monitor.py)). Two surfaces share **one event model**:

1. **REST pull** — an external monitor POLLS ReSIDS (`GET /events`, `/status`, `/health`).
   Machine-readable spec: [`openapi.yaml`](openapi.yaml).
2. **CoAP push** — ReSIDS PUSHES each event to an IoT hub (e.g., Magenta IoT Hub /
   ThingsBoard) as CoAP telemetry/attributes.

Authoritative payload schema: [`schemas/event.schema.json`](../schemas/event.schema.json).
Version: **1.1.0**. Events emitted: `architecture_change` (FL↔GL switch),
`node_failure` (a node identified as failing), and `intrusion_detected` (an attack
alarm — **notification only**, see §4).

---

## 1. Event model

| field | type | values | notes |
|---|---|---|---|
| `seq` | int | monotonic, ≥1 | unique; continues across runs |
| `ts` | string / int | ISO-8601 UTC (REST) · epoch-ms (CoAP) | timestamp |
| `type` | enum | `architecture_change`, `node_failure`, `intrusion_detected` | event kind |
| `round` | int | ≥0 | federation round |
| `from_mode` / `to_mode` | enum \| null | `federated`, `gossip` | architecture_change only |
| `reason` | enum \| null | `node_failure`, `recovery`, `scheduled` | cause |
| `failed_nodes` | int[] | node indices | authoritative on `node_failure` |
| `active_nodes` | int[] \| null | node indices | null = all active |
| `attack` | string \| null | attack class/type | `intrusion_detected` only |
| `k_votes` | int \| null | k in k-of-n | corroboration strength (`intrusion_detected`) |
| `n_flags` | int \| null | ≥0 | flagged samples/flows in the window (`intrusion_detected`) |
| `detector_nodes` | int[] | node indices | which specialists fired (`intrusion_detected`) |
| `source_node` | int \| null | node index | attributed emitter; **null when unavailable (typical v2)** |
| `confidence` | number \| null | score | optional (`intrusion_detected`) |
| `detail` | string \| null | free text | optional |

**Current state** (served by `/status`, pushed as CoAP attributes):
`current_mode`, `round`, `active_nodes[]`, `failed_nodes[]`, `last_seq`.

---

## 2. REST pull surface

Base: `http://<host>:<port>` (default `127.0.0.1:8722`). JSON, read-only.

| Method | Path | Purpose | Query | Success |
|---|---|---|---|---|
| GET | `/health` | liveness | — | `200` `{"status":"ok"}` |
| GET | `/status` | current state | — | `200` Status object |
| GET | `/events` | events (incremental) | `since`, `limit`, `type` | `200` Event[] |

`GET /events?since=<seq>` returns only events with `seq > since` (poll with the
last `seq` you saw). `type` filters by event kind.

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

Status codes: `200 OK`, `404 Not Found` (unknown path).

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
       "values":{"type":"intrusion_detected","round":14,"attack":"injection",
                 "k_votes":2,"n_flags":37,"detector_nodes":"[3,4]","source_node":null}}'
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
  emitter when known (GOOSE/SV source) and **null otherwise (typical in v2)**.
  This is a **notification** for the operator/monitor — ReSIDS does **NOT** isolate,
  quarantine, or otherwise actuate on it (see below). Attribution and any
  containment action are external / operator decisions (a Byzantine-aware,
  trust-driven containment loop is deferred to v3, see `docs/v3_trust_model.md`).
- **Delivery:** CoAP CON is retried on loss; the local JSONL audit trail
  (`EventStore`) is the durable fallback if the monitor is unreachable.
- **Ordering / idempotency:** `seq` is monotonic; consumers should de-duplicate by
  `seq`. Re-delivered CON messages carry the same `seq`.
- **Read-only:** the monitor observes; it never actuates the protected system
  (operator-gated, Level-2).

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

API version follows this document (**1.1.0**). Breaking changes to the event model
or endpoints bump the major version; additive fields bump the minor. The `type`
and `reason` enums may gain values in minor versions — consumers must ignore
unknown enum values gracefully. **v1.1.0** added the `intrusion_detected` event
type and its fields (`attack`, `k_votes`, `n_flags`, `detector_nodes`,
`source_node`, `confidence`) — additive, backward-compatible.

## 7. Files

- [`schemas/event.schema.json`](../schemas/event.schema.json) — authoritative payload schema (JSON Schema 2020-12).
- [`openapi.yaml`](openapi.yaml) — REST endpoints (OpenAPI 3).
- [`fd/monitor.py`](../fd/monitor.py) — reference implementation.
