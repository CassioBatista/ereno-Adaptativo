# ReSIDS command surface (Disaster-FD → agent)

**Status: specification. NOT implemented.** `fd/monitor.py` serves the read-only event
surface only ([`API.md`](API.md)); nothing here exists in code yet. §9 states the gap
precisely.

**Who commands.** The architecture has two roles only: **Disaster-FD** — one monitor
process per node and per server, federated by region — and the **ReSIDS agent**. Throughout
this document, "monitor" means **a Disaster-FD monitor process**; there is no external
monitor or supervisor. This is the half of the interface through which Disaster-FD
exercises the authority the rule of
[`decentralized_monitoring.md`](decentralized_monitoring.md) §1.1 gives it — over **every**
transition:

| Situation | Decides | Surface |
|---|---|---|
| node inactive while in FL → GL | **node-local Disaster-FD monitor** (local evidence) | `set_mode {to: gossip}` |
| further node loss while in GL | federated Disaster-FD | `set_mode` / `isolate_node` |
| node with attributed intrusion | federated Disaster-FD | `isolate_node` |
| **GL → FL return** | **federated Disaster-FD** | `set_mode` — and *only* this |
| node-local Disaster-FD process down | agent watchdog (FL→GL only) | none — no command exists to receive |

Every transition goes through a command, except the watchdog, which by definition fires when
there is no detector left to command.

Transport: the decision between CoAP and REST is still open
([`decentralized_monitoring.md`](decentralized_monitoring.md), opening note). §8 gives
the binding for both. Everything in §§1–7 is transport-independent.

Authoritative payloads: [`schemas/command.schema.json`](../schemas/command.schema.json)
and [`schemas/command_result.schema.json`](../schemas/command_result.schema.json).
Machine-readable endpoints: [`openapi.yaml`](openapi.yaml). API version **1.6.0**.

---

## 1. Three commands

| Command | Params | Effect |
|---|---|---|
| `set_mode` | `to` (`federated`\|`gossip`), `reason` | commit a mode transition at the next round boundary |
| `isolate_node` | `node`, `reason`, `evidence_seq[]` | remove a node from the federation / booster pool |
| `readmit_node` | `node`, `reason` | put it back |

That is the whole surface. §7 lists what is deliberately missing, which matters as much
as what is here.

The streams in `conf/scenarios/` exercise it, and between them they cover every commanded
action the simulation produces — 8 commands for 8 Disaster-FD-decided events:

| Stream | Round | Event | Command it implies |
|---|---|---|---|
| availability | 102 | `architecture_change` FL→GL | `set_mode {to: gossip, reason: node_failure}` — node-local FD |
| availability | 606 | `architecture_change` GL→FL | `set_mode {to: federated, reason: recovery}` — federated FD |
| intrusion | 8 | `node_isolated` node 11 | `isolate_node {node: 11, reason: intrusion}` |
| intrusion | 21 | `node_isolated` node 9 | `isolate_node {node: 9, reason: intrusion}` |
| intrusion | 21 | `architecture_change` FL→GL | `set_mode {to: gossip, reason: intrusion}` |
| intrusion | 23 | `node_isolated` node 10 | `isolate_node {node: 10, reason: intrusion}` |
| intrusion | 39 | `node_isolated` node 12 | `isolate_node {node: 12, reason: intrusion}` |
| intrusion | 48 | `node_isolated` node 8 | `isolate_node {node: 8, reason: intrusion}` |

The third stream, `fd_watchdog`, has the node-local FD down: its FL→GL (round 105) is the
agent's watchdog and implies **no** command; its GL→FL (round 606) is commanded as above.

`scripts/commands_from_stream.py` reconstructs exactly this from the event streams and
validates it, so the claim that the surface is sufficient is checked rather than
asserted. Note the FL→GL at intrusion round 21: commanded by the federated Disaster-FD,
driven by intrusion rather than inactivity.

`readmit_node` is exercised by no scenario yet. It is in the surface regardless, because
the false-positive baseline makes wrongful isolation expected, not hypothetical: 252 of
724 benign windows raise an alarm (34.8%). A surface that can isolate but not readmit
forces a node restart to undo a mistake the IDS made.

---

## 2. Anatomy of a command

```json
{
  "command_id": "f1c0a9de-5b2e-4a77-9c31-2b8d6e40a111",
  "issued_at": "2026-01-01T00:10:07.000420+00:00",
  "not_after": "2026-01-01T00:10:37.000420+00:00",
  "authority_epoch": 3,
  "issuer": {"kind": "monitor", "id": "mon-a"},
  "command": "isolate_node",
  "params": {"node": 11, "reason": "intrusion", "evidence_seq": [4, 7, 9]},
  "expect": {"mode": "federated", "n_active": 14, "seq": 9}
}
```

Four envelope fields carry the weight, and each answers a failure this project already
measured or argued:

**`command_id`** — a client-generated UUID, and **the** idempotency key. Execution is **at
most once**: re-submitting the same id returns the stored outcome with `replayed: true`
and executes nothing. This is not defensive decoration. The event surface's contiguous-seq
rule exists because delivery loses messages; the same channel duplicates them, and a
duplicated isolation or mode switch is not idempotent by nature. The key makes it so.

**`authority_epoch`** — a monotonic counter of *which* monitor holds authority. An agent
refuses anything below the highest epoch it has accepted. Without it, two monitors
commanding during a failover are indistinguishable from legitimate traffic, and a revoked
monitor keeps its powers indefinitely.

**`not_after`** — expiry. A command captured on the wire and replayed an hour later is
refused. The reference policy caps acceptance at `issued_at + 60 s` even when the field is
absent, so omitting it cannot buy unlimited replay.

**`expect`** — the state the monitor *believed* was in effect when it decided. If reality
differs, the command is refused with `409` and the current state, and **nothing is
applied**. This is the same argument as the contiguous seq, one layer up: the monitor
decides on a polled view that is already in the past, and an actuation surface must not let
a stale view act. `expect.seq` is the monitor's highest contiguous event seq — if the agent
has emitted events the monitor has not read, the decision was taken without them.

`expect` is optional. Omitting it is a deliberate statement that the command is
unconditional (a regional override by the federated Disaster-FD, typically), not an
oversight.

A fifth field, **`intent_id`**, appears when one decision is fanned out across instances: a
mode switch is federation-wide while commands are per-instance, so one decision becomes 15
commands, each with its own `command_id` and all sharing the `intent_id`. It was added after
the fact — specifying the surface against a single endpoint did not reveal the need; building
the 15-instance monitor did ([`MULTI_INSTANCE.md`](MULTI_INSTANCE.md) §4), along with the
asymmetric unanimity rule that governs such a fan-out: a partial FL→GL is safe, a partial
GL→FL splits the federation.

---

## 3. Outcomes

| Outcome | HTTP | Meaning |
|---|---|---|
| `applied` | `202 Accepted` | accepted; commits at the **next round boundary**, so it is *not yet* in effect when the answer arrives |
| `no_op` | `200 OK` | the requested state was already the current one — nothing done, nothing wrong |
| `rejected` | `4xx` | nothing applied; `error.code` says why |

| `error.code` | HTTP | When |
|---|---|---|
| `malformed` | 400 | fails the schema |
| `unauthenticated` | 401 | no valid channel identity |
| `bad_signature` | 403 | per-command signature invalid (§6.2) |
| `stale_epoch` | 403 | `authority_epoch` below the highest accepted |
| `expired` | 403 | past `not_after`, or past the 60 s cap |
| `expectation_failed` | 409 | `expect` does not match reality |
| `evidence_not_met` | 412 | the agent's own safety condition is unmet (§4.2) |
| `membership_floor` | 412 | the isolation would drop below `min_nodes` |
| `unknown_node` | 422 | no such node index |
| `too_soon` | 429 | flap protection (§4.3) |
| `not_ready` | 503 | agent starting or shutting down |

### 3.1 The response is not the commit record

A response can be lost. The authoritative record that a command took effect is **the
event** it produced, carrying `decided_by` and `command_id`. A monitor should:

1. send the command;
2. on any doubt — timeout, 5xx, lost connection — **reissue with the same
   `command_id`** (safe, by construction);
3. consider it committed only when the correlated event appears in `GET /events`.

Rejected commands produce **no event**. That is why every answered command, accepted or
not, gets a `cmd_seq` in the command log (§5): otherwise a refused actuation attempt —
a stale view, a bad epoch, a failed override — would leave no trace anywhere.

---

## 4. Semantics that are not negotiable

### 4.1 The watchdog D is not commandable

The fail-fast FL→GL is commanded by the **node-local** Disaster-FD monitor from local
evidence. Alongside it, the agent runs a watchdog **D** on that local process
([`monitor_commanded_switch.md`](monitor_commanded_switch.md)):

- the node-local FD commands `set_mode {to: gossip}` → commit, `decided_by: monitor`,
  `reason: node_failure`, `command_id` set;
- the node-local FD process stops answering for D rounds → the agent commits FL→GL on its
  own: `decided_by: autonomous`, `reason: autonomous_fallback`, no `command_id`;
- a command arriving *after* the watchdog fired → `200 no_op`: the state Disaster-FD wanted
  is the current state. Not an error.

**There is no command to cancel, hold or extend D, and there will not be one.** A commander
able to postpone it is a commander able to keep a node in FL with a dead aggregator and a
dead detector — the single point of failure GL exists to survive, reintroduced through the
control channel. D is configuration (`watchdog_rounds`), changed by redeploying policy, not
by a command in flight.

### 4.2 Disaster-FD decides *whether*; the agent verifies *whether it is possible*

Authority is not omnipotence. Two commands meet a local safety condition:

**`set_mode {to: federated}`** — the GL→FL return. The agent refuses with
`evidence_not_met` unless membership has been full for `dwell` rounds. Returning to FL
means **re-accepting the hub**, the point of failure and of trust GL exists to survive;
doing that while nodes are still missing is the one transition that can make things
worse. The dwell/unanimity gates are *evidence the agent checks against the command*, not
a commit rule of its own.

**`isolate_node`** — refused with `membership_floor` if it would take the federation below
`min_nodes` (default 3). The floor is where the measurement stops supporting the claim:
recall holds at 100% from 14 nodes down to 3 because the retained union keeps every
booster, and below that nothing was measured.

`force: true` overrides both. It is recorded in the event's `detail`, it never overrides
authentication, epoch, expiry or idempotency, and it exists because the federated
Disaster-FD, with a regional view, sometimes knows more than one agent's local evidence does.

An accepted isolation returns a **warning**, not a refusal, when it removes the last
holder of a specialist: the pool loses the ability to *type* that attack precisely, while
recall is unaffected. The commander should know; the decision is still theirs.

### 4.3 Commands commit on a round boundary, and not too often

A command accepted during round *r* takes effect at *r+1*. A mode change applied halfway
through an aggregation would split the round between two protocols. `effective_round` in
the result says when.

A commanded mode switch within `min_switch_interval` of the last one is refused with
`too_soon`. Rationale: booster diffusion is the control plane's only real cost, and a
commander oscillating FL⇄GL would starve it while every individual command looked
legitimate.

### 4.4 Isolation does not stop traffic

Worth stating on the command surface, because this is where the misunderstanding would be
acted upon: `isolate_node` removes a node from the federation and from the booster pool.
It does **not** disconnect it from the process bus. An isolated node keeps publishing
GOOSE/SV, keeps being observed, and keeps producing alarms — the event stream shows
exactly that, with `source_node` still attributed after isolation. Containment is a
**network** action (VLAN, port, ACL) that ReSIDS neither performs nor claims to. The
result carries this as a `warning` on every accepted isolation.

---

## 5. Endpoints

| Method | Path | Purpose | Success |
|---|---|---|---|
| POST | `/commands` | issue a command | `202` applied · `200` no_op/replay |
| GET | `/commands/{command_id}` | the stored outcome of one command | `200` · `404` |
| GET | `/commands?since=<cmd_seq>` | the command log, incremental | `200` |

`GET /commands` mirrors `GET /events` exactly — `since` exclusive, ascending, gap-free
`cmd_seq`, poll with the highest **contiguous** value — so a second monitor, an auditor,
or the same monitor after a restart can reconstruct what was commanded, including what was
refused. The reconciliation argument from [`API.md`](API.md) §2.1 applies unchanged, and
for the same measured reason.

`/status` gains the fields a commander needs to decide (§9 lists them as unimplemented):

```json
{
  "api_version": "1.6.0",
  "current_mode": "gossip",
  "round": 606,
  "active_nodes": [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
  "isolated_nodes": [],
  "switch_pending": null,
  "authority_epoch": 3,
  "evidence": {"membership_full_for_rounds": 4, "dwell_required": 3},
  "last_seq": 205,
  "last_cmd_seq": 6,
  "retained_from_seq": 1
}
```

`evidence` is the operative addition: without it the monitor cannot know whether a GL→FL
return will be accepted, and would be reduced to guessing and retrying.

---

## 6. Security

The read-only surface is unauthenticated and bound to loopback
([`API.md`](API.md) §2.1). **That stance does not extend here, and nothing in this
document should be read as permitting it.** This surface changes the behaviour of a
protection system.

### 6.1 Channel

Two channels differ in exposure. Between the agent and its **own** node-local Disaster-FD
process the channel is local to the host (IPC); it never crosses the network, and
authentication can be correspondingly lighter. Commands from the **federated** Disaster-FD
cross the network and get everything below.

For those, mutual authentication is mandatory: mTLS with TLS 1.3 on the REST binding, DTLS 1.3 with
raw public keys or PSK on the CoAP binding. Unauthenticated requests get `401` before
anything is parsed. This aligns with IEC 62351-3 (TLS for TCP/IP profiles) — see
[`decentralized_monitoring.md`](decentralized_monitoring.md) §7.1 on why the standard
offers no DTLS profile and what that implies for the CoAP option.

### 6.2 Per-command signature

Channel authentication proves the **connection**, not the **decision**. Put a
TLS-terminating proxy or an IoT hub in the path — which the CoAP binding explicitly does
— and that intermediary becomes an implicit commander, with an audit trail unable to tell
the difference. A detached Ed25519 signature over the canonical serialization of the
command (every member except `signature`, keys sorted, no insignificant whitespace) keeps
authority provable from the log alone, after the fact, without trusting the transport.

### 6.3 The threat this surface creates

It did not exist while the monitor was read-only, and it should be stated plainly: a
compromised commander can force GL (degrading efficiency), force FL (re-accepting a hub
it may control), isolate honest nodes one by one, or readmit a node it had isolated.
The mitigations above are bounds, not solutions — epoch for failover, floor for
membership, flap limit for oscillation, signature for attribution, and the deliberate
absence of a deadline veto so the worst case stays *degraded*, never *disarmed*.
Byzantine-aware trust of the commander itself is deferred to v3
(`docs/v3_trust_model.md`).

---

## 7. What is deliberately absent

The surface is small on purpose. Each of these was considered and excluded:

| Not offered | Why |
|---|---|
| cancel / hold / extend the deadline D | equivalent to `D → ∞`: reintroduces the single point of failure GL exists to survive (§4.1) |
| upload or replace a booster | the diffusion plane would become a command surface, and a commander able to inject a model is a commander able to poison detection |
| change `k`, thresholds or the feature set at runtime | these are the detection integrity parameters; remote mutation turns a control channel into a way to blind the IDS silently |
| supply labels or training data | a commander able to label traffic can teach the system that an attack is benign. Training data stays out of band |
| a third mode, or topology edits | the mode set is `{federated, gossip}`; overlay structure is not runtime policy |
| acknowledge or suppress alarms | alarm triage belongs to the monitor's own state, not to the agent's. Suppression at the agent would silently drop evidence from the trail |
| block traffic / quarantine | ReSIDS does not touch the process bus (§4.4). Claiming it through this API would be a false promise with safety consequences |

---

## 8. Transport bindings

Both are specified because the transport decision is open. The command model, the
idempotency key, the epoch and the authority rule are identical in each.

### REST (TLS 1.3, mTLS)

As §5. `POST /commands`, JSON body per
[`command.schema.json`](../schemas/command.schema.json).

### CoAP (DTLS 1.3)

ReSIDS is the device; the monitor commands via server-side RPC. The agent `Observe`s
`/api/v1/$TOKEN/rpc`; each RPC carries the command object as `params`.

| Direction | CoAP | Payload |
|---|---|---|
| monitor → ReSIDS | RPC (CON) | `{"method": "isolate_node", "params": {<command object>}}` |
| ReSIDS → monitor | RPC response | `{<command result object>}` |
| ReSIDS → monitor | POST `/telemetry` | the resulting event, with `command_id` |

Two CoAP-specific notes. The idempotency key must be the command object's `command_id`,
**not** the CoAP message id or token: CON retransmission reuses neither across a
reconnect, and the hub may re-deliver. And `4.xx` response codes map as in §3 (`4.00`
malformed, `4.01` unauthenticated, `4.03` epoch/signature/expiry, `4.09` conflict,
`4.12` evidence, `4.29` too soon) — `4.09` and `4.29` are not registered CoAP codes, so a
deployment choosing CoAP must either carry `error.code` in the payload (recommended) or
define them privately.

---

## 9. Not implemented

| Piece | State |
|---|---|
| `POST /commands`, `GET /commands`, `GET /commands/{id}` | specified here, **absent** from `fd/monitor.py` |
| `switch_pending`, `authority_epoch`, `evidence`, `isolated_nodes`, `last_cmd_seq` in `/status` | specified, absent (as are `api_version` and `retained_from_seq`, per [`API.md`](API.md) §2.1) |
| at-most-once execution, command log | specified, absent |
| mTLS / DTLS, per-command signature, epoch enforcement | specified, absent — the current surface has no authentication at all |
| `monitor_commanded` switch policy, deadline D | architecture only, per [`monitor_commanded_switch.md`](monitor_commanded_switch.md) |
| `min_nodes`, `min_switch_interval`, `dwell` as enforced preconditions | the values exist as scenario timing; they are not enforced against commands |

What *is* implemented and verifiable today: the event side of the loop. Every commanded
action in both streams carries `decided_by: monitor`, and
[`scripts/validate_events.py`](../scripts/validate_events.py) rejects a stream that
attributes autonomy anywhere the authority rule does not allow it.
`scripts/commands_from_stream.py` derives the command log those events imply and validates
it against the schemas here — which demonstrates that the surface is sufficient for the
scenarios, not that it is built.

## 10. Configuration sketch

```yaml
monitor:
  commands:
    enabled: false              # opt-in; default remains read-only
    require_signature: true
    accept_window_seconds: 60   # hard cap, even when not_after is absent
    min_nodes: 3                # isolation floor (measured: recall holds 14 -> 3)
    min_switch_interval: 10     # rounds, flap protection
    dwell_required: 3           # rounds of full membership before GL -> FL is accepted
    authority_epoch: 3          # bumped on monitor failover
architecture:
  switch_policy: monitor_commanded
  deadline_rounds: 3            # D — not commandable at runtime
```

## 11. Files

- [`schemas/command.schema.json`](../schemas/command.schema.json) — command payload.
- [`schemas/command_result.schema.json`](../schemas/command_result.schema.json) — result / log entry.
- [`openapi.yaml`](openapi.yaml) — endpoints, `$ref`ing both.
- [`scripts/validate_commands.py`](../scripts/validate_commands.py) — conformance, including the invariants JSON Schema cannot state.
- [`scripts/commands_from_stream.py`](../scripts/commands_from_stream.py) — derive the implied command log from an event stream.
- [`API.md`](API.md) — the observation half.
- [`monitor_commanded_switch.md`](monitor_commanded_switch.md) — the FL→GL policy and deadline D.
- [`decentralized_monitoring.md`](decentralized_monitoring.md) §1.1 — the authority rule.
