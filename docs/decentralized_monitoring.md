# Decentralized observability: RD + Observe + ping

**Status: design.** Today `fd/monitor.py` serves ONE local REST endpoint bound to the
aggregation plane (`build_monitor(conf)` in `main_dist.py`, wired into `HybridStrategy`).
This document specifies the decentralized alternative — one ReSIDS agent per IED, each
with its own CoAP endpoint — and states precisely what is not implemented (§9).

> **OPEN DECISION — transport, deliberately postponed.** Everything below assumes the
> CoAP binding (Resource Directory + Observe + DTLS 1.3). That premise is **not settled**.
> For the *decision* monitor — which, under the authority rule of §1.1, commands
> isolation and the GL→FL return — REST/TLS scores better on ordered reliable delivery,
> firewall traversal, tooling, IEC 62351-3 alignment, and is the surface already
> implemented (stdlib, no dependency). CoAP keeps the edge for IoT-hub telemetry and
> constrained links; at our volume (558 events in 83 min) its byte savings are
> irrelevant on a substation LAN. If REST is chosen for the monitor path, the
> **Resource Directory (§4) largely loses its purpose**, since HTTP deployments usually
> solve discovery by provisioning or conventional service discovery.
>
> The event model, the contiguous-`seq` reconciliation, the deadline D and the authority
> rule are **transport-independent** and survive either choice — which is exactly why the
> decision can wait. The command surface was specified the same way:
> [`COMMANDS.md`](COMMANDS.md) §§1–7 bind to no transport, and §8 gives both bindings side
> by side. Writing it out did surface one asymmetry worth weighing when the decision is
> taken: two of the command rejections (`409` conflict, `429` too soon) have no registered
> CoAP equivalent, so the CoAP binding must carry `error.code` in the payload.

---

## 1. Three planes, three clocks

| Plane | Cadence | Traffic | Constraint |
|---|---|---|---|
| **Data** — score each sample against the local booster pool, fuse *k*-of-*n* | one sample / ~0.21 ms (SV cadence, in bursts) | **none** — the decision is local | CPU only: ~4.7 k samples/s |
| **Control** — booster diffusion, mode switch, control digest | rounds (~1 s ≈ 4.7 k samples) | ~13 KB per booster, **only when knowledge changes** | must not contend with GOOSE/SV (lower 802.1Q priority) |
| **Observability** — events to the monitor | event-driven + periodic ping | ~300 B per event | none critical |

The separation is the whole point: diffusing *models* (rarely) is what makes the
per-sample decision purely local, so no per-sample traffic ever competes with the
protection traffic that must meet TT6 (3 ms).

## 1.1 Authority rule — autonomy only where inaction costs

There are **two** drivers of node reduction (inactivity and attributed intrusion) and
two mode transitions, but **only one autonomous action in the whole system**:

| Situation | Authority | Latency |
|---|---|---|
| **node inactive while in FL → GL** | **ReSIDS, autonomous fail-fast** | ~2 rounds (measured) |
| further node loss while in GL | monitor | policy |
| node with attributed intrusion (any mode) | monitor | policy |
| **GL → FL return** | **monitor** | policy |

The rule follows the measurements. In **GL**, losing nodes costs nothing: recall stays
**100% from 14 down to 3 nodes**, because the retained union keeps every booster. No
degradation, no urgency, no need to spend autonomy. In **FL** it costs: each lost node
removes its specialist from the aggregate, k≥2 loses redundancy, and the server is the
single point of failure — measured 96.02 → 86.33 F1 under cascading loss.

**Autonomy is granted only in the direction that fails safe.** Being stuck in FL with a
dead server is dangerous, so FL→GL is autonomous *and* carries the deadline fallback D.
Being stuck in GL is merely slower (Θ(N) dissemination), so GL→FL needs **no fallback**
at all: if the monitor is unreachable, the system simply stays in the degraded-but-safe
state. GL is the resting state; returning to the efficient-but-fragile FL means
**re-accepting the hub**, which is the very point of failure and of trust that GL exists
to survive — a trust decision that belongs to the operator, not to a heuristic.

Consequences: the dwell and unanimity gates stop being an autonomous commit rule and
become **evidence** ReSIDS supplies to the monitor ("membership full for N rounds"); and
the autonomous attack surface shrinks to a single trigger — a forged `node_failure` while
in GL can no longer cause anything by itself.

Events carry `decided_by` (`autonomous` | `monitor` | `operator`) so the audit trail
never leaves this ambiguous.

## 2. Components

```
   ┌────────────┐  register (DTLS)   ┌──────────────┐
   │ ReSIDS     │ ─────────────────► │  Resource    │
   │ agent @IED │ ◄───────────────── │  Directory   │
   └─────┬──────┘   lookup           └──────┬───────┘
         │ Observe notifications (CON)      │ lookup (cached)
         │ ───────────────────────────►     ▼
         │                            ┌──────────────┐
         │ ◄─────────────────────────── │   Monitor    │
         │   GET /events?since=<contig> └──────────────┘
         │   (reconcile + liveness + recover, one round trip)
```

**ReSIDS agent** (one per IED) — CoAP server exposing `/status`, `/events`, and an
observable event resource; CoAP client registering itself in the RD.
**Resource Directory** (RFC 9176) — registry: which agents exist and where.
**Monitor** — RD lookup (cached) + Observe subscription + periodic ping + triage.

## 3. Identity: three tiers, deliberately different scopes

| Tier | Content | Lives where | Needed for |
|---|---|---|---|
| Neighbour map | index → address | **each node**, degree-sized (2 on a ring) | gossip, peer timeout |
| Membership roster | valid index space, `n_active` | every node, but tiny (⌈N/8⌉-byte bitmaps) | quorum, unanimity gate |
| Identity map | index ↔ IED ↔ base URI | **RD + monitor only** | showing the operator which IED |

Nodes emit **indices**; the monitor resolves them. Replicating a provisioned identity
map across N nodes would create contradictory reports when one copy goes stale — unlike
the Tier-2 benign spec, which is safely replicated because every node *derives* it from
the same benign traffic.

## 4. Message flows

**Discovery (rare).** Agent registers with an **explicit** `base=` (substation IEDs have
fixed IPs; the implicit source-address base breaks behind NAT). `ep=` carries the IED
name; the logical index is a registration attribute. The monitor looks up once and caches.

**Events (on occurrence).** The agent notifies the monitor over Observe, CON, one JSON
object per event. Four types: `intrusion_detected`, `node_failure`, `node_recovery`,
`architecture_change`. Every event carries `seq`, `ts`, `round`, `window_start`,
`window_end`, `mode`, `n_active`, `active_nodes`.

**Reconciliation (periodic) — one request, three jobs.** The monitor tracks, **per
agent**, the highest **contiguous** `seq` (the largest *c* with every seq ≤ *c*
received) and periodically issues `GET /events?since=<c>`:

* an **empty** answer proves, at once, that the agent is alive, that nothing is
  missing, and that there is nothing to recover;
* a **non-empty** answer both reveals the gap and fills it, in the same round trip.

Polling `/status` first to compare `last_seq` and only then fetching costs **two** round
trips for the same result; `since=<contiguous>` is the compact form.

Using the highest `seq` *seen* instead of the contiguous one is wrong: a later
notification advances the tail and hides interior holes (measured on the replay harness:
33 events permanently lost under 20% loss with the tail test, 0 with the contiguous one).

**State bootstrap (not periodic).** `GET /status` returns `current_mode`, `round`,
`active_nodes[]`, `failed_nodes[]` and `last_seq`. The monitor reads it when it starts,
or after a long outage, to resync state without replaying the whole event history.

**Optional fast liveness.** If node loss must be noticed faster than the reconciliation
period, add a CoAP ping (empty CON → RST, §7). It is cheap but carries no state, so it
complements rather than replaces `since=`.

## 5. Duplicate alarms are a feature, not a bug

Every node holds the same diffused union, so **every node reaches the same verdict** and
will emit the *same* `intrusion_detected`. With per-node endpoints the monitor receives
N copies of each alarm. Two admissible policies:

* **Deduplicate** by (`window_start`, `attack`) and keep the count of reporting agents;
* **Treat the count as corroboration across agents** — an alarm reported by 1 of 14
  agents when all hold the same pool is anomalous, and is exactly the signal a
  compromised/lying agent would produce (v3 territory).

Note `seq` is **per agent** in this architecture; the monitor keeps one contiguity
counter per agent.

## 6. Failure cases and the required reactions

| Observation | Possible causes | Correct reaction |
|---|---|---|
| Notification never arrives | UDP loss | the next `GET /events?since=<contiguous>` both reveals and fills the hole |
| `since=` request times out | agent down **or** address changed | **re-lookup in the RD before declaring it down** — a DHCP renewal must not become a false "node down" |
| `since=` times out *and* RD registration expired | agent really gone | report loss of visibility; the IDS keeps switching autonomously |
| Monitor unreachable while a switch is pending | management partition | after **D** rounds the node falls back to autonomous fail-fast — without this the monitor becomes the SPOF the GL mode exists to survive |

Measured on the replay harness (`scripts/monitor_interaction_sim.py`): autonomous
fail-fast switches 2 rounds after the trigger, monitor-commanded 3 rounds, deadline
fallback 5 rounds — and it **always** switches.

## 7. Transport: DTLS 1.3 (target), 1.2 (interoperability floor)

DTLS is **not optional** here (§8), so the sizing must assume it. **Target DTLS 1.3**
(RFC 9147): its unified record header (1 B config + 1–2 B sequence + optional 2 B
length) plus an 8 B tag costs **~10–13 B**, against **29 B** for DTLS 1.2
(13 B header + 8 B explicit nonce + 8 B tag with `TLS_PSK_WITH_AES_128_CCM_8`).

Stacking the layers for the smallest message, the CoAP ping (IPv4):

| Layer | plain CoAP | + DTLS 1.2 | + DTLS 1.3 |
|---|---|---|---|
| CoAP (empty CON) | 4 | 4 | 4 |
| DTLS record | — | +29 | +11 |
| UDP | +8 | +8 | +8 |
| IPv4 | +20 | +20 | +20 |
| **IP packet** | 32 | 61 | **43** |
| Ethernet frame | 64 (**padded**) | 79 | 64 (**still padded**) |
| **On the wire** (preamble + IFG) | **84** | **99** | **84** |

With 1.3 the whole record fits under Ethernet's 46-byte minimum payload, so on a
substation LAN the ping costs **exactly what plain CoAP costs** — the security is free
at that size. On larger messages the share drops from +15% to ~+5% (a ~200 B `/status`)
and from +10% to ~+4% (a ~300 B event).

1.3 also brings a shorter handshake, AEAD-only cipher suites and a native Connection ID
(in 1.2 that is the separate RFC 9146), which matters if NAT is on the path: the session
survives a mapping change without renegotiating.

**Why 1.2 stays in the document.** RFC 7252 makes DTLS **1.2** with
`TLS_PSK_WITH_AES_128_CCM_8` the mandatory-to-implement profile for CoAP, and several
constrained stacks still ship 1.2 only. So 1.2 is the interoperability floor, not the
design target.

### 7.1 Alignment with IEC 62351 — TLS, not DTLS

IEC 62351 does **not** define a DTLS profile. It delegates transport protection of
**TCP/IP profiles to TLS** (62351-3; 62351-4 for MMS), while GOOSE/SV (62351-6) are
protected **inside the PDU** with group keys (62351-9), precisely because TLS does not
serve multicast with a 3 ms budget.

Two consequences:

1. **This channel is not an IEC 61850 profile.** ReSIDS↔monitor is an out-of-band
   *management* interface; 62351 does not prescribe it. Choosing DTLS here is not a
   deviation from the standard — it is outside its scope.
2. **Where the utility applies 62351, the natural alignment is TLS.** The whole design
   survives the change: **RFC 8323** carries CoAP over **TCP, TLS and WebSockets**, with
   its own Ping/Pong signalling, so `Observe` and `GET /events?since=<contiguous>` keep
   working unchanged — only the transport differs.

The byte budget above then no longer applies (TCP adds header, handshake and
keep-alives), but this channel is **sparse** — 558 events in 83 min on the replay — so
the saving only matters on a constrained link (6LoWPAN, radio), not on a substation LAN.
Whichever transport is chosen, it should reuse the site PKI and cipher policy of
**62351-9** rather than a parallel credential scheme.

**Adopted for this design: DTLS 1.3 over CoAP/UDP**, matching the IoT-hub integration
(ThingsBoard / Magenta) the monitor interface already targets, and keeping the byte
budget of §7. TLS (RFC 8323, CoAP-over-TLS) is the recorded migration path for
deployments governed by 62351 or crossing the utility WAN; since it preserves `Observe`
and `since=<contiguous>` unchanged, the switch costs transport configuration only — no
change to the event model, the reconciliation logic or the fallback semantics.

## 8. Identity and security

The identity is self-asserted unless the transport authenticates it.

* **DTLS everywhere** (`coaps://`, 5684). A bearer token in the URI path travels in
  clear over plain CoAP and leaks into logs and proxies.
* **Per-device X.509** preferred over a path token: identity is in the handshake.
* **Authenticate RD registration** — whoever can write a registration can redirect the
  monitor to a rogue endpoint that answers pings and forges `node_failure`, which is the
  fail-fast trigger.
* **Corroborate before acting.** ReSIDS already requires `min_witnesses` internally; a
  single-source `node_failure` arriving at the monitor deserves the same treatment.

## 9. What is NOT implemented

* CoAP entirely — push, Observe, DTLS. `fd/monitor.py` is stdlib REST on 127.0.0.1 only.
* Resource Directory: no registration, no lookup.
* Per-node endpoints: today there is one endpoint, on the aggregation plane.
* `node_recovery` is not in `schemas/event.schema.json` (the enum has three types), and
  `detector_nodes` should be `detector_specialists` (they are boosters, not nodes).
* `window_start`/`window_end` and the `members` block of `/status` are proposals.
* Membership is a **static index space**: nodes go up and down, but no node joins with a
  new index. Dynamic membership changes N, hence the quorum and the bitmap width — that
  belongs with the v3 package.
