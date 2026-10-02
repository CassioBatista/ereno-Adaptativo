# Decentralized observability: RD + Observe + ping

**Status: design.** Today `fd/monitor.py` serves ONE local REST endpoint bound to the
aggregation plane (`build_monitor(conf)` in `main_dist.py`, wired into `HybridStrategy`).
This document specifies the decentralized alternative — one ReSIDS agent per IED, each
with its own CoAP endpoint — and states precisely what is not implemented (§9).

> **OPEN DECISION — transport, deliberately postponed.** Everything below assumes the
> CoAP binding (Resource Directory + Observe + DTLS 1.3). That premise is **not settled**.
> For the Disaster-FD monitor processes — which, under the authority rule of §1.1, decide
> every transition — REST/TLS scores better on ordered reliable delivery,
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

## 1.1 Authority rule — Disaster-FD decides; the agent only protects itself

**Two roles, and only two** (decision of 2026-10-02):

| Role | Who | Does |
|---|---|---|
| **Failure detector and decider** | **Disaster-FD** — one monitor process per node and per server, federated by region | detects failures (suspicion, trust, reliability) **and decides every transition** |
| **Agent** | each ReSIDS node | executes Disaster-FD's commands; acts on its own **only** as a watchdog (below) |

There is no external monitor or supervisor in the architecture. The REST/CoAP surface of
[`API.md`](API.md) and [`COMMANDS.md`](COMMANDS.md) is the interface between the agent and
its Disaster-FD monitor process; in this repository that process is **simulated** (the
"monitor" of `fd/monitor.py`, the scenario generator and the 15-instance demo).

> Failure detection is federated (Disaster-FD); the protective action is local to the
> agent; every other decision belongs to Disaster-FD.

| Situation | Who decides | Evidence | Latency |
|---|---|---|---|
| **node inactive while in FL → GL** | **node-local Disaster-FD monitor** | **local only** — no regional agreement | `detect_lag` (~2 rounds) |
| further node loss while in GL | federated Disaster-FD | regional | policy |
| node with attributed intrusion (any mode) | federated Disaster-FD | regional | policy |
| **GL → FL return** | **federated Disaster-FD** | regional (full membership for `dwell` rounds) | `dwell` + `cmd_latency` |
| *node-local Disaster-FD process down* | **agent watchdog** — FL→GL only | none: absence of its own FD | `detect_lag` + D |

**Why Disaster-FD can command even the fail-fast transition.** The objection to a monitor
deciding FL→GL was the single point of failure: a *central* monitor may sit on the other
side of the very partition that took the server down, leaving the node stuck in FL with a
dead aggregator. A Disaster-FD monitor **co-located with the node** cannot be on the other
side of any partition from its own agent. So Disaster-FD holds authority over every
transition without reintroducing the single point of failure — **provided** the FL→GL is
decided from the local instance's own evidence. If it waited for regional agreement, it
would inherit the federation's latency and its exposure to partitions. Hence the split:
**local to go down, federated to come back up** — the same "fast into the safe mode,
careful into the efficient one" asymmetry the measurements already justify.

The measurements behind the asymmetry are unchanged. In **GL**, losing nodes costs nothing:
recall stays **100% from 14 down to 3 nodes**, because the retained union keeps every
booster. In **FL** it costs: each lost node removes its specialist from the aggregate, and
the server is the single point of failure — measured 96.02 → 86.33 F1 under cascading loss.
GL is the resting state, so **GL→FL has no fallback of any kind**: if the federated
Disaster-FD cannot decide, the system stays in the degraded-but-safe state. Returning to FL
means re-accepting the hub, which is a trust decision — and trust is precisely what
Disaster-FD (built on Impact-FD: reliability threshold, trust level, impact factor)
quantifies.

**The agent's only autonomy is a watchdog on its own detector.** If the node-local
Disaster-FD process crashes or hangs — a process failure, not a network one — the agent
falls back to FL→GL by itself after D rounds (`decided_by: autonomous`,
`reason: autonomous_fallback`). It is a safety net for the case where the detector no
longer exists, not a decision competing with Disaster-FD, and it exists in the fail-safe
direction only. Scenario `conf/scenarios/fd_watchdog.yaml` exercises it: the switch lands
D = 3 rounds later than when the local FD is alive (round 105 vs. 102).

Consequences: the dwell and unanimity gates are **evidence** for the federated Disaster-FD
("membership full for N rounds"), not an agent's commit rule; and a forged `node_failure`
while in GL causes nothing by itself.

Events carry `decided_by` so the trail is never ambiguous: `monitor` = a Disaster-FD
monitor process (local for the fail-fast FL→GL, federated otherwise); `autonomous` = the
agent's watchdog, and nothing else. [`scripts/validate_events.py`](../scripts/validate_events.py)
rejects a stream that claims `autonomous` anywhere but the watchdog FL→GL.

> **An extension of Disaster-FD, stated as such.** As published, Disaster-FD *detects*
> (suspicion, trust, reliability); it does not command an IDS. Making it the decider is an
> extension this design requires, to be agreed with its authors and presented as such.

## 1.2 Time reference — two cadences, both from Disaster-FD

**The agent infers no time from the traffic.** The time reference is supplied by
Disaster-FD, and the window of an event is simply what the agent scored between two
ticks. Traffic timestamps, when the protocol carries them (IEC 61850 GOOSE/SV), ride along
as optional metadata (`traffic_time_start/end`) for correlation with sequence-of-event
records; where the data has none (CICIoT2023), they are absent and nothing else changes.

There are **two cadences**, matching the two halves of the authority rule:

| Cadence | Ticked by | Counts | Delimits |
|---|---|---|---|
| **local** (`round`) | node-local Disaster-FD | `detect_lag`, watchdog D | the traffic window of each event |
| **federated** (`fed_round`) | regional Disaster-FD | `dwell`, `cmd_latency` | when federated decisions may be taken |

*Local to go down, federated to come back up* — in clocks as in authority. A false
FL→GL suspicion only moves the node to the safe mode, so the local cadence can be short and
aggressive; the federated one can be slow and careful. If the federation slows down under
degradation, protection does not slow with it. Measured (`conf/scenarios/two_cadences.yaml`,
federated tick every 5 local rounds): the FL→GL stays at round 102, the GL→FL moves from
606 to 630 = 600 + (dwell + cmd_latency) × 5, on a federated tick; isolations land on the
next federated tick (8→10, 21→25, 23→25, 39→40, 48→50).

**Protection latency** = detection time of the node-local Disaster-FD + at most one local
tick. Co-location makes the command delay negligible; it does not make detecting a
*remote* failure faster, which is bounded by the detector's timeout and grows when the
network degrades. That is the price of not switching on a guess.

**Triage under a cadence that changes** (`scripts/cadence_sweep.py`). Thresholds are
calibrated once; Disaster-FD may then stretch its tick. Calibrating at 1 s (margin ×1.25)
and holding the thresholds fixed:

| Local tick | Rule volume OR fraction | False windows |
|---|---|---|
| stretched to 2 / 5 / 10 s | 98.1 / 98.4 / 98.6 % of attack windows | **0** |
| shortened to 0.5 / 0.25 s | 98.3 / 98.0 % | **1.3 / 6.7 %** of benign windows |
| count-based, 500 / 1000 samples | 95.3 / 91.6 % | 4.1 / 4.1 % |

Stretching — the direction degradation pushes — is safe. Shortening is not: false
positives **cluster in time**, so a short window that lands on a cluster has a high
flagged fraction. Neither `n_flags` nor the fraction is invariant to the window length,
and the event carries both because neither alone holds across cadences (at 10 s volume
keeps 98.2 % and fraction 92.8 %; at 0.25 s fraction keeps 98.0 % and volume 27 %). Hence
the rule: **calibrate at the shortest cadence Disaster-FD may use**. For data without time,
a count-based tick should not be shorter than the traffic's natural burst: recalibrated at
500 or 1000 samples, detection falls to 58 % and 64 %.

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
map across N nodes would create contradictory reports when one copy goes stale.

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
| `since=` times out *and* RD registration expired | agent really gone | report loss of visibility; the node's own local Disaster-FD still governs its fail-fast |
| Node-local Disaster-FD process unresponsive | FD process crashed or hung | after **D** rounds the agent's watchdog falls back to FL→GL on its own — the only autonomous action, fail-safe direction only |
| Federated Disaster-FD cannot decide | regional partition | nothing happens: the system rests in GL; GL→FL has no fallback by design |

In the regenerated streams: the local Disaster-FD commands FL→GL `detect_lag` = 2 rounds
after the inactivity (round 102 for a failure at 100); with the local FD down, the
watchdog lands D = 3 rounds later (round 105) — and it **always** switches.

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
