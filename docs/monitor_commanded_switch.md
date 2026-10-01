# Monitor-commanded FL→GL switch (with deadline fallback) — architecture

**Status:** architecture design only — **NOT implemented**. The v2 default remains the
autonomous, decentralized switch (`DistributedArchManager`). This document specifies
an *optional* supervised switching policy for deployments that require operator
oversight / auditability of architecture changes (e.g., critical infrastructure).

> **Where the wire format lives.** This document is the *policy*: who decides, the PENDING
> state, and why the deadline D is mandatory. The `set_mode` command itself — payload,
> idempotency, preconditions, failure codes, authentication — is specified in
> [`COMMANDS.md`](COMMANDS.md). One consequence stated there is worth repeating here:
> **no command can cancel, hold or extend D.** A commander able to postpone it is a
> commander able to keep a node in FL with a dead aggregator, which is the single point of
> failure GL exists to survive.

## Motivation

In v2 the FL↔GL switch is **autonomous**: a node detects a peer timeout locally and
self-protects (fail-fast), and the monitor is **read-only**. Some deployments instead
want the mode change to be **authorized by an external authority** (an operator, or a
policy engine at the monitor) — the IDS detects and *reports*, but *waits* for a switch
command. This moves the monitor from observer to **control authority (actuator)**.

## Switch policies (this is a third, configurable one)

| policy | who decides the switch | monitor role |
|---|---|---|
| `static` (v1) | fixed schedule | — |
| `autonomous` (v2, default) | the node itself (fail-fast) | read-only |
| **`monitor_commanded`** (this) | the monitor / operator | **actuator** |

## Authority rule: what stays autonomous, and what does not

This document is about **FL→GL on node inactivity** — and that is the **only** action
the system ever takes autonomously. Everything else is monitor-governed, including the
**GL→FL return**:

| Situation | Authority | Needs a deadline fallback? |
|---|---|---|
| **node inactive while in FL → GL** | **node, fail-fast** (or commanded, per this doc) | **yes** — see below |
| further node loss while in GL | monitor | no |
| node with attributed intrusion | monitor | no |
| **GL → FL return** | **monitor** | **no** |

The asymmetry is deliberate: **autonomy is granted only in the direction that fails
safe.** Staying in FL with a dead server is dangerous, so FL→GL must be able to proceed
without the monitor — hence the mandatory deadline D. Staying in GL is merely slower
(Θ(N) dissemination) while detection is unaffected (recall stays 100% from 14 down to
3 nodes), so **GL→FL deliberately has no fallback**: if the monitor is unreachable, the
system rests in the degraded-but-safe state.

Returning to FL means **re-accepting the hub** — the single point of failure and of
trust that GL exists to survive. That is a trust decision, and it belongs to the
operator rather than to a heuristic. Consequently the v2 recovery gates (per-node dwell
plus unanimity of local health) stop being an autonomous commit rule and become
**evidence reported to the monitor**: "membership full for N rounds".

## Control flow

1. A node detects a **peer timeout** (T rounds silent) — same detector as v2.
2. It **reports** to the monitor (`node_failure` telemetry) and publishes a
   **`switch_pending`** state (`{FL→GL, since round r}`).
3. It enters the **PENDING** state and **starts a deadline timer of `D` rounds**. It
   does **not** switch yet.
4. **Two outcomes (tiered policy):**
   - **Commanded (≤ D rounds):** the monitor issues `set_mode {to: gossip}`; the node
     commits FL→GL and confirms (`architecture_change`, reason `commanded`).
   - **Deadline fallback (> D rounds, no command):** the monitor is unreachable /
     partitioned; the node **falls back to autonomous fail-fast** and commits FL→GL on
     its own (`architecture_change`, reason `node_failure (autonomous fallback)`).

### State machine

```
FL-active ──peer timeout──▶ FL-pending (await command, timer=D)
   FL-pending ──set_mode(gossip) from monitor──▶ GL        (commanded)
   FL-pending ──timer expires, no command──────▶ GL        (autonomous fallback)
```

## Why the deadline fallback is mandatory (the crux)

The purpose of GL is to **survive the loss of central infrastructure**. If the switch
*decision* depended on a **central monitor** with no fallback, a monitor that is
unreachable — plausibly in the **same partition** that took down the FL server — would
leave the node **stuck in FL**, unable to self-protect. That reintroduces a single
point of failure into the very mechanism meant to tolerate one.

The **deadline fallback** resolves this: *ask the monitor, but self-protect if it is
silent*. Under normal conditions the operator retains control and auditability; under
partition/monitor-loss the node degrades gracefully to the v2 autonomous behavior. The
choice of `D` is the **control-vs-resilience knob**: larger `D` = more operator
authority, slower worst-case self-protection; `D→∞` = strictly commanded (max
auditability, min resilience — **not recommended**); `D→0` = effectively autonomous.

## CoAP interface (mapping)

This adds a **command channel** from the monitor to ReSIDS. On a ThingsBoard/Magenta
IoT hub this is **server-side RPC**: the device (ReSIDS) subscribes for RPC (CoAP
`Observe` on `/api/v1/$TOKEN/rpc`), and the monitor issues the command.

| direction | CoAP | payload |
|---|---|---|
| ReSIDS → monitor | POST `/telemetry` | `node_failure {failed_node, round}` |
| ReSIDS → monitor | POST `/attributes` | `switch_pending {FL→GL, since round r}` |
| monitor → ReSIDS | **RPC** (Observe/CON) | `{"method":"set_mode","params":{"to":"gossip","reason":"node_failure"}}` |
| ReSIDS → monitor | RPC response | `{"result":"ok","mode":"gossip","round":r'}` |
| ReSIDS → monitor | POST `/telemetry` | `architecture_change {FL→GL, reason: commanded \| autonomous_fallback}` |

## Trade-offs and security

- **Latency:** the commanded path waits a round-trip + operator decision → slower than
  fail-fast. Bounded by `D` (then fallback).
- **Actuation security:** the command channel is an **actuation surface** — it must be
  authenticated and integrity-protected (DTLS + token, ideally signed commands). A
  spoofed/Byzantine monitor could force GL or *block* GL; this risk did **not** exist in
  the read-only design and connects to the Byzantine-server concern (v3,
  `docs/v3_trust_model.md`).
- **Breaks two v2 principles deliberately:** "distributed & automatic" and "monitor
  read-only (Level-2)". Hence it is an **opt-in policy**, never the default.

## Configuration sketch (not implemented)

```yaml
architecture:
  manager: distributed
  switch_policy: monitor_commanded   # autonomous (default) | monitor_commanded | static
  deadline_rounds: 3                 # D: fall back to autonomous fail-fast after D
  # monitor RPC / auth reuse the existing monitor.coap block (DTLS + token)
```

## Figure
- `results/monitor_commanded_sequence.{pdf}` — the CoAP sequence: report → PENDING →
  (RPC `set_mode` → commit) **or** (deadline → autonomous fallback → commit).

## Relation to the rest of ReSIDS
- Detection reuses the v2 timeout detector (`fd/peer_failure.py`).
- The `intrusion_detected` / `node_failure` / `node_recovery` / `node_isolated` /
  `architecture_change` events reuse the monitor API (`docs/API.md`, schema ≥ 1.3.0);
  this adds the `set_mode` **command** and the `switch_pending` state, both specified in
  [`COMMANDS.md`](COMMANDS.md) (API 1.4.0). Every event carries `decided_by`
  (`autonomous` | `monitor` | `operator`) and, when a command caused it, `command_id` —
  so the trail never leaves the authority ambiguous, and a commanded action can always be
  traced back to the command and the identity that issued it.
- **Intrusion never drives an autonomous transition.** An attributed, corroborated
  intrusion is reported; removing the node (`node_isolated`, `reason: intrusion`) is
  **commanded by the monitor**, and any mode change that follows is commanded too. This
  keeps the notification-only stance of `intrusion_detected` intact, and depends on the
  attribution (`source_node`) that is **null in v2** — so this driver is a v3 premise,
  not v2 behaviour.
- The wider authority rule is stated in
  [`decentralized_monitoring.md`](decentralized_monitoring.md) §1.1.
- Byzantine-aware trust of the *commander* (a lying monitor) is v3.
