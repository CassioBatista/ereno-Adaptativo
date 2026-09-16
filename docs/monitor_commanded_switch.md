# Monitor-commanded FL→GL switch (with deadline fallback) — architecture

**Status:** architecture design only — **NOT implemented**. The v2 default remains the
autonomous, decentralized switch (`DistributedArchManager`). This document specifies
an *optional* supervised switching policy for deployments that require operator
oversight / auditability of architecture changes (e.g., critical infrastructure).

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
- The `intrusion_detected` / `node_failure` / `architecture_change` events reuse the
  monitor API (`docs/API.md`); this adds the `set_mode` **command** and the
  `switch_pending` state.
- Byzantine-aware trust of the *commander* (a lying monitor) is v3.
