# Disaster-FD-commanded FL→GL switch (with the agent's watchdog)

**Status: architecture, adopted 2026-10-02. Implemented in the scenario generator and the
conformance checks; the Disaster-FD side is simulated.** This replaces the earlier design
in which the agent switched autonomously by default and a central "monitor" could
optionally command it. There is no central monitor and no external supervisor: the only
roles are **Disaster-FD** (one monitor process per node and per server, federated by
region) and the **ReSIDS agent**. The authority rule is stated in
[`decentralized_monitoring.md`](decentralized_monitoring.md) §1.1; this document is the
policy for the one transition that must be fast.

> **Where the wire format lives.** The `set_mode` command — payload, idempotency,
> preconditions, failure codes, authentication — is specified in
> [`COMMANDS.md`](COMMANDS.md). One consequence is worth repeating here: **no command can
> cancel, hold or extend the watchdog D.** A commander able to postpone it would be able to
> keep a node in FL with a dead aggregator.

## Who decides FL→GL

| Situation | Decides | Evidence | Fallback |
|---|---|---|---|
| **node inactive while in FL → GL** | **node-local Disaster-FD monitor** | local only — no regional agreement | agent watchdog after D |
| further node loss while in GL | federated Disaster-FD | regional | none |
| node with attributed intrusion | federated Disaster-FD | regional | none |
| **GL → FL return** | **federated Disaster-FD** | regional | **none, by design** |

## Why the node-local instance, and why local evidence only

The purpose of GL is to survive the loss of central infrastructure. A *central* decider
could be on the other side of the partition that took the server down and leave the node
stuck in FL. A Disaster-FD monitor **co-located with the node** cannot be: it shares every
partition with its own agent. So it can hold authority over the fail-fast transition
without becoming the single point of failure GL exists to remove.

That holds only if the decision uses **local evidence**. Waiting for the regional
federation to agree would import its latency and its exposure to partitions into the one
transition whose value is speed. Hence: **local to go down, federated to come back up** —
fast into the safe mode, careful into the efficient one. Recall stays 100% from 14 down to
3 nodes in GL, so resting there costs nothing; returning to FL re-accepts the hub, which is
a trust decision, and trust is what the federated Disaster-FD quantifies.

## Control flow

1. The node-local Disaster-FD monitor observes inactivity (its own detection; in this
   repository, simulated with `detect_lag` rounds).
2. It commands `set_mode {to: gossip, reason: node_failure}` to its agent.
3. The agent commits FL→GL at the next round boundary and emits `architecture_change`
   with `decided_by: monitor`, `reason: node_failure` and the `command_id`.

The agent's watchdog runs alongside:

- if its node-local Disaster-FD process does not answer for **D** rounds — the process has
  crashed or hung — the agent commits FL→GL on its own: `decided_by: autonomous`,
  `reason: autonomous_fallback`, no `command_id`;
- if the command arrives after the watchdog already fired, the agent answers `no_op`: the
  state Disaster-FD wanted is the current one.

```
FL-active ──local FD: inactivity──▶ set_mode(gossip) ──▶ GL          (decided_by=monitor)
FL-active ──local FD silent for D──────────────────────▶ GL          (decided_by=autonomous,
                                                                       reason=autonomous_fallback)
GL ──federated FD: full membership for dwell──▶ set_mode(federated) ──▶ FL   (no fallback)
```

## What the watchdog is, and what it is not

It is a **health check on the agent's own detector**, not a race against a remote decider.
It fires only when the local Disaster-FD process has stopped responding, it acts only in the
fail-safe direction, and it never returns to FL. With the local FD alive it never fires.

Measured in the regenerated streams: with the local FD alive, a failure at round 100 is
followed by the commanded FL→GL at round 102 (`detect_lag` = 2); with it down
(`conf/scenarios/fd_watchdog.yaml`), the watchdog lands at round 105 (D = 3 more). The
price of a dead local detector is D extra rounds in FL — bounded, and the switch always
happens.

The choice of D trades how long the agent trusts that its detector is merely slow against
how long it stays exposed if the detector is dead. It is configuration, not a command.

## Security

- Between the agent and its **own** node-local Disaster-FD process the channel is local
  (IPC on the same host), so authentication can be lighter.
- Commands from the **federated** Disaster-FD cross the network and keep the full
  treatment of [`COMMANDS.md`](COMMANDS.md) §6: mutual authentication, per-command
  signature, authority epoch, expiry.
- A compromised Disaster-FD could force GL, force FL or isolate honest nodes; the bounds of
  `COMMANDS.md` §6.3 apply. Byzantine-aware trust of the decider is v3.

## An extension of Disaster-FD

As published, Disaster-FD detects (suspicion, trust, reliability) and federates monitors by
region; it does not command an IDS. Deciding the transitions is an extension this
architecture requires — to be agreed with the Disaster-FD authors and presented as an
extension, not as an existing capability.

## Configuration sketch

```yaml
architecture:
  manager: distributed
  switch_policy: disaster_fd        # the only policy; FL->GL from the node-local FD
  watchdog_rounds: 3                # D: agent falls back to FL->GL if its local FD is silent
```

In the scenario format this is `timing.detect_lag`, `timing.watchdog` and
`local_fd_available` ([`SCENARIOS.md`](SCENARIOS.md)).

## Figure

`results/monitor_commanded_sequence.{png,pdf}` still draws the earlier design (a remote
monitor racing a deadline). It needs to be redrawn for the node-local Disaster-FD and the
watchdog.

## Relation to the rest of ReSIDS

- The simulated Disaster-FD reuses the v2 timeout detector (`fd/peer_failure.py`) for the
  inactivity signal.
- Events carry `decided_by` and, when a command caused the action, `command_id`;
  [`scripts/validate_events.py`](../scripts/validate_events.py) rejects `autonomous`
  anywhere but the watchdog FL→GL.
- Intrusion never drives the fail-fast transition: isolation and any mode change it leads
  to are decided by the federated Disaster-FD. Attribution (`source_node`) is null in v2,
  so that driver is a v3 premise.
