# ReSIDS trust-level surface (Disaster-FD → agent), API 1.9.0

**Status: specification**, with a reference replay of the policy over the Disaster-FD pilot
run ([`scripts/tl_policy_replay.py`](../scripts/tl_policy_replay.py)); the live endpoint is
not in `fd/monitor.py` yet (§8).

## 1. What changes in 1.9.0

Until 1.7.0 Disaster-FD **decided** every transition and the agent executed commands
([`COMMANDS.md`](COMMANDS.md)). From 1.9.0 the roles are split by competence:

| Role | Does | Does not |
|---|---|---|
| **Disaster-FD** (one monitor colocated with each client) | detects failures of the aggregator and of the ring neighbours, and publishes the resulting **trust level (TL)** | choose a learning mode |
| **ReSIDS agent** | decides its own learning mode from the TL of its colocated monitor (§3) and records each switch as an `architecture_change` | probe nodes or compute trust |

Disaster-FD supplies **evidence**, never a mode. Every transition is therefore
`decided_by: agent`; `monitor` and `autonomous` no longer appear in a 1.9.0 stream, and the
watchdog of 1.5.0 becomes the stale-TL rule (§3.3).

**Deferred to future work**, together with the Byzantine-resilient trust model
([`v3_trust_model.md`](v3_trust_model.md)): the intrusion-detection API
(`intrusion_detected`, `node_isolated`, the commands `isolate_node` / `readmit_node`). The
types stay in the schema enum so that older streams still validate; a 1.9.0 stream does not
emit them.

## 2. The input: one observation per probe

Disaster-FD (Algorithm 1) probes each target every 5 s and, after each probe, updates the
monitor-wide TL — the sum of the impact factors of the targets currently trusted:

| Target | Impact factor | |
|---|---:|---|
| aggregator (server, `dev_id` 0) | 50 | |
| ring predecessor | 15 | |
| ring successor | 15 | |

so TL ∈ {0, 15, 30, 50, 65, 80}. Each probe produces one observation,
[`schemas/trust_level.schema.json`](../schemas/trust_level.schema.json), a one-to-one map of
a row of the monitor CSV:

```json
{"monitor": "monitor07", "client": 7, "msg_id": 182, "ts": "2026-10-08T19:02:17.615805+00:00",
 "target": {"dev_id": 0, "role": "server", "impact_factor": 50, "trusted": true, "is_timeout": false},
 "trust_level": 80.0, "threshold": 30.0, "trusted_system": true}
```

(the observation of the pilot run on which client 7 returns to federated five seconds
later, §3.4).

`trusted_system` (TL ≥ 30, the Disaster-FD region threshold) is carried for traceability
only: it means "healthy enough for *some* distributed learning", not federated. The agent
applies its own bands.

## 3. The agent policy

### 3.1 Bands

| TL | Mode | Meaning |
|---|---|---|
| TL < 30 | **local** | neither the aggregator nor both neighbours are trusted: each client trains and detects with its own specialists only |
| 30 ≤ TL < 50 | **gossip** | both ring neighbours trusted, aggregator not: peer-to-peer exchange |
| TL ≥ 50 | **federated** | the aggregator is trusted (IF 50 alone reaches the band) |

The cutoffs are configuration (`tl_bands: {gl: 30, fl: 50}`), recorded on every event.
`local` is new in 1.9.0: the mode set was `{federated, gossip}`, which could not express an
isolated client.

### 3.2 Settle time

A new band is committed only after it has held for **S** seconds (default **5 s**, one
probe cycle). Rows arrive target by target, so at a phase boundary the TL passes through
intermediate values: when the aggregator drops before the neighbours, TL goes 80 → 30 → 15,
and an agent that switched on every row would go federated → gossip → local within a
second. Over the 14 clients of the pilot run:

| S | switches per client | last switch after the boundary (median / max) |
|---:|---|---|
| 0 s | 5–8 (1–4 s transients) | −2.2 s / 3.7 s |
| 3 s | 4 | 0.8 s / 6.7 s |
| **5 s** | **4** (the nominal ones) | **2.8 s / 8.7 s** |
| 10 s | 4 | 7.8 s / 13.7 s |

Latencies are relative to the boundaries of an estimated `run_t0` (§6), so the negative
median for S = 0 means switching on the first row of the change. S plays the role that the
`dwell` gate had for the GL→FL return in 1.4.0–1.7.0: a hub is re-accepted only on
evidence that persists.

### 3.3 Stale trust level

If no observation arrives for **F** seconds (default **15 s**, three probe cycles), the TL
is stale and the agent falls back to **local** (`reason: tl_stale`). A silent monitor is
not evidence that the aggregator is healthy, so the safe direction is away from it. This
replaces the watchdog D of 1.5.0, which could only fall back FL→GL. In the pilot run the
largest gap between observations is 4.8 s, and the rule never fires.

### 3.4 What the agent emits

One `architecture_change` per committed switch, with the evidence that decided it:

```json
{"seq": 1, "ts": "2026-10-08T19:02:22.615805+00:00", "type": "architecture_change",
 "round": 181, "fed_round": null, "from_mode": "local", "to_mode": "federated",
 "mode": "federated", "reason": "trust_level", "decided_by": "agent",
 "trust_level": 80.0, "tl_bands": {"gl": 30.0, "fl": 50.0},
 "tl_ts": "2026-10-08T19:02:22.594895+00:00", "fd_monitor": "monitor07",
 "detail": "band held 5 s"}
```

`round` is the index of the 5-s probe cycle since `run_t0`. `node_failure` and
`node_recovery` remain valid types: they are Disaster-FD's own observations (per-target
trust flips) and may be relayed, without `decided_by`.

## 4. Event schema changes (additive)

[`schemas/event.schema.json`](../schemas/event.schema.json):

| Field | Change |
|---|---|
| `mode`, `from_mode`, `to_mode` | + `local` |
| `reason` | + `trust_level` (band changed and held for S), `tl_stale` (no fresh TL within F) |
| `decided_by` | + `agent` |
| `trust_level` | new: the TL the decision used |
| `tl_bands` | new: `{gl, fl}` in force |
| `tl_ts` | new: time of the observation that produced `trust_level` |
| `fd_monitor` | new: the colocated monitor |

Every pre-1.9.0 stream still validates against the schema (1,616 of 1,616 events).
[`scripts/validate_events.py`](../scripts/validate_events.py) adds the 1.9.0 authority
rule: `decided_by: agent` only on `architecture_change` with reason `trust_level` or
`tl_stale`; `to_mode` equal to the band of `trust_level`; `tl_stale` only towards `local`;
in a stream with agent decisions, no `monitor` or `autonomous` decisions and no intrusion
events.

## 5. Transport

The monitor is colocated with the client, so the channel is host-local (§6.1 of
[`COMMANDS.md`](COMMANDS.md)): it never crosses the network.

| Binding | Direction | Resource |
|---|---|---|
| REST (loopback) | monitor → agent | `POST /trust_level`, body per `trust_level.schema.json`; `204` |
| CoAP | agent observes the monitor | `GET /trust_level` with `Observe`; one notification per observation |

`/status` gains `trust`: `{"trust_level", "tl_ts", "band", "candidate", "candidate_since",
"stale"}`, so a consumer sees the pending band before it commits. The events are still
served and pushed by the surfaces of [`API.md`](API.md) §§2–3.

## 6. Validation on the pilot run

Run `20261008-154706` (14 clients, ring overlay, 83 min; `~/datasets/disaster_fd_run1`).
The run directory has no `run_t0`; as the team's guide prescribes, it is estimated from the
15-min all-clear (first jump to TL = 80 minus 900 s): 2026-10-08 18:47:15.69 UTC, spread
3.8 s over the 14 monitors.

```
python scripts/inspect_disaster_fd_trace.py ~/datasets/disaster_fd_run1   # checklist
python scripts/tl_policy_replay.py ~/datasets/disaster_fd_run1            # policy
```

* 41,802 observations, all valid against `trust_level.schema.json`.
* Every client makes exactly the four switches of the scenario —
  local → federated (15 min), federated → local (30 min), local → gossip (45 min),
  gossip → federated (60 min) — and spends 99.0–100% of each phase in the expected mode;
  the remainder is the switch latency after each boundary.
* 14 of 14 event streams conformant.

Output: `results/tl_policy/events_client<NN>.jsonl`, `results/tl_policy/tl_monitor07.jsonl`
(sample observation stream), `results/tl_policy_replay.txt`.

## 7. Superseded

| 1.4.0–1.7.0 | 1.9.0 |
|---|---|
| `set_mode` commanded by Disaster-FD | the agent switches on the TL |
| `decided_by: monitor` | `decided_by: agent` |
| watchdog D (FL→GL, `autonomous_fallback`) | stale TL → local (`tl_stale`) |
| `dwell` gate on GL→FL | settle time S on every switch |
| `isolate_node`, `readmit_node`, `intrusion_detected`, `node_isolated` | future work (with the Byzantine model) |

## 8. Not implemented

| Piece | State |
|---|---|
| `POST /trust_level`, CoAP observe, `/status.trust` | specified here, absent from `fd/monitor.py` |
| agent policy (§3) | reference implementation in `scripts/tl_policy_replay.py` (offline replay) |

Updated to 1.9.0: [`openapi.yaml`](openapi.yaml) (`POST /trust_level`, `/status.trust`,
`/commands` deprecated), the sequence diagrams
([`tl_switch_sequence.py`](../scripts/tl_switch_sequence.py), which supersedes
`monitor_commanded_sequence.py`, and
[`decentralized_monitoring_sequence.py`](../scripts/decentralized_monitoring_sequence.py)),
and the paper section [`paper_v2_api.tex`](paper_v2_api.tex); its 1.7.0 version is kept
as [`future_work_command_surface_api_1.7.0.tex`](future_work_command_surface_api_1.7.0.tex).
