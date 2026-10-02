# Fifteen instances: the federated Disaster-FD view over fifteen agents

**Status: running against the real API.** The 15 endpoints are served by `fd/monitor.py`
itself (`scripts/serve_instances.py`), and `scripts/multi_instance_monitor.py` polls them
over HTTP. What is *not* implemented is the command half: the fan-out below is computed and
printed, never sent, because `POST /commands` does not exist yet
([`COMMANDS.md`](COMMANDS.md) §9).

**Roles** ([`decentralized_monitoring.md`](decentralized_monitoring.md) §1.1): each agent
is paired with its own node-local Disaster-FD monitor process, and the monitors federate by
region. `scripts/multi_instance_monitor.py` simulates the **federated** Disaster-FD — the
regional view that reconciles all fifteen agents and takes the decisions that need it
(losses in GL, isolation, the GL→FL return). The fail-fast FL→GL needs none of this: each
node-local instance commands its own agent from local evidence. In this document "the
monitor" means that federated Disaster-FD view; there is no external monitor.

One monitored instance per ReSIDS agent — the FL aggregator plus the 14 clients:

| Instance | Port | Role | Emits | Exists in |
|---|---|---|---|---|
| `srv` | 8722 | FL aggregator | control plane only | **FL only** |
| `c00`–`c13` | 8723–8736 | IED agent | alarms + control plane | both modes |

Each serves the full surface of [`API.md`](API.md) independently. This is not a fifteenfold
repetition of the single-endpoint case: four things change, and three of them are places
where a naive implementation is wrong.

---

## 1. Fifteen watermarks, not one

Every instance has its own `seq` space, starting at 1 and gap-free *within that instance*.
A `since` value from one endpoint is meaningless on another. The monitor therefore holds 15
contiguous watermarks and applies the reconciliation rule of [`API.md`](API.md) §2.1 once
per instance.

The measured reason for the rule is unchanged — polling with the highest seq *seen* rather
than the highest *contiguous* lost 33 events at 20% notification loss — but the exposure is
now fifteen times wider, and a single shared counter would be silently wrong rather than
merely lossy.

## 2. The server's silence is not a fault

In GL the aggregator has no role and stops emitting. A monitor that treats its 15 endpoints
uniformly declares the server dead **immediately after the FL→GL switch it has just
observed** — the fail-fast transition the node-local Disaster-FD instances commanded — so
this false alarm fires exactly when the regional picture most needs to be clear.

The rule is derived from the stream rather than special-cased: while the federation mode is
`gossip`, `srv` is expected quiet. The monitor knows the mode because it read the
`architecture_change` event.

A consequence worth naming: **no scenario currently kills the server.** The loss that most
justifies GL's existence is unexercised, because `conf/scenarios/*.yaml` schedules failures
of client nodes only, and the schedule has no way to name the aggregator. That is a gap in
the scenarios, not in the monitor.

## 3. Agreement is not corroboration

Alarms for the same window arrive from many instances. What that means depends on what the
instances observe, and the two views exist to keep the distinction honest:

**`replicated`** — every client scores the whole window with the same diffused booster
union. All live clients reach the *same* verdict, by design: that is what a converged union
means. Fourteen identical alarms are **agreement**, not independent evidence, and counting
them as corroboration would inflate confidence with replication. Here the monitor's value is
the opposite: **divergence**. An instance reporting a different label, or carrying a
different membership view, is reporting something real — and an isolated-but-live node is
exactly that case, since isolation freezes its view of membership without stopping its
agent.

**`sharded`** — the window's samples are partitioned across the 14 agents, each seeing only
its own slice. This is the deployment case, where an agent observes its own bus segment.
Alarms from different instances are then independent evidence about the same window, and
k-of-n corroboration *at the monitor* becomes meaningful — a second, outer fusion layer
above the k-of-n that already runs inside each node's booster pool.

The replicated view is faithful to what this project measured: one ERENO test split, scored
by everyone. The sharded view is a modelling premise, declared as such. §6 reports the
measured difference between them.

## 4. Fan-out, and why unanimity is asymmetric

A mode switch is federation-wide; commands are per-instance. In normal operation the
fail-fast FL→GL involves no fan-out at all — each node-local Disaster-FD commands its own
agent, and nodes may switch a round or two apart, which is harmless because they all
converge on GL. A fan-out from the federated Disaster-FD (a GL→FL return, or an FL→GL
driven by intrusion) can partially fail, and the two directions do not tolerate that
equally:

| Intent | Partial application | Unanimity |
|---|---|---|
| `set_mode → gossip` (FL→GL) | the unreachable instances converge on GL anyway, which is the resting state | **not required** |
| `set_mode → federated` (GL→FL) | **splits the federation**: some instances aggregate through a hub the others have abandoned | **required** |

So the federated Disaster-FD pre-checks all 15 and aborts the whole intent if any would
refuse a GL→FL return. This is the one place where the multi-instance setting changes command *semantics*
rather than merely repeating them, and it follows directly from the authority asymmetry
([`decentralized_monitoring.md`](decentralized_monitoring.md) §1.1): GL is safe to rest in,
FL is not safe to enter halfway.

The fifteen commands of one decision share an `intent_id` so the trail shows they were one
act, while each keeps its own `command_id` — the per-instance idempotency key, which a
fan-out needs more than a single call does, since a retry after a partial failure must not
re-apply where it already succeeded.

---

## 5. Running it

```bash
python scripts/instance_streams.py conf/scenarios/intrusion.yaml     # 15 streams, both views
python scripts/serve_instances.py results/instances/intrusion_sharded --paced 200 &
python scripts/multi_instance_monitor.py --seconds 30 --tag intrusion_sharded
```

The endpoints are ordinary HTTP and can be poked by hand — with that instance's own
watermark:

```bash
curl 'http://127.0.0.1:8723/events?since=0&limit=3'
curl 'http://127.0.0.1:8722/status'
```

## 6. Measured

Run over real HTTP against the 15 endpoints, intrusion scenario, sharded view, paced 20×:
3,789 events consumed, **no gaps on any instance**, each reconciled on its own contiguous
`seq`.

### 6.1 Agreement vs corroboration

How many of the 14 client instances report the same window:

| Window class | View | Distribution | ≥2 inst. | ≥5 inst. |
|---|---|---|---|---|
| **attack** (n=282) | `replicated` | `{14: 282}` | 100% | 100% |
| | `sharded` | `{14: 263, 2: 1, 1: 18}` | **93.6%** | 93.3% |
| **benign** (n=252) — every alarm is a FALSE POSITIVE | `replicated` | `{14: 87, 13: 49, 12: 62, 11: 54}` | 100% | 100% |
| | `sharded` | `{14: 84, 13: 47, 12: 63, 11: 55, 7: 3}` | **100%** | 100% |

(Benign counts stop short of 14 because three nodes fail during that scenario.)

Read the two bold figures together, because they are the finding:

**Instance-level corroboration is strictly harmful as a filter.** Requiring k≥2 *instances*
discards 6.4% of true detections and removes **zero** false positives. Every false positive
is corroborated by at least 7 instances, even when each agent observes only its own slice.

The reason is structural and should have been predictable: a false positive is not an
idiosyncrasy of one agent, it is a property of the traffic window. A benign window that looks
like an attack looks like one to everybody — so splitting the observation does not decorrelate
the error, it only reduces how much of it each agent sees. Triage therefore has to stay on
**volume** (`n_flags ≥ 275`, which removes every false positive while keeping 91.8% of true
alarms), exactly as the single-endpoint baseline indicated. Fanning out to 15 endpoints buys
reconciliation, attribution and divergence detection; it does not buy a second opinion.

Two further notes on the attack side. The sharded distribution is **bimodal** — either all 14
fire or exactly one does — and the single-instance tail is the sparse windows, where so few
samples fall inside the second that only one agent's slice contains any. And an artifact to
declare: those sparse windows concentrate on `c00` (286 events against 269–270 for the
others), because round-robin slice assignment gives index 0 the first sample of every window.
A deployment partitioned by bus segment would not carry that bias; the 6.4% figure is sound,
its attribution to one instance is not.

### 6.2 Divergence detects isolation on its own

Minority instances, against the majority view of the same window:

| Instance | Windows in the minority | Frozen view | Isolated at |
|---|---|---|---|
| `c11` | 245 | `(federated, 13)` | round 8 |
| `c09` | 245 | `(federated, 12)` | round 21 |
| `c10` | 228 | `(gossip, 11)` | round 23 |
| `c12` | 219 | `(gossip, 10)` | round 39 |
| `c08` | — | `(gossip, 9)` | round 48 |

The monitor identifies **4 of the 5 isolations from divergence alone**, without reading a
single `node_isolated` event. No instance ever disagreed about *what* the attack was — the
divergence is entirely in the membership view, which is the expected signature of a node cut
out of the federation while its agent keeps running.

`c08` is invisible to this signal, and the reason is structural rather than accidental: it
froze at `(gossip, 9)`, which was already the final membership, so no later change exposed
the staleness. **The signal reveals isolation only when membership changes afterwards** —
worth stating, since a monitor relying on divergence alone would miss the last isolation of
any sequence.

### 6.3 The server instance

3 events, all control-plane, and `node_isolated` seen only twice — the isolations of rounds
8 and 21, which happened while the aggregator still had a role. After the FL→GL switch at
round 21 it is out of the protocol and silent, and the monitor classifies that silence as
expected rather than as a fault.

### 6.4 Fan-out, with one instance genuinely unreachable

With `c07`'s port closed (`scripts/_run_fanout_abort_demo.sh`):

```
set_mode -> gossip      reachable 14/15  unanimity not required  may_proceed=True
set_mode -> federated   reachable 14/15  unanimity required      may_proceed=False
    abort: a partial return splits the federation; staying in GL is the safe option,
    and by design it has no deadline fallback.
```

Raw artifacts: `results/instance_corroboration_<scenario>.csv` (one row per view and
window, with the number of instances that fired) and
`results/multi_instance_monitor_<tag>.txt`.

## 7. Files

- [`scripts/instance_streams.py`](../scripts/instance_streams.py) — fan a federation stream out into 15 per-instance streams, in both views.
- [`scripts/serve_instances.py`](../scripts/serve_instances.py) — 15 endpoints, preloaded or paced.
- [`scripts/multi_instance_monitor.py`](../scripts/multi_instance_monitor.py) — the monitor.
- [`API.md`](API.md) — the surface each instance serves.
- [`COMMANDS.md`](COMMANDS.md) — the command surface the fan-out would use.
