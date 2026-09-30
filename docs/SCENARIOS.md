# Scenarios: reproducing an event stream

A **scenario** is a declarative description of a test timeline. Together with the ERENO
data it is everything a third party needs to regenerate a monitor event stream
byte-for-byte — the schedule is data, not code.

* Contract: [`schemas/scenario.schema.json`](../schemas/scenario.schema.json)
* Files: [`conf/scenarios/*.yaml`](../conf/scenarios/)
* Generator: [`scripts/scenario_events.py`](../scripts/scenario_events.py)
* Emitted events: [`API.md`](API.md) (schema ≥ 1.3.0)

```bash
python scripts/scenario_events.py conf/scenarios/availability.yaml conf/scenarios/intrusion.yaml
# -> results/events_availability.jsonl, results/events_intrusion.jsonl
python scripts/validate_events.py results/events_availability.jsonl    # conformance
```

Several scenarios in one call share the trained model, so adding a scenario is cheap.

---

## 1. Why the schedule is a file

It used to be hard-coded. An earlier run scheduled the last node returns at rounds
760/820 while the benign-only stream has only ~724 windows: those returns **never
happened**, membership never became full again, and the GL→FL transition silently never
fired. Nothing crashed and nothing warned — the stream simply came out wrong.

Making the schedule data closes that failure mode by construction: §3 validates every
round against the number of windows the selected stream actually has.

## 2. Fields

| Field | Meaning |
|---|---|
| `name` | scenario id; names the output as `results/events_<name>.jsonl` |
| `stream` | which windows to replay: `benign_only`, `with_attack`, `all` |
| `window_seconds` | round length; one round = one traffic window (1.0 s ≈ 4.7k SV samples) |
| `nodes` | N, the **static** logical index space |
| `fusion_k` | k in the k-of-n decision fusion |
| `timing.detect_lag` | rounds from inactivity to the **autonomous** fail-fast FL→GL |
| `timing.dwell` | rounds of full membership required as **evidence** before GL→FL |
| `timing.cmd_latency` | rounds the monitor takes to issue a command |
| `attribution` | attack class → emitter node — **synthetic, v3 premise** (see §5) |
| `isolation.*` | evidence threshold, isolations before the monitor acts, and before it commands FL→GL |
| `schedule[]` | membership changes: `{round, event: node_failure\|node_recovery, node}` |

`stream: benign_only` is worth its own note: those windows contain **no attack sample**,
so every alarm produced there is a **false positive**. That scenario doubles as the
false-positive baseline the monitor has to filter
([`fp_baseline_stream_a.py`](../scripts/fp_baseline_stream_a.py)).

## 3. Validation — three layers, all fatal except the last

1. **Against the schema** — fields, types, enums, required keys.
2. **Against the stream** — every `round` must exist. This is the check that was missing:

   ```
   [scenario] INVALID:
     schedule[4] round=760 node_recovery node=12: round beyond the stream, which has 724 windows
   ```
3. **Against the state machine** — a node cannot fail while down, nor recover while up.

Plus one **warning** (not fatal): if the last return sits so close to the end that
`dwell + cmd_latency` no longer fits, the GL→FL return will not fire, and the generator
says so instead of staying silent.

## 4. The authority rule is enforced, not configured

`timing` tunes the latencies; it cannot change **who decides**. The generator emits
`decided_by` on every event, and there is exactly one `autonomous` value in the system:

| Situation | `decided_by` |
|---|---|
| node inactive while in FL → GL | `autonomous` (fail-fast) |
| further node loss while in GL | nothing happens |
| intrusion-driven isolation | `monitor` |
| GL → FL return | `monitor` |

See [`decentralized_monitoring.md`](decentralized_monitoring.md) §1.1 for why: autonomy
is granted only in the direction that fails safe.

## 5. Honest scope

* **`attribution` is synthetic.** `source_node` does not exist in v2 — it is `null` in
  every real event. The map exists so the intrusion driver can be exercised end to end;
  it is a **v3 premise, not a measurement**.
* **Time is reconstructed.** Windows come from sorting the test split by its absolute-time
  column, which interleaves ERENO's overlapping per-attack scenarios. The result is a
  coherent synthetic stream, not a single wire capture. UTC stamps derive from a
  synthetic epoch; in a deployment `window_start`/`window_end` come from the GOOSE/SV
  frame timestamps.
* **Membership is a static index space.** Nodes go down and come back with the *same*
  index. A node joining with a new index would change N, hence the quorum and the
  control-digest bitmap width — that belongs with the v3 package.
* **Detection is real.** The specialists are XGBoost boosters trained on ERENO and scored
  on the real full test split; `attack`, `k_votes` and `n_flags` are measured, not
  scripted. Only the *timeline* is scripted.

## 6. Adding a scenario

1. Copy a file in `conf/scenarios/`, set `name` and `stream`.
2. Write the `schedule`. Round numbers index **that stream's** windows — the counts are
   printed by the generator (`stream 'benign_only' -> 724 windows`), so run it once and
   read the number before writing rounds near the end.
3. Run the generator, then the conformance script on the output.
4. Record the pairing scenario → stream in your provenance manifest.
