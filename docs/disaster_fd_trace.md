# Dataset preparation for the Disaster-FD trust-level simulation

Generator: [`scripts/build_disaster_fd_trace.py`](../scripts/build_disaster_fd_trace.py);
configuration: [`conf/scenarios/disaster_fd_trace.yaml`](../conf/scenarios/disaster_fd_trace.yaml);
output: `results/disaster_fd_trace/{trace,calibration}.npz` and `manifest.json` (SHA-256 of
the inputs, configuration, per-phase composition). Items marked **(to confirm)** await the
Disaster-FD team.

## Description (for the simulation section)

The simulation uses ERENO, a public dataset of IEC 61850 GOOSE and Sampled Values traffic
for digital substations, with seven attacks against the publish-subscribe messages
(random replay, inverse replay, masquerade as fault, masquerade as normal, injection, high
status number and high-rate poisoning). Each sample is a message described by 58 features;
the specialists use the 24 selected by GRASP. The release provides separate training and
test files. In the original test trace, 2,955,648 messages over 83 minutes, all seven
attacks are interleaved with normal traffic over the whole span, so the trace cannot
express a scenario in which attacks become known, and appear, phase by phase.

The test trace is therefore recomposed by phase. No message is altered: features, labels
and the scores of every specialist are those of the original release; only the order of
the messages and the time axis are synthetic. The 83 minutes follow the five phases of the
scenario, each with the learning mode that the trust level computed by Disaster-FD selects:
isolated clients using their local models (0-15 min, TL < 30), federated learning
(15-30 min, TL >= 50), local models again after a network instability (30-45 min, TL < 30),
gossip learning after a partial recovery (45-60 min, 30 <= TL < 50) and federated learning
once the aggregator returns (60-83 min). Four attacks are known from the start, two are
distributed to clients at minute 30 and one at minute 60 **(to confirm: which ones)**, and
the traffic of each phase contains the attacks already known **(to confirm: or all seven,
so that the not-yet-known ones go undetected until their specialist exists)**.

Each phase keeps the mean rate of the original trace, 35,582 messages per minute. Attack
messages come from the test file only, in contiguous chunks taken in their original order,
so that the message sequence inside each attack, on which features such as the time since
the last status change and the sequence and status numbers depend, is preserved. The
messages of each attack are spread over the phases in which it appears, in proportion to
their durations, so that no message is used twice, and the attack share of a phase is
capped at that of the original trace, 6.78%. Normal messages come from a benign pool made
of the normal messages of the test file followed by the normal messages of the training
file that lie beyond the 500,000 used to train the specialists, which no specialist has
seen. Within a phase, messages are placed at uniform random times, each attack keeping its
internal order. A further ten minutes of normal messages from the same pool, outside the
83 minutes, form a calibration block for the window thresholds, which are recalibrated on
it rather than taken from the original trace. Every random choice is seeded, and the
generator records the SHA-256 of its inputs.

## Resulting composition

| Phase | Minutes | Mode | Trust level | Messages | Attacks in traffic | Attack share |
|---|---|---|---|---:|---|---:|
| P1 | 0-15 | local | TL < 30 | 533,725 | random replay, inverse replay, injection, high StNum | 4.99% |
| P2 | 15-30 | federated | TL >= 50 | 533,725 | the same four | 4.99% |
| P3 | 30-45 | local | TL < 30 | 533,725 | + masquerade as fault, high-rate poisoning | 6.78% |
| P4 | 45-60 | gossip | 30 <= TL < 50 | 533,725 | the same six | 6.78% |
| P5 | 60-83 | federated | TL >= 50 | 818,378 | + masquerade as normal | 6.78% |
| calibration | (10 min, outside) | — | — | 355,816 | none | 0% |

Benign pool: 2,755,139 normal test messages and 2,259,425 unseen normal training messages;
the trace uses 2,772,113 and the calibration block 355,816. Attack messages used out of
those available: random replay 36,117 / 39,000; inverse replay 28,076 / 30,319; injection
36,117 / 39,000; high StNum 36,117 / 39,000; masquerade as fault 15,207 / 17,200;
high-rate poisoning 16,420 / 18,570; masquerade as normal 13,111 / 17,420.

## What is synthetic, and what it changes

* **Unchanged:** every message (features, label) and the verdict of every specialist on it.
* **Synthetic:** the order of the messages and their time stamps. The `traffic_time` of the
  events no longer refers to the original ERENO time; the original trace is bursty (about
  one thousand occupied one-second windows in 83 minutes), whereas the recomposed one has
  a uniform rate, so per-window volumes differ from those of the original trace.
* **Recalibrated:** the window-triage thresholds, on the calibration block. The figures of
  the original trace (99.3% of attack windows at 0.14% false alarms) remain valid for that
  trace and are not compared with those of this scenario.
* **Phase effects to keep in mind:** P1-P2 carry a lower attack share (4.99%) because only
  four attacks are present; masquerade as normal is present only in P5, where it makes up
  1.6% of the messages.

## Open choices (Disaster-FD team)

1. Which attacks are known at minute 0, 30 and 60 (configuration `known_from`).
2. Attacks in the traffic: only the known ones (`traffic: known`, current) or all seven
   (`traffic: all`).
3. Attack share per phase: the original 6.78% (current) or another target (`prevalence`).
4. Uniform arrival times (current) or a bursty arrival that imitates the original trace.
