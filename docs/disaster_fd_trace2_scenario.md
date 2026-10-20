# Details about the scenario — Trace 2 (request)

- This document describes a second scenario combining IDSs and Disaster-FD, requested from the
  Disaster-FD team. It follows the format and conventions of the first pilot run
  (`scenario.md`, `guide.md`, run `20261008-154706`).
- Duration: 98 minutes (5880 s).
- Clients (IDS): 14, plus one aggregator/server.
- Known attacks: 7 + normal case.
- The 15-minute phase schedule below drives the network failure/recovery events.
  Attack injection/distribution is the IDS side's responsibility and is not part of the
  Disaster-FD evman schedule.
- Purpose: a disaster-recovery sequence in which the aggregator is lost first and a region of
  clients is lost afterwards, in two steps, while the aggregator is still down; the aggregator
  then returns while the lost clients remain unreachable.
- Compatibility: Disaster-FD configuration, IDS-side mode cutoffs, phase cadence and CSV
  outputs are the same as in the first run; the only difference in the schedule is one more
  event (75 min) and a run 15 minutes longer.

## For Federated Learning (FL)

- 1 server / aggregator for Federated Learning (FL).
- 14 IDS clients participating in the FL process.

## For Gossip Learning (GL)

- No aggregator for Gossip Learning (GL).
- 14 clients participating in the GL process in a ring topology.
- Each client is logically connected to a successor and a predecessor, forming a ring structure
  for communication and model updates.

## For Disaster-FD

Unchanged from the first run:

- 1 monitor colocated with each client/IDS.
- Each monitor watches:
  1. the aggregator (id 0);
  2. the predecessor client in the ring;
  3. the successor client in the ring.
- Impact factors:
  - aggregator: `50`;
  - each ring peer (predecessor/successor): `15`.
- Healthy total trust level per monitor: `TL = 50 + 15 + 15 = 80`.
- **Disaster-FD Algorithm 1**, single region threshold `threshold = 30`, monitor interval 5 s.
- IDS-side mode interpretation uses the same two policy cutoffs on the raw numeric TL as in the
  first run:
  - `T_GL = 30`: minimum TL to leave LL.
  - `T_FL = 50`: minimum TL to enter FL.
  Therefore:
  - `TL < 30` → **LL** (Local Learning);
  - `30 <= TL < 50` → **GL** (Gossip Learning);
  - `TL >= 50` → **FL** (Federated Learning).
- As in the first run, the only reachable GL score is `TL = 30`: a client stays in GL only while
  both of its ring peers are reachable, so each lost client sends its two ring neighbours to LL.
  The lost clients are therefore taken as one contiguous region of the ring, so that only the
  two clients at the border of the region lose a peer.

## Timeline

Events occur every 15 minutes. As in the first run, all endpoints start with
`drop_percentage = 100` via the initial IoT-Sim device config; there is **no evman event at
t = 0**. Lost clients are made unreachable (`drop_percentage = 100`), not crashed: as in the
first run, this models a communication outage, so local IDS computation continues.

- **0 min** - start of the scenario:
  - All 15 monitored endpoints (aggregator + all 14 clients) are isolated.
  - `TL = 0` for every monitor; every monitor is in **LL**.
- **15 min** - switch to FL:
  - All 15 endpoints restore to normal; `TL = 80` for every monitor.
  - The network is healthy; every monitor is in **FL**.
- **30 min** - aggregator lost:
  - The aggregator becomes unreachable (`drop_percentage = 100`); all 14 clients stay reachable.
  - Every monitor has its two ring peers reachable → `TL = 30`.
  - Every monitor is in **GL**.
- **45 min** - two clients lost:
  - Clients `{4, 5}` become unreachable; the aggregator remains unreachable.
  - M04 and M05: their own links are cut → `TL = 0` (**LL**).
  - M03 and M06: one ring peer down → `TL = 15` (**LL**).
  - M01, M02, M07–M14: both ring peers up → `TL = 30` (**GL**).
- **60 min** - four more clients lost:
  - Clients `{2, 3, 6, 7}` become unreachable; clients `{4, 5}` and the aggregator remain
    unreachable (the region `{2, …, 7}`, 6 of 14 clients, is lost).
  - M02–M07: `TL = 0` (**LL**).
  - M01 and M08: one ring peer down → `TL = 15` (**LL**).
  - M09–M14: both ring peers up → `TL = 30` (**GL**).
- **75 min** - aggregator restored:
  - The aggregator restores; clients `{2, 3, 4, 5, 6, 7}` remain unreachable through the end
    of the run.
  - M09–M14: `TL = 80` (**FL**); M01 and M08: aggregator + one ring peer → `TL = 65` (**FL**).
  - M02–M07: `TL = 0` (**LL**).
  - The run continues to 98 min.

## Expected phase timeline (relative to ev-man t0)

| Phase start | evman offset (ms) | Network state | Expected per-monitor TL | Downstream mode |
| ----------- | ----------------- | ------------- | ----------------------- | --------------- |
| 0:00 | (initial config) | All 15 endpoints isolated (`drop = 100`) | 0 | LL |
| 15:00 | 900000 | All 15 endpoints cleared (`drop = 0`) | 80 | FL |
| 30:00 | 1800000 | Aggregator dropped | 30 | GL |
| 45:00 | 2700000 | Clients `{4, 5}` dropped | M04, M05: 0; M03, M06: 15; others: 30 | LL (4) / GL (10) |
| 60:00 | 3600000 | Clients `{2, 3, 6, 7}` dropped | M02–M07: 0; M01, M08: 15; M09–M14: 30 | LL (8) / GL (6) |
| 75:00 | 4500000 | Aggregator cleared | M02–M07: 0; M01, M08: 65; M09–M14: 80 | LL (6) / FL (8) |
| 75:00 → end (98:00) | — | Aggregator up, 6 clients still dropped | as above | as above |

Phase-boundary tolerance as in the first run (about ±10 s around 900, 1800, 2700, 3600 and
4500 s).

## Appendix: trust level by number of reachable clients

Per-monitor TL with the aggregator up or down and `n` of the 14 clients reachable, the lost
clients forming one contiguous region of the ring (as in this run). For `2 <= n <= 13` the
reachable clients form a chain: the `n - 2` interior clients see both ring peers and the two
clients at the ends see one. A lost client's own monitor reaches nothing (`TL = 0`, LL).

TL is per monitor: each entry gives the TL of each monitor and, in brackets, how many monitors
have it (it is not a sum over the network).

| Reachable clients `n` | Aggregator up: per-monitor TL | Mode | Aggregator down: per-monitor TL | Mode | Lost clients (TL 0, LL) |
| ---: | --- | --- | --- | --- | ---: |
| 14 | 80 [14] | 14 FL | 30 [14] | 14 GL | 0 |
| 3–13 | 80 [n − 2], 65 [2] | n FL | 30 [n − 2], 15 [2] | n − 2 GL, 2 LL | 14 − n |
| 2 | 65 [2] | 2 FL | 15 [2] | 2 LL | 12 |
| 1 | 50 [1] | 1 FL | 0 [1] | 1 LL | 13 |

With the aggregator up every reachable client is in FL whatever `n`; with it down, GL needs
at least three reachable contiguous clients, and the two clients at the ends of the chain fall
to LL.

## Requests for this run

1. Same telemetry as the first run: the 14 per-monitor CSV files (36 columns) and the `logs/`
   directory.
2. **Persist `run_t0`** in the run directory (for example a file `t0.txt` with the
   `Scenario t0:` line), and flush `logs/evman.log`. In the first run `run_t0` was printed
   only to the terminal and had to be estimated from the 15-min all-clear.
3. Run length 5880 s (98 min): the last event is at 75 min, followed by 23 minutes, as in the
   first run (60 min + 23 min).
