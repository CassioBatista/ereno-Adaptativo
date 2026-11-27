# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses simple
`vMAJOR.MINOR` tags.

## [v2.1.2] — 2026-11-27

Artifact of the second paper, revised. The code of v2.1.1 plus one replay policy; the
reported results of v2.1.1 are unchanged.

### Added
- **Replay over seeds 43-46** (`scripts/ev1_tl_runners/run_tl_faithful_seeds.sh`,
  `scripts/ev1_tl_seeds_summary.py` -> `results/ev1_tl_seeds_summary.txt`): the 24 runs of
  the faithful replay with the partition and training seed changed, 96 runs; configs and
  per-node evaluation summaries in `results/ev1_tl/faithful_seeds/`.
- **Communication and criterion sensitivity** (exploratory; `scripts/ev1_tl_cost_sensitivity.py`
  -> `results/ev1_tl_cost_sensitivity.txt`): bytes exchanged per policy as implemented
  (full state) and the protection criterion at 90 and 99 %.
- **Aggregator re-election** (exploratory, no training): replay policy `failover_federated`
  in `TrustLevelArchManager` (majority quorum over the largest connected component, election
  of 1 or 3 rounds; the existing policies give the same rounds as before), a constructed
  regional variant of T2 (`scripts/build_regional_trace.py`), runner and template
  (`scripts/ev1_tl_runners/run_tl_failover.sh`, not run) and summaries
  (`scripts/ev1_tl_failover_summary.py`, `scripts/ev1_checks/failover_gap.py`).
- **Noisy trust levels** (exploratory, no training): constructed variants of T1 and T2 with
  probe loss or misreported levels (`scripts/build_noisy_trace.py`) and the settle time at 5
  and 10 s (`scripts/ev1_checks/noise_*.py` -> `results/ev1_tl_noise_summary.txt`).
- Protocol change log entries of 2026-11-26 in `docs/EVALUATION_PROTOCOL.md`; draft updates.

## [v2.1.1] — 2026-11-25

Artifact of the second paper. Same code as the tag `v2.1.0`, which was created before
`CITATION.cff` and `.zenodo.json` were updated and has no release; v2.1.1 adds only that
metadata and this entry.

### Added
- **Trust-level switching (API 1.9.0).** Disaster-FD publishes a trust level per node and
  each node decides its own mode: local below 30, gossip from 30, federated from 50; settle
  time 5 s; local learning when no level arrives for 15 s. `TrustLevelArchManager` replays
  recorded Disaster-FD runs (`links: scenario`, `round_offset_s`, `gl_peers`, `t_gl`).
- **Retention in both modes** (`node_holdings: retain`), per-round holdings log and per-node
  evaluation (`scripts/ev1_tl_node_eval.py`).
- **Four domains under pre-registered protocols:** ERENO, improved CICIDS2017, Morris-4 and
  CICIoT2023; tags `protocol-v1` (dataset screening), `fusion-v1` (fusion rule) and
  `evaluation-v1` (evaluation), amendments dated in `docs/EVALUATION_PROTOCOL.md`.
- **Evaluation scripts and results:** step 1 (`scripts/ev1_step1.py`, seeds 42-46), node
  loss, cross-detection, coverage rule (exploratory), trust-level replay runners
  (`scripts/ev1_tl_runners/`) and summaries, round-period sensitivity, window alarms with
  blocked-5 calibration, attack thinning and cadence (exploratory), and the checks cited in
  the protocol (`scripts/ev1_checks/`).
- **GRASP** under the best-subset retention rule, per-type seeding and resumable
  checkpoints; per-type and combined selections of the four domains.
- **Domain profiles** (`docs/PROFILES.md`).

### Changed
- Fusion by OR at 0.5, selected by the pre-registered procedure, instead of k-of-n voting.
- The evaluated switch is decided by each node from the Disaster-FD trust level, rather than
  by the decentralized fail-fast/unanimity controller (`DistributedArchManager`, kept).
- The command surface of API 1.7.0 is kept as future work.

### Not versioned
- Datasets (see `docs/DATASET_PROTOCOL.md`); large per-run outputs (boosters, score caches,
  per-round holdings, per-node evaluations, run logs) and GRASP checkpoints.

## [v2.0.0] — 2026-09-15

Version 2 turns the round-by-round switch from a **static schedule** into a
**distributed, automatic** one driven by **decentralized failure detection**, and
adds outbound observability. Byzantine trust is scoped for v3.

### Added
- **`DistributedArchManager`** ([fd/arch_manager.py](fd/arch_manager.py)) — automatic
  FL↔GL switch. N per-node controllers; environment `faults` schedule is ground
  truth the controllers only *detect*, never read.
- **Decentralized detection over the GLow substrate**:
  - **`fd/peer_failure.py`** — local per-neighbour timeout + suspicion diffusion,
    with `min_witnesses` and an agreement-quorum view.
  - **`fd/vote_diffusion.py`** — mode votes reuse the GLow head-election
    neighbourhood-union diffusion; decentralized quorum (no central tally).
  - **Fail-fast FL→GL** (commit on local corroboration by `min_witnesses`, no
    network consensus) vs **careful GL→FL recovery** (per-node local dwell +
    unanimity of local health, which is sound under overlay partition by dead
    nodes). Control-plane digest (vote + suspicion bitmaps) rides the real
    `GlowStrategy` FitIns messages; overhead accounted per round.
- **Retained-union seeding** at FL→GL ([fd/strategy/hybrid_strategy.py](fd/strategy/hybrid_strategy.py))
  so a node that fails just before the switch still has its booster in the GL pool
  (recovers F1 to the healthy level instead of getting stuck degraded).
- **Outbound monitor** ([fd/monitor.py](fd/monitor.py)) — pull REST endpoints
  (`/events`, `/status`, `/health`) + JSONL audit trail; `architecture_change` and
  `node_failure` events (which node failed). JSON/CoAP API docs in
  [docs/API.md](docs/API.md), [docs/openapi.yaml](docs/openapi.yaml),
  [schemas/event.schema.json](schemas/event.schema.json).
- **Live endogenous curves** ([scripts/plot_endogenous_live.py](scripts/plot_endogenous_live.py))
  — the FL↔GL switch round is produced by the decentralized detection inside the
  same run that measures per-round F1 (fail-fast down, careful up), reproducing the
  modeled latency. Self-tests: `scripts/distarch_decentralized_selftest.py`,
  `scripts/peer_failure_selftest.py`, `scripts/vote_diffusion_selftest.py`.
- **v2/v3 design docs**: [docs/ARQUITETURA_v2.md](docs/ARQUITETURA_v2.md),
  [docs/B1_live_run_gap.md](docs/B1_live_run_gap.md),
  [docs/v3_trust_model.md](docs/v3_trust_model.md) (3-driver unified trust model).
- Project metadata: `LICENSE` (MIT), `CITATION.cff`, `.zenodo.json`.

### Changed
- README updated with the v2 architecture; the placeholder `ApiArchManager` is
  superseded by `DistributedArchManager` + the outbound monitor.

### Notes
- Byzantine detection (valued votes CONFIA/REJEITA, reputation weighting, the
  TrustedUnion admission test, false-accusation corroboration, and wiring valued
  votes/trust into the live `GlowStrategy`) is **deferred to v3** — see
  [docs/v3_trust_model.md](docs/v3_trust_model.md).

## [v1.0]

Initial ereno-Adaptativo: centralized/federated/gossip in one run with a **static**
`FixedArchManager` schedule; `GlowStrategy` gossip (ring/chain/star), GRASP feature
selection, XGBoost specialists, OR / k-of-n fusion, GL=FL equivalence study.
