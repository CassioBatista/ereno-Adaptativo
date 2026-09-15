# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses simple
`vMAJOR.MINOR` tags.

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
