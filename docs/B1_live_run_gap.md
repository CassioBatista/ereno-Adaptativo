# B1 — Live single-run gap (decentralized detection over the live pipeline)

**Status:** design note. B2 (composition) is what we ship for Paper 2; B1 is the
fuller "single live run" and is scoped here, not implemented.

## What B1 means

A *single live run* where nodes actually go silent over rounds, the **decentralized**
detection fires (per-neighbour local timeout + GLow diffusion of suspicions/votes),
the FL→GL switch is decided from each node's own view, and F1 is measured
round-by-round — as opposed to B2, which **composes** two separately-validated
pieces: the measured steady-state F1 levels (`results/redund_k2_fulltest.csv`) and the
modeled detection latency (`fd/peer_failure.py` + `fd/vote_diffusion.py`).

## Already present in the live pipeline (no rework needed)

1. **Node loss on the time axis.** `DistributedArchManager` carries a `faults`
   schedule + `_env_alive(round)` (fd/arch_manager.py) — nodes actually drop/restore
   per round as environment ground truth (never read by the control logic).
2. **Per-round F1.** `HybridStrategy.evaluate` calls `round_eval_fn(round, mode, params)`
   (fd/strategy/hybrid_strategy.py), labelled with the mode the round ran in.
3. **The switch itself** (fail-fast FL→GL; recover-carefully with dwell/cooldown/quorum)
   runs and commits today.

## What is missing (the decentralized sensing/decision path)

### Gap 1 — detection is a central oracle today, not peer-to-peer
`aggregate_fit` feeds `observe(server_round, self._reported_nodes(results))`
(hybrid_strategy.py) — `reported_nodes` = who returned fit results *to the Flower
aggregator*, i.e. a global up/down view. Inside `observe`, every replicated
`_NodeController` receives the **same** global `reported` set (arch_manager.py). The
decentralized models we built and self-tested — `PeerFailureDetector` (per-neighbour
local timeout + GLow diffusion) and `VoteDiffusion` (per-node heard-sets +
decentralized quorum) — are **not wired in**.
- **To close:** each node runs `PeerFailureDetector.observe_and_step` over its *local*
  neighbourhood; each `_NodeController` decides from its **own** `heard`-set, not the
  global set. Mode votes likewise via `VoteDiffusion`, committing when the local
  heard-quorum ≥ q — replacing the central tally `votes_gl >= q`.

### Gap 2 — the control-plane digest does not ride real gossip messages
In the model, votes/suspicions piggyback abstractly (we only count the bytes). In B1
they must be attached to the boosters exchanged in `GlowStrategy` (head-election +
neighbourhood-union loop) and carried across rounds. `GlowStrategy` moves boosters
today; it carries no digest.

### Gap 3 — per-node mode divergence vs. Flower's one-strategy-per-round (STRUCTURAL)
`_committed_mode` is a single global var; `get_mode(round)` returns one mode for the
whole federation. A genuine decentralized run lets nodes **transiently disagree**
during the diffusion window (some already GL, some still FL). But Flower runs **one
server-side Strategy per round** — divergent per-node mode cannot be expressed in the
current sim harness (this is *why* GLow is a Strategy stand-in; Flower has no real P2P).
- **To close:** either model nodes as independent agents **outside** the Flower loop
  (new simulator), or accept the approximation (decentralized control plane decides,
  but the server materializes one mode/round).

### Gap 4 — F1 during the mixed window
`round_eval_fn` evaluates one aggregated parameter set per round. During the detection
window you want F1 **as each node would compute it locally** (some on the FL aggregate,
some already on the GL union) → per-node evaluation, not one global `evaluate`.

## Effort summary

| Piece                    | State | Missing                                        |
|--------------------------|-------|------------------------------------------------|
| Node loss over time      | done  | —                                              |
| Per-round F1             | done  | per-node eval in the mixed window (Gap 4)      |
| Peer detection + vote    | modeled + self-tested | wire in place of the central oracle (Gap 1) |
| Digest on gossip         | —     | attach to `GlowStrategy` boosters (Gap 2)      |
| Per-node divergent mode  | —     | hits Flower's 1-strategy/round limit (Gap 3)   |

**Minimal viable path:** Gaps 1+2+4 are incremental and feasible within Flower (swap
the `observe` feed for the local detectors, carry the digest, evaluate per-node).
**Gap 3 is the watershed** — faithful per-node mode divergence needs stepping out of
the Flower harness into a P2P agent simulator; the approximation (decentralized
decision, one materialized mode/round) keeps B1 within reach.

**Recommendation:** For Paper 2, B2 (measured levels + modeled, self-tested latency)
is defensible. Full B1 with Gap 3 is only warranted if a reviewer demands a live run
with genuine per-node divergence — that cost is a simulator change, not a tweak.

## Related
- `fd/peer_failure.py`, `fd/vote_diffusion.py` — the decentralized models (self-tested).
- `scripts/plot_latency_aware.py` — the B2 composition curve.
- `scripts/redund_k2_fulltest.py` — the steady-state F1 levels (analytical column-drop).
- v3 Byzantine package (valued votes + reputation weight, false-accusation
  corroboration) is a separate track; B1 wiring is the FL↔GL live plumbing, not the
  Byzantine trust model.
