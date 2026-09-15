# v3 — Unified trust model (3-driver reputation)

**Status:** design spec for v3 (Byzantine). NOT implemented in v2. This is the trust
layer that turns the current *binary* control plane (node up/down, unary votes) into
a *valued* one (reputation-weighted exclusion, ranking, and votes).

## The unified trust signal

Each node maintains a reputation for every peer, fused from **three drivers**:

```
trust(j) ← f( availability ,        # driver 1: # crash / timeout
              model-concordance ,    # driver 2: Byzantine audit (LocalAudit / TrustedUnion)
              IDS-verdict-on-j )     # driver 3: IDS detects j as the SOURCE of an attack
                                     #           (attributed + corroborated)
```

`trust(j)` then drives: (a) **exclusion** (a peer below a threshold is dropped from
aggregation / vote counting), (b) **ranking** (peer sampling / head-election bias),
and (c) **valued/weighted votes** (a vote counts proportionally to the voter's trust,
replacing the v2 unary "who voted" tally).

## Where we stand today (v2) — honest baseline

There is **no `trust(j)` function today** and **no reputation accumulator**. Of the
three drivers, only driver 1 has an implemented *signal*, and even that is an
instantaneous boolean/quorum (drives the FL↔GL switch), not a score.

| Driver | Today | Nearest existing code |
|---|---|---|
| 1. Availability (crash/timeout) | **signal exists** (not accumulated into a score) | `fd/peer_failure.py` — local timeout + diffusion, `min_witnesses`, agreement-quorum |
| 2. Model-concordance (LocalAudit/TrustedUnion) | **not implemented** | `GlowStrategy` `score` agg = self-reported accuracy weight (NOT concordance, NOT used in deployed `xgb_union`) |
| 3. IDS-verdict-on-j (attribution) | **not implemented** | IDS classifies *samples* attack/benign; never attributes to an *emitter* node |

## Driver specs (what each needs to leave paper)

### Driver 1 — availability
- **Signal:** count of crash/timeout events for j over a window (from `PeerFailureDetector`).
- **To do:** turn the current instantaneous suspicion into a **decaying counter** per peer
  (e.g. EWMA of miss-rate), stored per node and diffused with the suspicion bitmap.
- **Corroboration:** crash is objectively observable by any neighbour → `min_witnesses=1`
  suffices (v2 already). No change needed for the crash regime.

### Driver 2 — model-concordance (Byzantine audit)
- **Signal:** does j's contributed booster/update agree with a trusted reference?
  - **LocalAudit:** each auditor re-scores j's booster on its own local validation slice;
    an update that degrades local F1 beyond a margin is a concordance failure.
  - **TrustedUnion admission:** a candidate aggregate is admitted only if it equals the
    dedup-union of *already-trusted* contributions (the `[v3]` admission test currently
    marked inert in `DistributedArchManager`).
- **To do:** implement the admission test (`agg == TrustedUnion`) + the LocalAudit
  re-score; feed pass/fail into `trust(j)`.
- **Corroboration (sparse-graph limit):** a single auditor can be wrong/lying. Needs
  ≥ k independent auditors agreeing on the concordance verdict before it moves `trust(j)`
  — the same corroboration problem as driver 3 (see below).

### Driver 3 — IDS-verdict-on-j (the novel data-plane → control-plane loop)
- **Signal:** the ReSIDS ensemble flags traffic as an attack AND that attack is
  **attributed to node j as the source** (not merely "an attack happened").
- **To do (two hard parts):**
  1. **Attribution:** map a malicious sample/flow to the emitting node. In IEC 61850
     this is plausible via the **GOOSE/SV source** (MAC / APPID / gocbRef → publisher
     node); today the pipeline classifies samples, not emitters, so attribution must be
     added to the feature/label path.
  2. **Corroboration:** a false IDS-verdict-on-j (a compromised or noisy detector
     accusing an honest j) must be resisted → require ≥ k independent detectors to
     attribute the same attack to j before it lowers `trust(j)`.
- **Why it matters:** this closes a loop the literature rarely does — the IDS's *own*
  detections feed peer trust, so a node that is *compromised and attacking* is
  down-weighted by the very system it is attacking. Distinct from drivers 1–2, which
  only see liveness and model quality, not malicious *behaviour*.

## Fusion `f(...)`

- Start simple and monotone: `trust(j) = clip( w1·avail(j) + w2·concord(j) + w3·ids(j) )`,
  each term in [0,1], decaying over time; exclusion at `trust(j) < τ`.
- Reputation itself is **diffused** over the GLow substrate (gossip-aggregated
  reputation), consistent with the AUPE grounding (Mukam 2026: gossip-aggregated
  reputation, Set Cleaner debiasing, bias-factor metric) already cited in
  `docs/ARQUITETURA_v2.md` §11.1 / `docs/refs_v2.bib` (`mukam2026byzantine`).
- **Valued vote** (CONFIA/REJEITA per target, weighted by `trust(voter)`) replaces the
  v2 unary vote — this is the divergent-vote regime where the quorum finally becomes
  discriminative (`f < q` Byzantine dissenters cannot force a decision).

## Corroboration is the cross-cutting hard problem

Drivers 2 and 3 both produce *accusations* that a single (possibly malicious) node
could fabricate. In a sparse overlay only `degree(j)` nodes directly observe j, which is
< a majority quorum — so "q independent observers" is infeasible for a specific peer
(the honest limit already documented in `fd/peer_failure.py`). v3 must therefore add a
**corroboration scheme**: ≥ k independent witnesses, accused-node defence, and
reputation-weighted counting — not a flat majority of the whole network.

## v3 package (this spec + already-deferred items)

- Unified 3-driver `trust(j)` + fusion (this doc).
- Valued vote CONFIA/REJEITA + reputation-weighted quorum (was deferred).
- TrustedUnion admission test made live (driver 2).
- Byzantine peer false-accusation corroboration (drivers 2 & 3).
- Wiring the valued votes/trust into the live `GlowStrategy` (extends the v2 Gap-1/2
  digest, which today carries only unsigned bitmaps + optional agg-sig).

## Related
- `fd/peer_failure.py`, `fd/vote_diffusion.py` — v2 decentralized signals (driver 1).
- `docs/B1_live_run_gap.md` — v2 live decentralized detection (Gaps 1+2 done).
- `docs/ARQUITETURA_v2.md` §11.1 — AUPE / Mukam grounding for gossip-aggregated reputation.
