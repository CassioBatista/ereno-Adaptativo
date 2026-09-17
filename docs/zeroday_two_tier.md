# Two-tier detection for zero-day attacks — analysis (Paper 1 revision)

**Status:** analysis / design on the v1.x lineage (branch `paper1-revision`). Standalone
experiments — independent of v1.1/v2 and of the XGBoost specialist pipeline (they only
use the core loader + the ERENO data). References in `docs/refs_zeroday.bib`.

## The question and why the obvious answers don't fit

ReSIDS Tier 1 is **supervised** per-attack specialists (attack-vs-benign) fused by
k-of-n. By construction it cannot detect a **true zero-day** (an attack with no
specialist): the ensemble floors at incidental **cross-firing (~3.8%** at the
specialists' ~0.6% FPR; from `scripts/new_attack_convergence.py`).

Two candidate answers are ruled out for *true* zero-day (zero examples):

- **Few-shot / in-context learning** (e.g., Manzoor et al. 2025 [manzoor2025zeroday],
  transformer ICL on ERENO, ~85% on a held-out masquerade). This needs a **few examples**
  of the attack in context — so the attack is **already observed/labelled**, i.e. it is
  *novel-but-seen*, not zero-example. Strong for early adaptation, but it does not answer
  "detect an attack we have never seen".
- Retraining a specialist — same objection (needs labelled examples).

The **only** zero-example paradigm is to **model the legitimate (benign) behaviour and
flag deviations** — statistical anomaly, or a protocol **specification**.

## Experiment 1 (negative): generic anomaly on the GRASP features fails

`scripts/zeroday_anomaly.py` — an Isolation Forest trained on benign only, over the 24
GRASP features. Zero-day (held-out `masquerade_fake_fault`) recall is **0%** at 0.5–1%
FPR and never reaches the 3.8% floor even at 5% FPR; all-attacks recall ~1% at 1% FPR.
The GRASP features were selected to **discriminate known attacks** (supervised), so they
**dilute** the novelty signal. This is consistent with the caution of Sommer & Paxson
[sommer2010outside] and with Manzoor et al.'s finding that standard baselines fail.

## Insight: the protocol invariants carry the signal

Per-class inspection of the IEC-61850 protocol fields (benign vs each attack):

| field | benign | violated by |
|---|---|---|
| **F44 gooseTimeAllowedtoLive (TTL)** | **11000 (constant)** | injection, high_StNum (≈5e4) |
| **F41 StNum** | ~1.5e3 | high_StNum (≈5.5e4), injection (≈5e3) |
| **F57 timeFromLastChange** | ~0.19 | random_replay (≈2.3e3) |
| F40 SqNum | wide (cycles) | (weak alone) |

## Experiment 2 (positive): anomaly restricted to the protocol fields

`scripts/zeroday_spec.py` — Isolation Forest on **only** the protocol fields
(F40 SqNum, F41 StNum, F42 cbStatus, F44 TTL, F57 timeFromLastChange), benign-only,
**per-attack held-out sweep** at benign FPR ~1% (each attack is genuinely unseen —
zero examples):

| attack (held-out) | zero-day recall | note |
|---|---|---|
| poisoned_high_rate | **100%** | rate/timing violation |
| high_StNum | **50%** | StNum out of range |
| injection | **45%** | TTL/SqNum violation |
| masquerade_fake_fault | **22%** | above floor (class 3) |
| inverse_replay | 10% | replay |
| random_replay | 9% | replay |
| **masquerade_fake_normal** | **0%** | **irreducible (mimics normal)** |

Restricting to the protocol fields **recovers the zero-day signal** the 24 GRASP
features diluted (0–1% → up to 100% for protocol-violating attacks) — with **zero attack
examples**. The **irreducible limit is `masquerade_fake_normal`** (0%), which mimics
legitimate traffic; masquerade is the hard case for every method (few-shot ICL reaches
~85% only *after* seeing examples). Open-set recognition [scheirer2013openset] frames
this "none-of-the-known / reject" decision.

## Experiment 3 (explicit rules): a deterministic protocol-VALUE specification

`scripts/zeroday_rules.py` — instead of a learned IsolationForest, one **explicit,
auditable rule per protocol field, learned from benign only** (zero attack examples):
allowed-set for near-constant fields, robust range otherwise. A packet is flagged if it
violates any rule. Learned specification:

```
TTL (F44) ∈ {11000}       StNum (F41) ∈ [16, 4954]     SqNum (F40) ∈ [1, 4998]
cbStatus (F42) ∈ {0, 1}   timeFromLastChange (F57) ∈ [0, 1000]
```

Per-attack held-out recall at **benign FPR = 0.52%** (deterministic, zero examples):

| attack | recall | caught by / why not |
|---|---|---|
| injection | **100%** | TTL≠11000 (+ StNum) |
| high_StNum | **100%** | StNum out of range (+ TTL) |
| random_replay | **82%** | stale timeFromLastChange |
| masquerade_fake_fault | 3% | mimics legitimate values |
| poisoned_high_rate | 1% | **RATE attack — needs inter-packet rate, not field values** |
| inverse_replay | 0% | **valid values in wrong ORDER — needs StNum/SqNum sequence state** |
| masquerade_fake_normal | 0% | irreducible (mimics normal) |

The explicit **value** rules are cleaner and stronger than the learned IF on the clear
value-violators (injection/high_StNum 100%, replay 82%) at a lower **0.52% FP**, and are
fully interpretable. Crucially, they also **precisely reveal the limits of per-packet
value rules**: `poisoned_high_rate` is a **rate** attack (temporal, needs inter-packet
rate rules) and `inverse_replay` violates **order** (needs StNum/SqNum monotonicity /
stateful spec), not field values — so both slip past value rules. A complete protocol
specification therefore needs three rule families: **value** (done here), **rate**
(inter-packet timing), and **sequence/state** (StNum/SqNum monotonicity). Masquerade
remains irreducible for all.

## Two-tier architecture (proposal)

`scripts/architecture_two_tier.py` — Tier 1 (supervised specialists, k-of-n, FL⇄GL,
known attacks) **+** Tier 2 (benign-only protocol anomaly, zero-day). A sample is a
zero-day candidate when it is **anomalous AND unclaimed by any specialist**. A novel
alarm **graduates** to a specialist once labelled, then diffuses via FL/GL.

## Honest scope and future work

- **Not solved for stealthy masquerade** by any zero-example method (`masquerade_fake_normal`
  = 0%). ReSIDS's core contribution remains resilient, runtime-switchable detection of
  **known** attacks; the Tier-2 protocol-anomaly adds coverage for **protocol-violating**
  novelties.
- **Explicit rule-based specification** (DONE — Experiment 3, `scripts/zeroday_rules.py`):
  value rules learned from benign catch the value-violators (injection/high_StNum 100%,
  stale-timing replay 82%) at **0.52% FP**, deterministic and interpretable. Remaining
  future work is the other two rule families the value rules revealed as necessary:
  **rate** rules (inter-packet timing → poisoned_high_rate) and **sequence/state** rules
  (StNum/SqNum monotonicity → inverse_replay).
- **Few-shot / ICL** [manzoor2025zeroday] complements this for the *observed-novelty*
  regime (after the first sightings) — pairs naturally with the graduation loop.

## Artifacts
- `scripts/zeroday_anomaly.py` + `results/zeroday_anomaly.{png,pdf,csv}` — negative (IF on 24 GRASP).
- `scripts/zeroday_spec.py` + `results/zeroday_spec.{png,pdf,csv}` — per-attack protocol-anomaly sweep.
- `scripts/architecture_two_tier.py` + `results/architecture_two_tier.{png,pdf}` — the two-tier design.
