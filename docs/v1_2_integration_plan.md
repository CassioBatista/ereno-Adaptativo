# v1.2 integration plan — Tier-2 zero-day (protocol specification)

**Status:** plan only (sketch). Target: **v1.2** on the v1.x lineage (branch
`paper1-revision`). Integrates the validated **Tier-2** zero-day detector (protocol
specification: VALUE + RATE families) into the ReSIDS pipeline as a parallel detector.
Grounded in `scripts/zeroday_rules_final.py` (result: injection/high_StNum/poisoned 100%,
random_replay 98%, inverse_replay 57% partial, masquerade irreducible, @ 0.57% benign FP).

## Scope

**In v1.2**
- Tier-2 protocol-spec detector: **VALUE** rules (SqNum, StNum, cbStatus, TTL,
  timeFromLastChange) + **RATE** rules (F55, F56) — per-sample, benign-only, zero attack
  examples.
- Runs in **both FL and GL, identically** (Tier-2 is monolithic/replicated — see below).
- Fusion with Tier-1: `novelty = spec-violation AND NOT claimed by any specialist`.
- Per-round novelty reporting (metric + log). CoAP/monitor notification is **optional**
  (see §Monitor).

**Deferred (v1.3+ / future)**
- **SEQUENCE (stateful)** rules — StNum/SqNum monotonicity over an *ordered per-source
  stream* (to close `inverse_replay`). Needs stream processing; the aggregated
  `all_in_one` dataset does not support it. Higher effort; orthogonal to FL/GL.
- **Graduation loop** (novel → labelled → new Tier-1 specialist → diffuse).
- **Masquerade** (irreducible for all zero-example methods).

## Key architectural point (why FL/GL cost ≈ 0)

Tier-2 models the **benign baseline**, common to all nodes, so every node learns the
**same complete specification** locally (benign ranges are global constants, e.g.
`TTL ∈ {11000}`). There is nothing to partition or aggregate → Tier-2 is **monolithic
per node and replicated**, **independent of the FL⇄GL switch**. No changes to
`DistributedArchManager` / `HybridStrategy` aggregation. This is the opposite of Tier-1
specialists, which are partitioned and need FL/GL to unite.

## Components and where they hook

| # | Component | File | Change | Effort |
|---|---|---|---|---|
| 1 | **`SpecDetector`** | `fd/spec_detector.py` (new) | `fit(X_benign)` learns per-field rules (allowed-set / robust range) on the VALUE+RATE fields; `flag(X) -> (bool novelty, field)`; `to_dict/from_dict`. Port the logic from `scripts/zeroday_rules_final.py`. | LOW (~½ day) |
| 2 | **Client hook** | `fd/client_xgb.py` | On first fit, learn `SpecDetector` from the node's benign slice (already available); expose the novelty flag in `evaluate`/metrics. No aggregation. | LOW–MED |
| 3 | **Fusion + reporting** | `main_dist.py` (eval / `_make_round_eval_fn`) | `known = k-of-n(specialists)`; `novel = spec_flag & ~known`; add per-round `ROUND;...;novel=<n>` line + novelty recall on held-out traffic. | MED |
| 4 | **Config** | `conf/*.yaml` | `detection.tier2: {enabled, fields, q_lo, q_hi}`. | LOW |
| 5 | **Self-test (regression)** | `scripts/spec_detector_selftest.py` (new) | Reproduce the per-attack recall (injection/high_StNum/poisoned 100%, random_replay ~98%, inverse_replay ~57%, masquerade low) @ ~0.57% FP as asserts. | LOW |

## Monitor / CoAP notification (decision point)

`fd/monitor.py` and the `intrusion_detected` event are **v2 artifacts — absent on the
v1.x lineage**. Two options for v1.2:
- **(A) Minimal (recommended for v1.2):** no monitor; report novelty via the per-round
  metric + log (Tier-2 is detection-only). Keeps v1.2 small and on-lineage.
- **(B) Backport:** cherry-pick `fd/monitor.py` + `schemas/event.schema.json` +
  `intrusion_detected` from v2/master into v1.2 to emit CoAP notifications. More work;
  couples the v1.x line to v2's monitor.

## Phased plan

1. **Phase 1 — detector + test: DONE.** `fd/spec_detector.py` + `scripts/spec_detector_selftest.py`
   (data-backed regression, all pass: FPR 0.52%, injection/high_StNum 100%, poisoned 99.56%).
2. **Phase 2 — fusion in the live eval: DONE.** `main_dist.py` `_make_round_eval_fn` computes
   `novel = spec_flag & ~known` and emits `ROUND-T2;<round>;<mode>;spec_fpr=..;novel_on_missed=..`;
   opt-in via `detection.tier2.enabled`. **Verified** on `conf/experiments/ereno_tier2_demo.yaml`
   (FL rounds 1-6, GL 7-12): the ROUND-T2 line appears in BOTH modes with identical
   `spec_fpr=0.4972%` every round → confirms Tier-2 is mode-independent (no held-out attack
   in the demo, so `novel_on_missed=0`). *(The eval binarizes labels; a held-out-attack config
   with multi-class eval is the zero-day demonstration — see scripts/zeroday_*.py for efficacy.)*
   The client-side hook (learning the spec at each `XgbClient`) is unnecessary because the
   detector is monolithic/replicated — fitting once on benign at eval is equivalent.
3. **Phase 3 (optional) — notification:** option (A) log-only (ROUND-T2 line, current), or
   (B) backport the v2 monitor for CoAP `intrusion_detected`.
4. **Phase 4 — release:** merge `paper1-revision` → tag **v1.2**; update `CHANGELOG` /
   `CITATION` on the v1.x line; keep VALUE+RATE, mark SEQUENCE/graduation as future work.

## Risks / caveats (honest)

- **Batch vs stream:** the live pipeline evaluates per-round in **batch**; VALUE+RATE are
  per-sample checks → drop in cleanly. SEQUENCE would need a stream loop (deferred).
- **Feature semantics:** F55/F56 (RATE) and F49/F50/F52 (SEQUENCE) roles are inferred
  from behaviour + ERENO consensus cores — **confirm against the ERENO feature dictionary**
  before publishing rule names.
- **Benign homogeneity:** the spec is identical across nodes only if benign profiles are
  homogeneous (true for ERENO). Heterogeneous benign would make per-node specs differ
  (still monolithic per node, just not identical) — a light federated merge of benign
  ranges could harmonize them if ever needed.
- **FPR calibration:** rules learned at benign quantiles [0.0005, 0.9995] give ~0.57% FP;
  expose the quantiles in config for site-specific tuning.

## Effort summary

- VALUE+RATE Tier-2 integrated, running in FL **and** GL: **a few days** (Phases 1–2),
  ~0 in FL-vs-GL differentiation (mode-independent).
- Optional monitor notification: +LOW (log) or +MED (backport).
- SEQUENCE (stateful, closes inverse_replay): separate, HIGHER effort, needs an ordered
  per-source stream — **not** in v1.2.
