"""Tier-2 zero-day detector: a protocol SPECIFICATION learned from benign only.

ReSIDS Tier 1 (per-attack XGBoost specialists + k-of-n) cannot detect a true zero-day
(an attack with no specialist). Tier 2 models the *legitimate* IEC-61850 behaviour on a
few protocol fields and flags VIOLATIONS — catching novel attacks with ZERO attack
examples. It is trained on benign only, so every node learns the same complete detector:
Tier 2 is **monolithic / replicated**, independent of the FL⇄GL switch (unlike the
partitioned Tier-1 specialists).

Rule per field (learned from benign): an *allowed-set* for near-constant fields (e.g.
TTL = {11000}) or a *robust range* [q_lo, q_hi] otherwise. A sample is flagged if it
violates any rule (union). Two families are used here (per-sample, drop-in):
  VALUE  — SqNum, StNum, cbStatus, TTL, timeFromLastChange  (value violations)
  RATE   — F55 (inter-packet / timing; catches high-rate attacks)
SEQUENCE (stateful StNum/SqNum monotonicity over an ordered stream) is deferred.

Validated on ERENO (scripts/zeroday_rules_final.py): injection/high_StNum/poisoned 100%,
random_replay ~98%, inverse_replay ~57% (partial), masquerade irreducible, @ ~0.57% FP.
"""
from __future__ import annotations

import numpy as np

# Protocol/rate fields as 1-based ERENO F-numbers. Only fields present in the model's
# feature vector are used (see fields_from_feature_list); F56 (RATE) is not in the
# GRASP-24 set, so the live detector relies on F55, which alone catches the rate attack.
VALUE_FNUMS = [40, 41, 42, 44, 57]      # SqNum, StNum, cbStatus, TTL, timeFromLastChange
RATE_FNUMS = [55, 56]                    # inter-packet / timing


def fields_from_feature_list(feature_list: list[int],
                             families: dict[str, list[int]] | None = None
                             ) -> dict[str, list[int]]:
    """Map 1-based F-numbers to COLUMN indices inside a feature vector built from
    `feature_list` (the ordered 1-based features the model actually carries, e.g. the
    GRASP-24). F-numbers not present in `feature_list` are skipped.
    """
    families = families or {"VALUE": VALUE_FNUMS, "RATE": RATE_FNUMS}
    pos = {f: i for i, f in enumerate(feature_list)}
    return {fam: [pos[f] for f in fnums if f in pos] for fam, fnums in families.items()}


class SpecDetector:
    """Benign-learned protocol specification (Tier 2). Column indices are into whatever
    feature matrix is passed to fit()/flag() — the caller maps F-numbers to columns."""

    def __init__(self, fields: dict[str, list[int]],
                 set_max_unique: int = 8, q_lo: float = 0.0005, q_hi: float = 0.9995
                 ) -> None:
        self.fields = {fam: list(cols) for fam, cols in fields.items()}
        self.cols = sorted({c for cols in self.fields.values() for c in cols})
        self.set_max_unique = int(set_max_unique)
        self.q_lo, self.q_hi = float(q_lo), float(q_hi)
        self.rules: dict[int, tuple[str, object]] = {}

    # ── learn from benign only (zero attack examples) ──────────────────────────

    def fit(self, X_benign: np.ndarray) -> "SpecDetector":
        for c in self.cols:
            vals = X_benign[:, c]
            uniq = np.unique(vals)
            if len(uniq) <= self.set_max_unique:
                self.rules[c] = ("set", frozenset(float(u) for u in uniq))
            else:
                lo, hi = np.quantile(vals, [self.q_lo, self.q_hi])
                self.rules[c] = ("range", (float(lo), float(hi)))
        return self

    # ── evaluate ───────────────────────────────────────────────────────────────

    def violations(self, X: np.ndarray) -> np.ndarray:
        """Boolean matrix (n_samples x n_cols): True where the field's rule is violated."""
        if not self.rules:
            raise RuntimeError("SpecDetector.fit() must be called before evaluation.")
        out = np.zeros((X.shape[0], len(self.cols)), dtype=bool)
        for k, c in enumerate(self.cols):
            kind, spec = self.rules[c]
            x = X[:, c]
            if kind == "set":
                out[:, k] = ~np.isin(x, np.fromiter(spec, dtype=float))
            else:
                lo, hi = spec
                out[:, k] = (x < lo) | (x > hi)
        return out

    def flag(self, X: np.ndarray) -> np.ndarray:
        """Per-sample novelty flag: True if ANY protocol rule is violated."""
        return self.violations(X).any(axis=1)

    def family_flags(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-family novelty flag (union within each family) — for interpretability."""
        V = self.violations(X)
        idx = {c: k for k, c in enumerate(self.cols)}
        return {fam: V[:, [idx[c] for c in cols]].any(axis=1) if cols else
                np.zeros(X.shape[0], bool) for fam, cols in self.fields.items()}

    # ── serialization (Tier 2 is replicated; can be shared/persisted verbatim) ──

    def to_dict(self) -> dict:
        return {"fields": self.fields, "set_max_unique": self.set_max_unique,
                "q_lo": self.q_lo, "q_hi": self.q_hi,
                "rules": {str(c): (k, (sorted(s) if k == "set" else list(s)))
                          for c, (k, s) in self.rules.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "SpecDetector":
        det = cls(d["fields"], d["set_max_unique"], d["q_lo"], d["q_hi"])
        det.rules = {int(c): (k, (frozenset(v) if k == "set" else tuple(v)))
                     for c, (k, v) in d["rules"].items()}
        return det
