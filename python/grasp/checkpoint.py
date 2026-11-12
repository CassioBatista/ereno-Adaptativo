"""Evaluation checkpoint for GRASP: resume by deterministic replay.

Every evaluation is appended to a JSONL file (flushed and fsync'ed) with the ordered
feature list, the metrics, its duration and a hash of the state of Python's `random`
at the moment the evaluation was requested. On a resume, GRASP runs again from the
start with the same seed: the construction, the neighbourhoods and the BitFlip draws
consume `random` in the same order, and each evaluation already in the file is served
from it instead of being recomputed. The run therefore reaches the point where it was
interrupted with the same RNG state, the same incumbent and the same counters as an
uninterrupted run, and continues from there.

The replay is strict: the k-th evaluation requested must have the same ordered feature
list as the k-th record, and, when the record carries one, the same RNG-state hash;
otherwise the run aborts (ReplayDivergence) instead of mixing two trajectories.

The elapsed time is virtual: a replayed evaluation advances the clock of the run by the
duration recorded for it, so the `max_time` check of GRASP sees the time an
uninterrupted run would have used. Records seeded from an old log carry estimated
durations (see scripts/grasp_ckpt_from_log.py).

File layout: line 1 is a header {"kind": "header", ...run configuration...}; every
other line is {"kind": "eval", "k": n, "features": [...], "f1": ..., ...}. A partial
last line (crash during a write) is ignored.
"""

import hashlib
import json
import os
import random
import time


class ReplayDivergence(RuntimeError):
    pass


def rng_hash() -> str:
    return hashlib.sha256(repr(random.getstate()).encode()).hexdigest()[:16]


class _CachedResult:
    """Stand-in for python.result.Result on records that only carry the F1 (seeded from
    a log). GRASP compares solutions on f1score; the other metrics are unknown."""

    def __init__(self, f1: float, used_fs: list[int]):
        self.cx = "cached"
        self.f1score = f1
        self.accuracy = float("nan")
        self.precision = float("nan")
        self.recall = float("nan")
        self.used_fs = list(used_fs)

    def get_f1score(self) -> float:
        return self.f1score


class EvalCheckpoint:
    def __init__(self, path: str, header: dict):
        self.path = path
        self.records: list[dict] = []
        self.pos = 0                     # next record to replay
        self.replayed = 0
        self.rng_checked = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            self._load(header)
        else:
            with open(path, "w") as fh:
                fh.write(json.dumps({"kind": "header", **header}) + "\n")
                fh.flush(); os.fsync(fh.fileno())
        print(f"[ckpt] {path}: {len(self.records)} evaluations to replay")

    def _load(self, header: dict) -> None:
        with open(self.path) as fh:
            lines = fh.read().split("\n")
        good = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                if i >= len(lines) - 2:          # torn last line from a crash
                    print(f"[ckpt] ignoring a partial last line ({len(line)} chars)")
                    break
                raise
        if not good or good[0].get("kind") != "header":
            raise ReplayDivergence(f"{self.path}: missing header line")
        saved = {k: v for k, v in good[0].items() if k != "kind"}
        for k, v in header.items():
            if k in saved and saved[k] != v:
                raise ReplayDivergence(f"{self.path}: run configuration differs on {k!r}: "
                                       f"checkpoint {saved[k]!r}, this run {v!r}")
        self.records = [r for r in good[1:] if r.get("kind") == "eval"]
        for n, r in enumerate(self.records, 1):
            if r["k"] != n:
                raise ReplayDivergence(f"{self.path}: record {n} has k={r['k']}")
        # rewrite without a torn tail, so that appends start on a clean line
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            for r in good:
                fh.write(json.dumps(r) + "\n")
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------
    def lookup(self, features: list[int], grasp):
        """Return the cached evaluation of the next record, or None past the end."""
        if self.pos >= len(self.records):
            return None
        r = self.records[self.pos]
        if list(r["features"]) != list(features):
            raise ReplayDivergence(
                f"evaluation {self.pos + 1}: checkpoint has {r['features']}, "
                f"the run asked for {features}")
        if r.get("rng"):
            h = rng_hash()
            if h != r["rng"]:
                raise ReplayDivergence(
                    f"evaluation {self.pos + 1}: RNG state hash {h} != checkpoint {r['rng']}")
            self.rng_checked += 1
        self.pos += 1
        self.replayed += 1
        if all(r.get(k) is not None for k in ("VP", "FN", "VN", "FP")):
            from python.result import Result
            res = Result("cached", r["VP"], r["FN"], r["VN"], r["FP"], r.get("avg_time", 0.0))
            if res.f1score != r["f1"]:
                raise ReplayDivergence(f"evaluation {self.pos}: rebuilt F1 {res.f1score} "
                                       f"!= recorded {r['f1']}")
        else:
            res = _CachedResult(r["f1"], features)
        # virtual clock: the replay takes ~0 s, the original took r["dur_s"]
        grasp.begin_time -= float(r.get("dur_s", 0.0)) * 1000
        if self.pos == len(self.records):
            print(f"[ckpt] replay done: {self.replayed} evaluations served from the "
                  f"checkpoint ({self.rng_checked} with the RNG hash verified); "
                  f"virtual elapsed {(time.time() * 1000 - grasp.begin_time) / 60000:.1f} min")
        return res

    def rng_hash_now(self) -> str:
        return rng_hash()

    def record(self, n: int, features: list[int], evaluation, dur_s: float, rng: str) -> None:
        rec = {"kind": "eval", "k": n, "features": list(features),
               "f1": float(evaluation.f1score) if evaluation else 0.0,
               "VP": getattr(evaluation, "VP", None), "FN": getattr(evaluation, "FN", None),
               "VN": getattr(evaluation, "VN", None), "FP": getattr(evaluation, "FP", None),
               "avg_time": getattr(evaluation, "avg_time", None),
               "dur_s": round(dur_s, 3), "rng": rng, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush(); os.fsync(fh.fileno())
        self.records.append(rec)
        self.pos = len(self.records)
