"""Per-node knowledge log of a run (HybridStrategy with node_holdings).

After every round, including the rounds in which every node is local, the set of boosters
each node holds is written as one JSON line of <dir>/rounds.jsonl:
    {"round", "t_s", "mode", "active", "node_modes", "holdings": [[sha, ...] per node]}
Each booster is stored once, as <dir>/boosters/<sha>.json (XGBoost JSON model); sha is
the SHA-256 (first 16 hex digits) of its serialized bytes, the same bytes that
dedup_union compares. <dir>/header.json describes the run (written by main_dist).
scripts/ev1_tl_node_eval.py turns the log into per-node detection over time.
"""
from __future__ import annotations

import hashlib
import json
import os

from flwr.common import Parameters, parameters_to_ndarrays


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:16]


class HoldingsLog:
    def __init__(self, directory: str):
        self.dir = os.path.expanduser(directory)
        os.makedirs(os.path.join(self.dir, "boosters"), exist_ok=True)
        self._known: set[str] = set(f[:-5] for f in os.listdir(os.path.join(self.dir, "boosters")))
        self._rounds = open(os.path.join(self.dir, "rounds.jsonl"), "w")

    def header(self, info: dict) -> None:
        with open(os.path.join(self.dir, "header.json"), "w") as f:
            json.dump(info, f, indent=1, default=str)

    def _ids(self, p: Parameters | None) -> list[str]:
        if p is None:
            return []
        out = []
        for a in parameters_to_ndarrays(p):
            if a.size == 0:
                continue
            raw = a.tobytes()
            h = sha(raw)
            if h not in self._known:
                with open(os.path.join(self.dir, "boosters", f"{h}.json"), "wb") as f:
                    f.write(raw)
                self._known.add(h)
            out.append(h)
        return out

    def snapshot(self, round: int, t_s: float | None, mode: str, active, node_modes,
                 pools: dict[int, Parameters], n_nodes: int, reachable=None) -> None:
        rec = {"round": round, "t_s": t_s, "mode": mode,
               "active": list(active) if active is not None else None,
               "node_modes": list(node_modes) if node_modes is not None else None,
               "reachable": [bool(x) for x in reachable] if reachable is not None else None,
               "holdings": [self._ids(pools.get(i)) for i in range(n_nodes)]}
        self._rounds.write(json.dumps(rec) + "\n")
        self._rounds.flush()

    def close(self) -> None:
        self._rounds.close()
