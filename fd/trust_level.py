"""Trust-level (TL) policy of the ReSIDS agents (docs/TRUST_LEVEL.md, API 1.9.0), on the
telemetry of a Disaster-FD run.

Each node is driven by the TL of its colocated monitor (monitor NN -> node NN - 1):
    band(TL) = local if TL < gl, gossip if gl <= TL < fl, federated if TL >= fl;
    a new band is committed once it has held for the settle time S;
    no observation for longer than F (stale TL) -> local.
The policy is the one of scripts/tl_policy_replay.py (same function, moved here so that the
pipeline and the replay share it).

Run layouts accepted:
    <run>/csv/log_monitorNN_Fleet_*.csv               (handoff package, e.g. scenario 2)
    <run>/runs/*/*/csv/log_monitorNN_Fleet_*.csv      (raw run directory, e.g. run 1)
run_t0: <run>/state/t0.json ("t0_actual", epoch s) when present; otherwise estimated as in
scripts/inspect_disaster_fd_trace.py (first TL = 80 of any monitor minus 900 s).
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

MODES = ("local", "gossip", "federated")


def band(tl: float, gl: float, fl: float) -> str:
    return "federated" if tl >= fl else "gossip" if tl >= gl else "local"


def policy(t, tl, S, F, gl, fl, t_end):
    """t: observation times (s, sorted), tl: TL after each. Returns the committed switches
    [(t_commit, from, to, reason, tl_used, t_obs_used)]. Starts in local (no TL yet)."""
    mode, out = "local", []
    cand, t_cand = None, None
    for i in range(len(t)):
        if i > 0 and t[i] - t[i - 1] > F and mode != "local":
            out.append((t[i - 1] + F, mode, "local", "tl_stale", None, None))
            mode, cand = "local", None
        b = band(tl[i], gl, fl)
        if b == mode:
            cand = None
        elif b != cand:
            cand, t_cand = b, t[i]
        t_next = t[i + 1] if i + 1 < len(t) else t_end
        horizon = t_next if t_next - t[i] <= F else t[i] + F
        if cand is not None and horizon - t_cand >= S:
            tc = t_cand + S
            j = np.searchsorted(t, tc, side="right") - 1
            out.append((tc, mode, cand, "trust_level", float(tl[j]), float(t[j])))
            mode, cand = out[-1][2], None
    return out


def _csv_files(run: str) -> list[str]:
    files = glob.glob(os.path.join(run, "csv", "log_monitor*_Fleet_*.csv"))
    return sorted(files or glob.glob(os.path.join(run, "runs", "*", "*", "csv",
                                                  "log_monitor*_Fleet_*.csv")))


def load_trace(run: str):
    """{monitor id: (t, tl, dev_id, is_timeout)} with t in s since run_t0, sorted; and
    (t0 epoch s, source of t0)."""
    import pandas as pd
    run = os.path.expanduser(run)
    data = {}
    for f in _csv_files(run):
        mid = int(re.search(r"monitor(\d+)", os.path.basename(f)).group(1))
        data[mid] = pd.read_csv(f).sort_values(["ts_arrival", "msg_id"], kind="stable")
    if not data:
        raise FileNotFoundError(f"no Disaster-FD monitor CSV under {run}")
    p = os.path.join(run, "state", "t0.json")
    if os.path.exists(p):
        t0, src = float(json.load(open(p))["t0_actual"]), "state/t0.json"
    else:
        jumps = [df.loc[df["trust_level"] >= 80, "ts_arrival"].min() for df in data.values()]
        t0, src = min(jumps) / 1e6 - 900.0, "estimated (first TL 80 - 900 s)"
    out = {}
    for mid, df in data.items():
        t = df["ts_arrival"].to_numpy() / 1e6 - t0
        out[mid] = (t, df["trust_level"].to_numpy(float), df["dev_id"].to_numpy(int),
                    df["is_timeout"].astype(str).str.lower().eq("true").to_numpy())
    return out, (t0, src)


class LinkStates:
    """Whether each node observes each of its monitored targets as up at time t: the last
    probe of that target (dev_id) at or before t, not a timeout, and not older than `stale`
    seconds. This is the environment of the run (who can reach whom), read from the same
    telemetry; it is what the fixed policies use instead of the trust level."""

    def __init__(self, trace: dict, stale: float = 15.0, symmetric: bool = False):
        """symmetric=False ('monitor'): a ring link is up for node i when i's own monitor gets
        answers from the neighbour. symmetric=True ('scenario'): a ring link is up only when the
        two monitors at its ends answer each other, i.e. neither client is dropped by the
        scenario. The two agree on a trace whose failures cut both directions (scenario 2); the
        pilot run drops clients without cutting their monitors' outbound probes, so a dropped
        client's monitor still sees its neighbours there and only the symmetric reading follows
        the scenario (docs: guide.md of the pilot, 30-45 min)."""
        self.stale = float(stale)
        self.symmetric = bool(symmetric)
        self.n_nodes = len(trace)
        self._obs = {}
        for mid, (t, _tl, dev, timeout) in trace.items():
            for d in np.unique(dev):
                m = dev == d
                self._obs[(mid - 1, int(d))] = (t[m], ~timeout[m])

    def up(self, node: int, dev_id: int, t: float) -> bool:
        k = (node, int(dev_id))
        if k not in self._obs:
            return False
        ts, ok = self._obs[k]
        i = int(np.searchsorted(ts, t, side="right")) - 1
        return i >= 0 and t - ts[i] <= self.stale and bool(ok[i])

    def ring_peers(self, node: int) -> tuple[int, int]:
        """dev_ids of the ring predecessor and successor of a node (0-based node, dev_id =
        node + 1; the aggregator is dev_id 0)."""
        n = self.n_nodes
        return ((node - 1) % n) + 1, ((node + 1) % n) + 1

    def aggregator_up(self, node: int, t: float) -> bool:
        return self.up(node, 0, t)

    def link_up(self, node: int, dev_id: int, t: float) -> bool:
        """Ring link between node (0-based) and the client dev_id (= neighbour + 1)."""
        if not self.up(node, dev_id, t):
            return False
        return (not self.symmetric) or self.up(dev_id - 1, node + 1, t)

    def peer_up(self, node: int, t: float) -> bool:
        return any(self.link_up(node, d, t) for d in self.ring_peers(node))

    def both_peers_up(self, node: int, t: float) -> bool:
        """Both ring links of the node are up: the condition of GL in the scenario documents
        ("a client stays in GL only while both of its ring peers are reachable")."""
        return all(self.link_up(node, d, t) for d in self.ring_peers(node))

    def reachable(self, node: int, t: float) -> bool:
        return self.aggregator_up(node, t) or self.peer_up(node, t)


class NodeModes:
    """Committed mode of every node at any time t (s since run_t0), from a trace."""

    def __init__(self, run: str, t_end: float, settle: float = 5.0, stale: float = 15.0,
                 gl: float = 30.0, fl: float = 50.0):
        self.trace, (self.t0, self.t0_source) = load_trace(run)
        self.t_end = float(t_end)
        self.params = dict(settle=settle, stale=stale, gl=gl, fl=fl)
        self.switches = {}
        for mid, (t, tl, _, _) in self.trace.items():
            keep = (t >= 0) & (t < self.t_end)
            self.switches[mid - 1] = policy(t[keep], tl[keep], settle, stale, gl, fl, self.t_end)
        self.n_nodes = len(self.switches)
        self._tc = {n: np.array([s[0] for s in sw]) for n, sw in self.switches.items()}
        self._to = {n: [s[2] for s in sw] for n, sw in self.switches.items()}

    def mode(self, node: int, t: float) -> str:
        tc = self._tc[node]
        i = int(np.searchsorted(tc, t, side="right")) - 1
        return "local" if i < 0 else self._to[node][i]

    def modes(self, t: float) -> list[str]:
        return [self.mode(n, t) for n in range(self.n_nodes)]
