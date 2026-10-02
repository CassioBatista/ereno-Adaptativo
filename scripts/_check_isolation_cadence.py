#!/usr/bin/env python3
"""Isolations are federated decisions: with the federated tick every 5 local rounds they
must land on rounds that are multiples of 5, at or after the alarm completing the evidence.
Runs the intrusion scenario both ways and compares; writes nothing permanent."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario_events import gen, load_scenario
from scored_cache import get_scored

ts, fired, y, nc, cv, spec, _ = get_scored("conf/scenarios/intrusion.yaml")
t0 = float(ts[0])
for every in (1, 5):
    sc = load_scenario("conf/scenarios/intrusion.yaml")
    sc["cadence"]["federated"]["every"] = every
    sc["name"] = f"_tmp_iso_fed{every}"
    out, _, _ = gen(sc, ts, fired, y, nc, cv, spec, t0)
    evs = [json.loads(l) for l in open(out)]
    acts = [(e["round"], e["type"], e.get("source_node")) for e in evs
            if e["type"] in ("node_isolated", "architecture_change")]
    print(f"federated every {every}: {acts}")
    if every > 1:
        bad = [a for a in acts if a[0] % every]
        print(f"  actions off a federated tick: {bad or 'none'}")
    os.remove(out)
