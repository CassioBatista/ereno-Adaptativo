#!/usr/bin/env python3
"""A profile with attribution 'none' (e.g. CICIoT2023's CSV release) must yield no
source_node anywhere -- and therefore no intrusion-driven isolation, since there is no
emitter to isolate. Runs the intrusion scenario under such a profile; writes nothing."""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario_events import gen, load_scenario
from scored_cache import get_scored

ts, fired, y, nc, cv, spec, _ = get_scored("conf/scenarios/intrusion.yaml")
sc = load_scenario("conf/scenarios/intrusion.yaml")
sc["profile"]["attribution"] = "none"
sc["name"] = "_tmp_attr_none"
out, _, _ = gen(sc, ts, fired, y, nc, cv, spec, float(ts[0]))
evs = [json.loads(l) for l in open(out)]
os.remove(out)
print("types:", dict(Counter(e["type"] for e in evs)))
print("events with a source_node:", sum(e.get("source_node") is not None for e in evs))
