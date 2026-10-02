#!/usr/bin/env python3
"""Print every architecture_change of the given streams: round, direction, decider, cause."""
import json
import sys

for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if e["type"] == "architecture_change":
            print(f"{path.split('events_')[-1]:22s} round {e['round']:4d} "
                  f"{e['from_mode']}->{e['to_mode']:9s} decided_by={e['decided_by']:10s} "
                  f"reason={e['reason']}")
