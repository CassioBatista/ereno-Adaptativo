#!/usr/bin/env python3
"""Make the domain profile explicit in every scenario (it was only a default)."""
import pathlib

BLOCK = """profile:                     # domain binding (docs/PROFILES.md)
  name: iec61850-goose-sv
  label_set: ereno-7
  attribution: protocol_source"""

for name in ("availability", "intrusion", "fd_watchdog", "two_cadences"):
    p = pathlib.Path(f"conf/scenarios/{name}.yaml")
    t = p.read_text(encoding="utf-8")
    if "\nprofile:" in t:
        print(f"{name}: already has a profile")
        continue
    assert t.count("traffic_time: frame\n") == 1, name
    p.write_text(t.replace("traffic_time: frame\n", "traffic_time: frame\n" + BLOCK + "\n"),
                 encoding="utf-8")
    print(f"{name}: profile added")
