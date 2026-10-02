#!/usr/bin/env python3
"""Replace the single-cadence `window_seconds: 1.0` of the scenario files with the
two-cadence block supplied by Disaster-FD. Values unchanged, so the streams must be too."""
import pathlib

BLOCK = """# Time reference supplied by Disaster-FD, in two cadences; the agent infers no time from
# the traffic. IEC 61850 profile: 1-s local tick, frame times carried as metadata.
cadence:
  local: {period_s: 1.0}     # node-local Disaster-FD tick: detect_lag, watchdog, event window
  federated: {every: 1}      # regional Disaster-FD tick: dwell, cmd_latency, federated decisions
traffic_time: frame"""

for name in ("availability", "intrusion", "fd_watchdog"):
    p = pathlib.Path(f"conf/scenarios/{name}.yaml")
    text = p.read_text(encoding="utf-8")
    if "cadence:" in text:
        print(f"{name}: already migrated")
        continue
    assert text.count("window_seconds: 1.0\n") == 1, name
    p.write_text(text.replace("window_seconds: 1.0\n", BLOCK + "\n"), encoding="utf-8")
    print(f"{name}: migrated")
