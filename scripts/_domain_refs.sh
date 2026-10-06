#!/usr/bin/env bash
# Count domain-specific (IEC 61850 / power) references in the v2 interface artifacts.
cd "$(dirname "$0")/.."
PAT='IEC|GOOSE|IED|62351|APPID|gocbRef|substation|SV cadence|SV samples|TT6|power system'
for f in docs/API.md docs/COMMANDS.md docs/decentralized_monitoring.md docs/MULTI_INSTANCE.md \
         docs/SCENARIOS.md docs/monitor_commanded_switch.md docs/openapi.yaml docs/paper_v2_api.tex \
         schemas/event.schema.json schemas/command.schema.json schemas/command_result.schema.json \
         schemas/scenario.schema.json README.md CITATION.cff .zenodo.json; do
  printf "%-40s %3s\n" "$f" "$(grep -c -i -E "$PAT" "$f" 2>/dev/null)"
done
