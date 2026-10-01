#!/usr/bin/env bash
# Confirms validate_commands.py returns 1 on a non-conformant log and 0 on a good one.
cd "$(dirname "$0")/.."
python3 scripts/validate_commands.py results/commands_intrusion.jsonl >/dev/null 2>&1
echo "log bom       -> exit $?  (esperado 0)"
printf '%s\n' '{"command_obj":{"command_id":"aaaaaaaa-1111-4222-8333-444444444444","issued_at":"2026-01-01T00:10:00+00:00","authority_epoch":3,"issuer":{"kind":"monitor","id":"m"},"command":"hold","params":{}}}' > /tmp/_bad.jsonl
python3 scripts/validate_commands.py /tmp/_bad.jsonl >/dev/null 2>&1
echo "verbo proibido -> exit $?  (esperado 1)"
rm -f /tmp/_bad.jsonl
