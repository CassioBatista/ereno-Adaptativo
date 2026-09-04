#!/usr/bin/env python3
"""Self-test do monitor de arquitetura (fd/monitor.py).

Sobe o endpoint REST, grava um architecture_change e um node_failure,
consulta /health, /status e /events?since=, e valida a trilha JSONL
persistida (sequência monotônica, identificação do nó em falha). Não
depende de Flower. Rodar: ~/venv-ereno314/bin/python scripts/monitor_selftest.py
"""
import json
import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.monitor import EventStore, MonitorServer, MonitorRecorder, build_monitor


def get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    tmp = tempfile.mkdtemp()
    trail = os.path.join(tmp, "monitor_events.jsonl")
    port = 8731

    store  = EventStore(jsonl_path=trail)
    state: dict = {}
    server = MonitorServer(store, state, port=port).start()
    rec    = MonitorRecorder(store, server, state)
    base   = f"http://127.0.0.1:{port}"
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK  " if cond else " FAIL ") + msg)
        ok = ok and cond

    # 1. health
    check(get(f"{base}/health").get("status") == "ok", "GET /health -> ok")

    # 2. simula um round federado, depois falha do nó 3, depois switch FL->GL
    rec.round_start(5, "federated", active_nodes=list(range(10)))
    rec.node_failure(6, [3], detail="timeout")
    ev = rec.architecture_change(6, "federated", "gossip",
                                 active_nodes=[0, 1, 2, 4, 5, 6, 7, 8, 9])

    # 3. /status reflete o estado corrente
    st = get(f"{base}/status")
    check(st["current_mode"] == "gossip", "/status current_mode == gossip")
    check(st["round"] == 6,               "/status round == 6")
    check(3 in st["failed_nodes"],        "/status failed_nodes contém 3")
    check(st["last_seq"] == 2,            "/status last_seq == 2 (2 eventos)")

    # 4. /events?since=0 devolve os dois eventos, em ordem de seq
    evs = get(f"{base}/events?since=0")
    check(len(evs) == 2, "/events?since=0 -> 2 eventos")
    check([e["type"] for e in evs] == ["node_failure", "architecture_change"],
          "ordem: node_failure, architecture_change")
    check(evs[0]["failed_nodes"] == [3], "node_failure identifica o nó 3")
    check(evs[1]["from_mode"] == "federated" and evs[1]["to_mode"] == "gossip",
          "architecture_change FL->GL")
    check(evs[1]["reason"] == "node_failure",
          "reason inferido = node_failure (havia falha na janela)")
    check(evs[1]["failed_nodes"] == [3],
          "architecture_change carrega recent_failed_nodes=[3]")

    # 5. polling incremental: since=last_seq não traz nada novo
    check(get(f"{base}/events?since={st['last_seq']}") == [],
          "/events?since=last_seq -> vazio (incremental)")

    # 6. trilha JSONL persistida e monotônica
    with open(trail, encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    seqs = [l["seq"] for l in lines]
    check(len(lines) == 2, "trilha JSONL tem 2 linhas")
    check(seqs == [1, 2], "seq monotônico 1,2 na trilha")

    # 7. persistência entre runs: novo EventStore no mesmo arquivo continua a seq
    store2 = EventStore(jsonl_path=trail)
    check(store2.last_seq() == 2, "novo EventStore relê a trilha (last_seq=2)")
    rec2 = MonitorRecorder(store2, None, {})
    e3 = rec2.architecture_change(9, "gossip", "federated", reason="recovery")
    check(e3["seq"] == 3, "seq continua em 3 no run seguinte (replay-friendly)")

    # 8. build_monitor desligado -> NullMonitor (zero overhead)
    null = build_monitor({"monitor": {"enabled": False}})
    check(null.architecture_change(1, "a", "b") is None, "NullMonitor no-op quando disabled")

    server.stop()
    print("\n" + ("MONITOR SELFTEST OK" if ok else "MONITOR SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
