#!/usr/bin/env python3
"""Self-test do DistributedArchManager (troca distribuída e automática).

Simula o laço round-a-round (participação vinda do ground-truth de faltas do
ambiente; detecção vinda da observação de quem reportou) e valida:
  * fail-fast FL->GL quando um nó cai (quórum de auto-proteção);
  * recover-carefully GL->FL só após participação plena estável (dwell) + cooldown;
  * o reason correto entregue via consume_switch_reason (node_failure / recovery);
  * cooldown evita flapping.
Não depende de Flower. Rodar: ~/venv-ereno314/bin/python scripts/distarch_selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd.arch_manager import DistributedArchManager


def run(mgr, rounds):
    """Reproduz o que o HybridStrategy faz: get_mode (início do round),
    participação = env_alive, observe(reported=env_alive). Retorna a linha do
    tempo de modos vistos NO INÍCIO de cada round + reasons commitados."""
    timeline, reasons = [], []
    for r in range(1, rounds + 1):
        timeline.append(mgr.get_mode(r))               # modo usado no round r
        active = mgr.get_active_clients(r)
        reported = set(active) if active is not None else set(range(mgr.n_nodes))
        mgr.observe(r, reported)                        # detecção pós-round
        rs = mgr.consume_switch_reason()
        if rs:
            reasons.append((r, rs["reason"], tuple(rs["failed_nodes"])))
    return timeline, reasons


def main() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK  " if cond else " FAIL ") + msg)
        ok = ok and cond

    # cenário: N=10, nó 7 cai no round 3, volta no round 8.
    faults = [{"round": 3, "nodes": [7]}, {"round": 8, "nodes": [7], "up": True}]
    mgr = DistributedArchManager(n_nodes=10, initial_mode="federated",
                                 dwell_rounds=3, cooldown_rounds=2, faults=faults)
    tl, reasons = run(mgr, 14)
    print("  timeline:", tl)
    print("  reasons :", reasons)

    # rounds 1..3 federados (a falta do round 3 só é detectada no observe(3),
    # então o round 3 ainda roda federated; a troca vale a partir do round 4).
    check(tl[0:3] == ["federated"] * 3, "rounds 1-3 federated (detecção no fim do 3)")
    check(tl[3] == "gossip", "round 4 já é gossip (fail-fast após detectar nó 7)")

    # commit FL->GL por node_failure identificando o nó 7
    fl_gl = [x for x in reasons if x[1] == "node_failure"]
    check(len(fl_gl) == 1 and fl_gl[0][0] == 3 and fl_gl[0][2] == (7,),
          "commit gossip @round 3, reason=node_failure, nodes=(7,)")

    # nó 7 volta no round 8: participação plena em 8,9,10 (dwell=3) → recovery.
    # cooldown=2 já satisfeito (último switch no round 3).
    rec = [x for x in reasons if x[1] == "recovery"]
    check(len(rec) == 1, "exatamente um commit de recovery")
    if rec:
        rec_round = rec[0][0]
        check(rec_round >= 10, f"recovery só após dwell de 3 rounds cheios (@{rec_round}>=10)")
        # o modo volta a federated no round seguinte ao commit
        check(tl[rec_round] == "federated", "modo volta a federated após o commit de recovery")

    # antes do 8, segue gossip (nó 7 ainda fora nos rounds 4..7)
    check(all(m == "gossip" for m in tl[3:7]), "rounds 4-7 permanecem gossip (nó 7 fora)")

    # ── cenário cooldown: duas faltas próximas não causam flap imediato ──
    mgr2 = DistributedArchManager(n_nodes=6, initial_mode="federated",
                                  dwell_rounds=1, cooldown_rounds=5,
                                  faults=[{"round": 2, "nodes": [4]},
                                          {"round": 3, "nodes": [4], "up": True}])
    tl2, reasons2 = run(mgr2, 8)
    print("  cooldown timeline:", tl2, "reasons:", reasons2)
    # cai no 2 → gossip no 3; volta no 3, mas cooldown=5 impede recovery antes do round 7
    switches = [r for (r, _, _) in reasons2]
    check(reasons2[0][1] == "node_failure", "cooldown: primeiro commit é node_failure")
    rec2 = [r for (r, kind, _) in reasons2 if kind == "recovery"]
    check(all(r - switches[0] >= 5 for r in rec2), "cooldown: recovery respeita cooldown>=5")

    print("\n" + ("DISTARCH SELFTEST OK" if ok else "DISTARCH SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
