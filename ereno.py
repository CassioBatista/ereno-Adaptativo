"""ERENO-FD-SF-GLow — ponto de entrada unificado.

Usage:
    python ereno.py central      <grasp_method> <clf_idx> <dataset_name> [folds]
    python ereno.py distributed  <grasp_method> <clf_idx> <dataset_name> [opções]

Subcomandos
-----------
  central      Pipeline centralizado (GRASP + cross-validation)
  distributed  Pipeline adaptativo  (GRASP + Flower 1.31 + HybridStrategy)
               Suporta comutação round-a-round entre federated e gossip.

Parâmetros comuns
-----------------
  grasp_method   : GR-G-BF | GR-G-VND | GR-G-RVND | F-G-VND | F-G-RVND | I-G-VND
  clf_idx        : 1=RandomTree  2=J48  3=REPTree  4=NaiveBayes  5=RandomForest
  dataset_name   : dataset sem extensão  (ex: all_in_one_wsn)

Parâmetros exclusivos de 'central'
-----------------------------------
  folds          : número de folds do CV  (default: 5)

Parâmetros exclusivos de 'distributed'
---------------------------------------
  --conf PATH          : arquivo de configuração YAML  (default: conf/base.yaml)
  --strategy NAME      : ensemble | federated_nb | xgb_bagging | xgb_cyclic
  --topology NAME      : star | ring | chain | <caminho.yaml>
  --clients N          : número de agentes/clientes  (default: lido do conf)
  --rounds N           : total de rounds de simulação  (default: lido do conf)
  --partitioner NAME   : iid | dirichlet | shard | exponential | linear | attack
  --partitioner-arg V  : alpha (dirichlet) ou shards_per_client (shard)

  O schedule de comutação federated↔gossip é definido em conf/base.yaml:
    architecture:
      manager: fixed
      schedule:
        - {from: 1,  to: 10, mode: federated}
        - {from: 11, to: 20, mode: gossip}

Exemplos
--------
  # Centralizado
  python ereno.py central GR-G-VND 2 all_in_one_wsn
  python ereno.py central GR-G-VND 2 all_in_one_wsn 10

  # Distribuído — federado puro (schedule só federated)
  python ereno.py distributed GR-G-VND 2 all_in_one_wsn --strategy ensemble --clients 3

  # Distribuído — gossip puro (schedule só gossip)
  python ereno.py distributed GR-G-VND 2 all_in_one_wsn --topology ring --rounds 20

  # Distribuído — híbrido adaptativo (definido no conf/base.yaml)
  python ereno.py distributed GR-G-VND 2 all_in_one_wsn --conf conf/base.yaml

  # Distribuído — particionamento não-IID
  python ereno.py distributed GR-G-VND 2 all_in_one_wsn --partitioner dirichlet --partitioner-arg 0.3
"""

import sys


def _usage_exit(msg: str = "") -> None:
    if msg:
        print(f"Erro: {msg}\n")
    print(__doc__)
    sys.exit(1)


def cmd_central(args: list[str]) -> None:
    """Despacha para o pipeline centralizado (main.py)."""
    if len(args) < 3:
        _usage_exit("central requer: <grasp_method> <clf_idx> <dataset_name>")
    import main as central
    central.main(args)


def cmd_distributed(args: list[str]) -> None:
    """Despacha para o pipeline adaptativo (main_dist.py)."""
    if len(args) < 3:
        _usage_exit("distributed requer: <grasp_method> <clf_idx> <dataset_name>")
    import main_dist as distributed
    distributed.main(args)


def main() -> None:
    if len(sys.argv) < 2:
        _usage_exit()

    subcmd = sys.argv[1].lower()
    args   = sys.argv[2:]

    match subcmd:
        case "central":
            cmd_central(args)
        case "distributed":
            cmd_distributed(args)
        case "help" | "--help" | "-h":
            print(__doc__)
        case _:
            _usage_exit(
                f"subcomando desconhecido: '{subcmd}'. "
                f"Use 'central' ou 'distributed'."
            )


if __name__ == "__main__":
    main()
