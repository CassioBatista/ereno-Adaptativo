"""ERENO-FD-SF — ponto de entrada unificado.

Usage:
    python ereno.py central   <grasp_method> <clf_idx> <dataset_name> [folds]
    python ereno.py federated <strategy> <grasp_method> <clf_idx> <dataset_name>
                              [num_clients] [partitioner] [partitioner_arg]

Subcomandos
-----------
  central    Pipeline centralizado (GRASP + cross-validation)
  federated  Pipeline federado (GRASP + Flower 1.31)

Parâmetros comuns
-----------------
  grasp_method   : GR-G-BF | GR-G-VND | GR-G-RVND | F-G-VND | F-G-RVND | I-G-VND
  clf_idx        : 1=RandomTree  2=J48  3=REPTree  4=NaiveBayes  5=RandomForest
  dataset_name   : dataset sem extensão  (ex: all_in_one_wsn)

Parâmetros exclusivos de 'central'
-----------------------------------
  folds          : número de folds do CV  (default: 5)

Parâmetros exclusivos de 'federated'
--------------------------------------
  strategy       : ensemble | federated_nb | xgb_bagging | xgb_cyclic
  num_clients    : clientes federados  (default: 3)
  partitioner    : iid | dirichlet | shard | exponential | linear  (default: iid)
  partitioner_arg: alpha p/ dirichlet (default 0.5) | shards_per_client p/ shard (default 2)

Exemplos
--------
  python ereno.py central   GR-G-VND 2 all_in_one_wsn
  python ereno.py central   GR-G-VND 2 all_in_one_wsn 10

  python ereno.py federated ensemble     GR-G-VND 2 all_in_one_wsn
  python ereno.py federated ensemble     GR-G-VND 2 all_in_one_wsn 5 dirichlet 0.3
  python ereno.py federated federated_nb GR-G-VND 4 all_in_one_wsn 3 shard 2
  python ereno.py federated xgb_bagging  GR-G-VND 2 all_in_one_wsn 3 exponential
  python ereno.py federated xgb_cyclic   GR-G-VND 2 all_in_one_wsn 3 linear
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


def cmd_federated(args: list[str]) -> None:
    """Despacha para o pipeline federado (main_fd.py)."""
    if len(args) < 4:
        _usage_exit(
            "federated requer: <strategy> <grasp_method> <clf_idx> <dataset_name>"
        )
    import main_fd as federated
    federated.main(args)


def main() -> None:
    if len(sys.argv) < 2:
        _usage_exit()

    subcmd = sys.argv[1].lower()
    args   = sys.argv[2:]

    match subcmd:
        case "central":
            cmd_central(args)
        case "federated":
            cmd_federated(args)
        case "help" | "--help" | "-h":
            print(__doc__)
        case _:
            _usage_exit(f"subcomando desconhecido: '{subcmd}'. Use 'central' ou 'federated'.")


if __name__ == "__main__":
    main()
