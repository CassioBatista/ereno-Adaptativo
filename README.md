# ereno-Adaptativo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22772409.svg)](https://doi.org/10.5281/zenodo.22772409)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pipeline adaptativo de detecção de intrusão (IDS) que combina **aprendizado centralizado, federado e gossip** em uma única execução, com **comutação de modo round a round**. Evolução do ERENO-FD-SF incluindo Gossip Learning baseado em implementação do framework Glow e construído sobre [Flower](https://flower.ai/) 1.31, scikit-learn e XGBoost.

A ideia central: em vez de escolher *a priori* entre treinar de forma federada (coordenação por servidor central) ou por gossip (troca de modelos entre vizinhos numa topologia), um **Architecture Manager** decide o modo de cada round. Isso permite simular cenários reais onde a infraestrutura muda durante o treinamento — por exemplo, perda do servidor central forçando os nós a continuar por gossip, ou nós que entram e saem da rede.

## Como funciona

```
┌─────────────────────────────────────────────────────────────────┐
│ ① GRASP (centralizado)      seleção de features no dataset      │
│ ② Particionamento           dados divididos entre N clientes    │
│                             (iid | dirichlet | shard | ...)     │
│ ③ Treinamento distribuído   Flower run_simulation()             │
│                                                                 │
│    round 1..10  ── federated ──► servidor agrega todos os nós   │
│    round 11..20 ── gossip ─────► head recebe modelos dos        │
│                                  vizinhos e agrega localmente   │
│         ▲                                                       │
│         └── ArchitectureManager decide o modo de cada round     │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Seleção de features (GRASP)

Antes da distribuição, a meta-heurística **GRASP** roda centralizada no dataset completo e seleciona o subconjunto de features usado por todos os clientes. Métodos disponíveis: `GR-G-BF`, `GR-G-VND`, `GR-G-RVND`, `F-G-VND`, `F-G-RVND`, `I-G-VND`, avaliados com um dos classificadores base (RandomTree, J48, REPTree, NaiveBayes ou RandomForest).

### 2. Particionamento

O dataset filtrado é dividido entre os clientes com um dos particionadores: `iid`, `dirichlet` (não-IID, controlado por `alpha`), `shard` (`shards_per_client`), `exponential` ou `linear` — permitindo estudar o efeito de dados heterogêneos.

### 3. Treinamento distribuído adaptativo

Três componentes cooperam dentro de uma única `run_simulation()` do Flower:

- **`ArchitectureManager`** ([fd/arch_manager.py](fd/arch_manager.py)) — responde "qual o modo do round R?" e, opcionalmente, "quais clientes estão ativos?". `FixedArchManager` lê um cronograma estático do YAML; **na v2, `DistributedArchManager` decide o modo automaticamente** por detecção descentralizada de falhas (ver [Novidades na v2](#novidades-na-v2)).

- **`HybridStrategy`** ([fd/strategy/hybrid_strategy.py](fd/strategy/hybrid_strategy.py)) — meta-estratégia do Flower que delega `configure_fit`/`aggregate_fit` para a sub-estratégia do modo ativo. Na **transição de modo**, `_transfer_model()` preserva a continuidade do treinamento:
  - *federated → gossip*: o modelo global vira o ponto de partida local de todos os nós;
  - *gossip → federated*: os modelos por nó são promediados num novo modelo global inicial.

- **Sub-estratégias**:
  - **Federado** — `ensemble` (voto majoritário de modelos locais), `federated_nb` (agregação exata de GaussianNB), `xgb_bagging` e `xgb_cyclic` (XGBoost);
  - **Gossip** — `GlowStrategy` ([fd/strategy/glow_strategy.py](fd/strategy/glow_strategy.py)): a cada round um nó é eleito **head** por round-robin sobre os nós ativos da topologia; o head recebe os modelos dos vizinhos e agrega por média ponderada por amostras (`inplace`), por acurácia local (`score`) ou por concatenação de árvores XGBoost (`xgb`). Cada nó mantém seu próprio modelo entre rounds.

As topologias de gossip ficam em [conf/topologies/](conf/topologies/): `ring`, `chain`, `star` ou um grafo arbitrário descrito em YAML.

### Avaliação

Ao final, o modelo distribuído é avaliado num conjunto de teste global e comparado com um **baseline centralizado treinado com as mesmas features**, reportando F1-score, accuracy, precision, recall e a matriz de confusão — ou seja, o pipeline mede diretamente o custo (ou ganho) de distribuir o treinamento.

## Novidades na v2

A v2 transforma a comutação de **cronograma estático** em **distribuída e automática**, dirigida por **detecção descentralizada de falhas** sobre o substrato de gossip GLow, e adiciona observabilidade externa. (A detecção/trust byzantina fica para a v3 — ver [docs/v3_trust_model.md](docs/v3_trust_model.md).)

- **`DistributedArchManager`** ([fd/arch_manager.py](fd/arch_manager.py)) — comutação FL↔GL **automática**, sem cronograma. Cada nó detecta falhas localmente; o cronograma `faults` é apenas *ground truth* do ambiente que os controladores **detectam**, nunca leem.
- **Detecção descentralizada** sobre o GLow:
  - [fd/peer_failure.py](fd/peer_failure.py) — timeout local por vizinho + difusão de suspeitas (`min_witnesses`, agreement-quorum);
  - [fd/vote_diffusion.py](fd/vote_diffusion.py) — votos de modo reusam a difusão head-election do GLow (quórum descentralizado, sem tally central);
  - **FL→GL fail-fast** (comita na corroboração local por `min_witnesses`, **sem consenso de rede**) vs **GL→FL careful** (dwell local por nó + **unanimidade de saúde local**, sã sob partição do overlay por nós mortos). O digest de controle (bitmaps de voto + suspeita) viaja nas mensagens `FitIns` reais do `GlowStrategy`.
- **Semeadura por união retida** na transição FL→GL — um nó que cai pouco antes da troca ainda tem seu booster no pool GL, recuperando o F1 ao nível saudável.
- **Monitor externo** ([fd/monitor.py](fd/monitor.py)) — REST *pull* (`/events`, `/status`, `/health`) + trilha de auditoria JSONL; eventos `architecture_change` e `node_failure` (qual nó falhou). Docs da API JSON/CoAP: [docs/API.md](docs/API.md), [docs/openapi.yaml](docs/openapi.yaml).
- **Curvas endógenas ao vivo** ([scripts/plot_endogenous_live.py](scripts/plot_endogenous_live.py)) — o round da troca FL↔GL é produzido pela detecção descentralizada no mesmo run que mede o F1 (descida fail-fast, subida careful). Self-tests: `scripts/distarch_decentralized_selftest.py`, `scripts/peer_failure_selftest.py`, `scripts/vote_diffusion_selftest.py`.

Config da v2 (ver [conf/experiments/](conf/experiments/) `*_decentralized*.yaml`):

```yaml
architecture:
  manager: distributed
  decentralized: true      # detecção descentralizada (fail-fast / unanimidade)
  topology: ring
  peer_timeout: 2
  min_witnesses: 1         # 1 = crash-trusted (v2); Byzantine (>=2) -> v3
  dwell_rounds: 3
  cooldown_rounds: 2
```

Detalhes em [docs/ARQUITETURA_v2.md](docs/ARQUITETURA_v2.md) e [docs/B1_live_run_gap.md](docs/B1_live_run_gap.md).

## Uso

Ponto de entrada unificado: [ereno.py](ereno.py).

```bash
# Centralizado (GRASP + cross-validation)
python ereno.py central GR-G-VND 2 all_in_one_wsn

# Distribuído — federado puro
python ereno.py distributed GR-G-VND 2 all_in_one_wsn --strategy ensemble --clients 3

# Distribuído — gossip puro em anel
python ereno.py distributed GR-G-VND 2 all_in_one_wsn --topology ring --rounds 20

# Distribuído — híbrido adaptativo (cronograma no conf/base.yaml)
python ereno.py distributed GR-G-VND 2 all_in_one_wsn --conf conf/base.yaml

# Distribuído — dados não-IID (Dirichlet)
python ereno.py distributed GR-G-VND 2 all_in_one_wsn --partitioner dirichlet --partitioner-arg 0.3
```

Argumentos posicionais: `<grasp_method> <clf_idx> <dataset_name>`, onde `clf_idx` é 1=RandomTree, 2=J48, 3=REPTree, 4=NaiveBayes, 5=RandomForest e `dataset_name` é o CSV sem extensão (ex.: `all_in_one_wsn`).

## Configuração

O cronograma de comutação é definido em [conf/base.yaml](conf/base.yaml):

```yaml
architecture:
  manager: fixed          # fixed | api (futuro)
  default_mode: federated
  schedule:
    - {from: 1,  to: 10, mode: federated}
    # active_clients é opcional — omitir usa todos os nós
    - {from: 11, to: 20, mode: gossip, active_clients: [0, 1, 2]}
```

O mesmo arquivo configura dataset, particionador, topologia, número de rounds e a estratégia federada.

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `ereno.py` | CLI unificada (`central` \| `distributed`) |
| `main.py` | pipeline centralizado (GRASP + CV) |
| `main_dist.py` | pipeline adaptativo (GRASP + Flower + HybridStrategy) |
| `main_fd.py` | pipeline federado simples (sem comutação) |
| `fd/` | estratégias Flower, topologias, particionamento, clientes |
| `python/` | GRASP, classificadores, subconjuntos de features, avaliação |
| `conf/` | configuração base e topologias de gossip |

## Datasets

Datasets de trabalho: **CICIDS2017** (regenerado dos originais — `all_in_one_cicids_v2`) e **ERENO IEC-61850** (GOOSE/SV, domínio-alvo), além de WSN, NSL-KDD e SWaT no formato "all-in-one". Subconjuntos de features em `python/feature_subsets/`.

## Documentação

| Documento | Conteúdo |
|---|---|
| [docs/ARQUITETURA.md](docs/ARQUITETURA.md) | visão geral do sistema: pipelines, dados, estratégias, achados e pendências |
| [docs/RESULTADOS.md](docs/RESULTADOS.md) | resultados ERENO: centralizado × federado × gossip, métricas por cliente e curvas de convergência |
| [docs/GOSSIP_DESIGN.md](docs/GOSSIP_DESIGN.md) | desenho do gossip learning: fidelidade ao GLow, operadores de agregação XGBoost, análise k-de-n |
| [docs/PREPARACAO_DADOS.md](docs/PREPARACAO_DADOS.md) | pipeline de dados completo: preparação dos datasets, seleção de features (GRASP) e particionamento |
| [docs/DATASETS.md](docs/DATASETS.md) | proveniência, hashes de verificação e receitas de regeneração |
| [docs/PLANO_VALIDACAO.md](docs/PLANO_VALIDACAO.md) | plano de validação e testes: matriz de experimentos, métricas e sanity checks |
| [docs/ARQUITETURA_v2.md](docs/ARQUITETURA_v2.md) | **v2**: comutação distribuída/automática, semeadura por união retida, monitor externo |
| [docs/B1_live_run_gap.md](docs/B1_live_run_gap.md) | **v2**: detecção descentralizada ao vivo (fail-fast / unanimidade) e limites (Gap 3/4) |
| [docs/v3_trust_model.md](docs/v3_trust_model.md) | **v3**: modelo de confiança unificado de 3 drivers (availability, concordância, veredito do IDS) |

## Instalação

```bash
pip install -r requirements.txt
```

Solução desenvolvida em Python 3.14.

## Referências

- Beutel, D. J., Topal, T., Mathur, A., Qiu, X., Fernandez-Marques, J., Gao, Y., ... & Lane, N. D. (2020). Flower: A friendly federated learning research framework. *arXiv preprint* arXiv:2007.14390.
- Belenguer, A., Pascual, J. A., & Navaridas, J. (2026). GLow — A Novel, Flower-Based Simulated Gossip Learning Strategy. *Journal of Parallel and Distributed Computing*, 105272.

## Licença

Distribuído sob a licença **MIT** — ver [LICENSE](LICENSE).

## Como citar

Se usar este software, cite-o conforme [CITATION.cff](CITATION.cff). Cada release
é arquivada no Zenodo com um DOI:

- **v2.0.0** — [10.5281/zenodo.22772409](https://doi.org/10.5281/zenodo.22772409)

Histórico de versões em [CHANGELOG.md](CHANGELOG.md).

> Nota: o DOI acima é o desta versão. Se você criar o *concept DOI* (all-versions,
> que sempre aponta para a última release) no Zenodo, troque o badge por ele.
