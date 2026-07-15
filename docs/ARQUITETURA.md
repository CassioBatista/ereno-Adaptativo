# Arquitetura do ereno-Adaptativo

## O que é

Pipeline de pesquisa para IDS adaptativo em aprendizado **centralizado ou
distribuído**: oferece múltiplos classificadores — incluindo XGBoost — sobre
datasets de detecção de intrusão (tráfego TI de testbed, CICIDS2017 [ref]; e
tráfego IEC-61850 gerado pelo framework ERENO, que simula a comunicação GOOSE/SV entre subestações elétricas
[ref]) e, no aprendizado distribuído, permite comutar, round a round, entre
aprendizado federado e *gossip learning* na mesma simulação, medindo o
custo/ganho de cada arquitetura contra um baseline monolítico. Construído
sobre o framework Flower 1.31+ [ref], cuja simulação utiliza o Ray [ref]
como backend de execução, com scikit-learn [ref] e XGBoost [ref]; a
implementação de *gossip learning* é baseada na estratégia GLow [ref].
Executa em Ubuntu Linux sobre o Windows Subsystem for Linux (WSL2) com
Python 3.14.

Referências: Flower — Beutel et al., 2020 (arXiv:2007.14390); Ray — Moritz
et al., OSDI 2018; scikit-learn — Pedregosa et al., JMLR 2011; XGBoost —
Chen & Guestrin, KDD 2016; GLow — Belenguer, Pascual & Navaridas, JPDC 2026
(arXiv:2501.10463); CICIDS2017 — Sharafaldin et al., ICISSP 2018; ERENO —
Quincozes et al.

## Os três pipelines (separados por desenho)

### 1. Seleção de features — `python ereno.py grasp` (uma vez por dataset)

GRASP (construção aleatorizada + busca local VND com bit-flip/IWSS/IWSSR,
avaliação por CV de 5 folds) executado **até a convergência** (K iterações
sem melhora global), em subamostra estratificada para datasets grandes,
com classes ultra-raras filtradas e seed fixado. O vencedor é persistido em
`features/<dataset>.json` com metadados auditáveis (método, avaliador,
avaliações, tempo, classes descartadas). Estado: matriz 2×2 completa —
seleções canônicas J48 (ERENO: 8 features; CICIDS: 13) + variantes XGBoost
de estabilidade, com núcleos de consenso entre avaliadores identificados.

### 2. Simulação distribuída — `python ereno.py distributed`

Nunca executa seleção de features (carrega `grasp.features_file`). Fluxo:

```
features_file (JSON) ──► particionamento (rótulos multi-classe) ──► split
        │                  iid|dirichlet|shard|exp|linear|attack      │
        ▼                                                             ▼
   filtro de colunas                                     test_file do autor (ERENO)
                                                         ou 80/20 estratificado (CICIDS)
        └──► binarização (normal=0/ataque=1) + scale_pos_weight por cliente
                    └──► run_simulation (Flower/Ray)
```

Dentro da simulação, a **HybridStrategy** consulta o **ArchitectureManager**
a cada round (cronograma YAML: `{from, to, mode, active_clients}`) e delega:

- **Federado**: `xgb_bagging` com fusão configurável — `sum` (merge de
  árvores; margens somam) ou `or` (um booster por cliente; alarme se
  qualquer um disparar) — além de `xgb_cyclic`, `ensemble`, `federated_nb`;
- **Gossip (fiel ao GLow)**: head eleito por round-robin sobre os nós
  ativos da topologia (`ring`/`chain`/`star`/grafo YAML); **os clientes
  adotam o modelo recebido e continuam o boosting** (`xgb_model=`); a
  agregação roda na Strategy como *stand-in* do head (merge `xgb`, união
  com memória deduplicada `xgb_union`, médias `inplace`/`score`). Numa
  implantação real, a agregação ocorre no cliente head (o Flower não
  suporta cliente↔cliente; o servidor é somente roteador — topologia
  física estrela, topologia lógica definida pelo roteamento);
- **Comutação**: federated→gossip semeia o modelo global em todos os nós;
  `active_clients` vale nos dois modos (mínimos ajustados por round,
  topologia sincronizada) — habilita client sampling e perda de nós;
- **Rastreabilidade**: mapeamento proxy↔partição aprendido dos
  `metrics['cid']` (os IDs do simulador são opacos).

Instrumentação: por round, `ROUND;n;modo;f1;recall;fpr` (opcionalmente em
subamostra do teste); ao final, métricas do distribuído (F1, accuracy,
precision, recall, **FPR**, matriz de confusão), baseline monolítico com
mesmas features/split, e deltas.

### 3. Centralizado — `python ereno.py central` (linhagem original)

GRASP + cross-validation, sem federação.

## Os dados

- **ERENO IEC-61850** (domínio-alvo): 5,9 M mensagens GOOSE/SV geradas pelo
  framework ERENO, que simula a comunicação IEC-61850 entre subestações elétricas (Kaggle/Quincozes), convertidas
  para 58 features numéricas — 11 atributos nominais descartados
  (identificadores do ambiente experimental) e 3 marcadores de tempo
  absoluto fora da RCL de seleção, justificados atributo a atributo;
  **split train/test do autor respeitado** (estrutura de blocos — re-split
  aleatório vaza; comprovado: CV interna ~99% vs teste real ~41%);
  93,4% normal (14:1);
- **CICIDS v2** (baseline de comparabilidade): 1,3 M fluxos regenerados dos
  arquivos originais do CICIDS2017 (proveniência provada por soma de
  colunas; receita versionada), prevalência realista de 79,5% benigno;
- Arquivos grandes fora do git, com receitas reprodutíveis, hashes e
  limitações em [DATASETS.md](DATASETS.md) e
  [PREPARACAO_DADOS.md](PREPARACAO_DADOS.md).

## Extensibilidade a outros datasets

O pipeline é agnóstico ao dataset — os dois em uso são escolha
experimental, não limite. **Disponíveis no repositório sem código novo**:
WSN-DS (`all_in_one_wsn.csv`, 374 mil amostras) e NSL-KDD
(`all_in_one_kdd.csv`, 148 mil), com classes de features registradas —
basta a seleção one-shot (`ereno.py grasp`) e um YAML de experimento.
Para um dataset inteiramente novo:

1. converter para o formato all-in-one (ARFF numérico `F1..FN` + classe
   por último; primeira linha da classe normal) — o
   `scripts/build_ereno_dataset.py` serve de gabarito, inclusive para as
   decisões de descarte de atributos;
2. registrar a classe de features em `python/feature_subsets/` (~10
   linhas: RCLs) e o nome no seletor de datasets;
3. rodar a seleção one-shot → `features/<dataset>.json`;
4. opcionais conforme o dado: `dataset.test_file` (splits pré-definidos /
   estrutura de blocos), `grasp.sample` (datasets grandes) e a
   documentação de proveniência/higiene em `docs/DATASETS.md`.

## Ambiente de execução

O projeto roda em **Ubuntu Linux sobre o WSL2** (Windows Subsystem for
Linux) — não por preferência, mas por necessidade: o motor de simulação
do Flower depende do **Ray** como backend, e o Ray não publica pacote
para Windows no Python 3.14. O fluxo de desenvolvimento reflete isso —
edição e validação de lógica no Windows, execução das simulações e do
GRASP no WSL (venv `venv-ereno314`, projeto em `~/ereno-Adaptativo`).

Como o WSL2 roda num contêiner leve, tem limites próprios (memória
configurável em `.wslconfig`) e é sensível à suspensão do Windows — em
execuções longas, convém manter a máquina ligada e usar os scripts de
retomada idempotente (`scripts/resume_grasp_selections.sh`).

## Particionadores

`iid`, `dirichlet(α)`, `shard`, `exponential`, `linear` e **`attack`**:
1 classe de ataque por cliente + fatia IID de benignos (sensores
especialistas), com três regimes conforme N — agrupamento (N < nº de
ataques), 1:1 estrito, e sensores redundantes (N > nº de ataques: classes
maiores ganham 2 sensores com metades disjuntas).

## Configuração

Precedência CLI > YAML > default. Cenários prontos em `conf/experiments/`
(federado/gossip/misto × 2 datasets), todos apontando para as features
canônicas via `grasp.features_file`.

## Achados experimentais estabelecidos

1. Sob especialização extrema por ataque, a fusão por soma de margens
   colapsa o recall (veto da maioria) e a união OR o recupera ao custo de
   FPR — três pontos de operação mapeados contra o monolítico;
2. O gossip portado era sem memória (curvas periódicas, período = tamanho
   do anel) — corrigido com adoção no cliente + pool acumulativo; difusão
   comprovada;
3. No ERENO com partição IID, o bagging federado superou o monolítico em
   +15 pp de F1 sob *distribution shift* (1 seed; a campanha confirmará);
4. Núcleos de consenso de features entre famílias de avaliador (ERENO:
   SqNum, StNum, cbStatus, gooseTimeAllowedtoLive, timeFromLastChange;
   CICIDS: 7 features de fluxo, incluindo Destination Port e
   Init_Win_bytes_backward).

## Pendências deliberadas

Teto de árvores do merge (crescimento multiplicativo), votação k-de-n,
`ApiArchManager` (comutação dirigida por API REST), split deduplicado do
CICIDS, warm-start do GRASP, e a campanha completa do
[PLANO_VALIDACAO.md](PLANO_VALIDACAO.md).
