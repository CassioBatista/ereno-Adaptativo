# Preparação de Dados — pipeline completo

Documenta todos os passos entre os dados brutos e o treinamento distribuído:
**(1) preparação dos datasets → (2) seleção de features → (3) particionamento**,
com os pontos de implementação e as decisões metodológicas de cada etapa.

```
dados brutos ──► all_in_one_*.csv ──► GRASP ──► filtro de features ──► partições
 (originais)      (ARFF numérico)    (fase 1)   (colunas do vencedor)   por cliente
                                                                            │
                        avaliação final ◄── split global / test_file ◄─────┤
                        (teste íntegro)                                    ▼
                                                              splits locais 80/20
                                                              + binarização (xgb)
```

---

## 1. Preparação dos datasets

Todos os datasets usam o mesmo formato de entrada: **ARFF com features
numéricas `F1..FN` + atributo de classe** (`@class@`) por último. O loader é
[python/util.py](../python/util.py) → `load_arff()`: devolve `X` (float64),
`y` (índices inteiros das classes) e a lista de nomes de classe. A convenção
`normal_class` (classe "tráfego normal") é o rótulo da **primeira linha** do
arquivo — todos os all-in-one começam com uma amostra normal/BENIGN.

### 1.1 CICIDS v2 (`all_in_one_cicids_v2.csv`) — dataset TI

- **Origem**: 3 dias do CICIDS2017 oficial (formato MachineLearningCVE,
  78 features do CICFlowMeter). Proveniência **provada** por soma de colunas
  (390/390 idênticas) — ver [DATASETS.md](DATASETS.md).
- **Receita**: [scripts/regenerate_cicids_v2.py](../scripts/regenerate_cicids_v2.py):
  1. concatena Tuesday + Wednesday + Thursday-Morning;
  2. normaliza rótulos (`DoS Hulk`→`hulk`, `Web Attack XSS`→`xss`, ...);
  3. **higiene**: remove linhas com NaN/Infinity (1.696 = 0,13%);
  4. **sem teto por classe** (a v1 antiga limitava BENIGN e hulk a ~10 mil,
     invertendo a prevalência para 83% de ataques);
  5. grava ARFF com as 78 features na ordem canônica.
- **Resultado**: 1.307.282 amostras, **79,5% BENIGN** (prevalência realista),
  11 classes.
- **Split**: aleatório estratificado 80/20 em tempo de execução (seed do YAML).
  *Limitação documentada*: fluxos duplicados do CICIDS atravessam o split
  (~73% do teste tem cópia exata no treino) e as partições dos clientes são
  sorteadas do arquivo inteiro — métricas absolutas inflam; comparações
  entre arquiteturas permanecem válidas (mesma inflação em todos os braços).

### 1.2 ERENO IEC-61850 (`all_in_one_ereno_train/test.csv`) — dataset alvo

- **Origem**: Kaggle `sequincozes/ereno-iec61850-ids` (download anônimo via
  `kagglehub`), tráfego GOOSE/SV gerado pelo framework ERENO, que simula a comunicação IEC-61850 entre subestações elétricas.
- **Receita**: [scripts/build_ereno_dataset.py](../scripts/build_ereno_dataset.py):
  1. lê os `train.arff`/`test.arff` do autor (~2,96 M linhas cada);
  2. **mantém as 58 features numéricas** (elétricas: correntes/tensões/RMS/
     áreas trapezoidais; protocolo: SqNum, StNum, frames; derivadas: stDiff,
     sqDiff, timestampDiff, delay...);
  3. **descarta os 11 atributos nominais**, justificados um a um:

     | Atributo | O que é | Justificativa da remoção |
     |---|---|---|
     | `ethDst` | Endereço MAC destino | Identificador de hardware específico da rede experimental — não generaliza para outras redes |
     | `ethSrc` | Endereço MAC origem | Mesmo motivo do `ethDst` |
     | `ethType` | Tipo Ethernet (0x88b8 = GOOSE) | Constante estrutural do protocolo, não característica de comportamento |
     | `gooseAppid` | AppID do publicador GOOSE | Constante de configuração do ambiente simulado |
     | `TPID` | Tag VLAN (0x8100) | Constante de configuração de rede — não varia com o comportamento |
     | `gocbRef` | Referência do bloco de controle GOOSE | Identificador de configuração da subestação — específico do ambiente de captura |
     | `datSet` | Nome do dataset GOOSE | Rótulo de configuração, não característica do tráfego em si |
     | `goID` | Identificador GOOSE | ID fixo de configuração — não varia com ataques |
     | `test` | Flag de teste (IEC 61850) | Constante (FALSE) no tráfego legítimo do testbed |
     | `ndsCom` | Flag *needs commissioning* | Estado de configuração, constante no ambiente |
     | `protocol` | Tipo da mensagem (GOOSE/SV) | Indicador estrutural, não comportamental |

     **Evidência medida** (amostra de 147.532 linhas do train): todos os
     11 são quase-constantes — um único valor cobre 99,997% das linhas
     (na taxa base de 6,6% de ataque) — e **todos os desvios raros (1–5
     ocorrências) são 100% ataque** (frames forjados pelo gerador com
     MACs broadcast, ethertype 0x77b7, appid/TPID alternativos, flags
     TRUE). Consequências: (a) não separam nada no grosso dos dados — os
     ataques abundantes usam os valores legítimos por definição; (b) onde
     desviam, são vazamento de artefato do testbed (memorização de
     identidade, não de comportamento); (c) em implantação real são
     forjáveis trivialmente, e as mesmas mensagens anômalas já alteram
     campos numéricos mantidos (frameLen, APDUSize, gooseLengthDiff...) —
     o sinal legítimo não se perde;
  4. zero descartes por higiene (dado gerado pelo framework é limpo).
  5. **Exclusão adicional na SELEÇÃO** (não no arquivo): os marcadores de
     posição temporal absoluta — `F1 Time`, `F38 t`, `F39 GooseTimestamp`
     — ficam fora das RCLs do GRASP (`python/feature_subsets/ereno.py`),
     pela mesma justificativa dos nominais: posição na linha do tempo da
     simulação é artefato do ambiente, não comportamento — não generaliza
     e explora o viés de blocos da CV interna (a primeira seleção
     definitiva elegeu `t`, confirmando o risco). As temporais
     **relativas** (`timestampDiff`, `tDiff`, `timeFromLastChange`,
     `delay`) permanecem elegíveis — são comportamentais.
- **Resultado**: 8 classes (normal + 7 ataques GOOSE), **93,4% normal**
  (desbalanceamento ~14:1).
- **Split**: **o train/test do autor é preservado** (`dataset.test_file` no
  YAML). Motivo: cada GOOSE periódico acompanha ~4.800 mensagens SV — as
  features GOOSE se repetem em blocos, e re-split aleatório vaza.
  *Comprovação empírica*: CV aleatória interna do GRASP reporta F1 ~99%;
  o test set em blocos do autor, ~41% — a diferença É o vazamento.

### 1.3 Binarização (estratégias XGBoost)

O cliente XGBoost treina `binary:logistic`; os rótulos multi-classe são
binarizados (**normal=0, ataque=1**) em [main_dist.py](../main_dist.py) —
**depois do particionamento**, para os particionadores por classe
(`dirichlet`, `shard`, `attack`) enxergarem as classes originais.
O desbalanceamento local é compensado por `scale_pos_weight = n_neg/n_pos`
calculado **por cliente** ([fd/client_xgb.py](../fd/client_xgb.py)) — as
classes passam a pesar igual na perda sem duplicar dados (o baseline
monolítico usa o mesmo mecanismo, para comparação justa).

---

## 2. Seleção de features (GRASP — fase 1, centralizada)

Roda **uma única vez, antes** da simulação distribuída
(`run_grasp()` em [main_dist.py](../main_dist.py)); o subconjunto vencedor
fica congelado durante todos os rounds.

| Passo | O quê | Onde |
|---|---|---|
| RCL | Lista de candidatas conforme o método: `GR-*` = ranking Gain Ratio; `F-*` = todas; `I-*` = subconjunto IWSSR do classificador | `python/feature_subsets/<dataset>.py` |
| Construção | Sorteia `num_features` (5) da RCL — aleatorizada a cada iteração (multi-start) | `python/grasp/base.py` |
| Avaliação | Cross-validation 5 folds com o classificador escolhido (J48 etc.); métrica-critério: F1 | `python/grasp/base.py::avaliar` |
| Busca local | VND com estruturas bit-flip, IWSS, IWSSR (pode mudar a cardinalidade) | `python/grasp/local_searches.py` |
| Parada | `grasp.max_iterations` do YAML (e limites de avaliações/tempo) | `conf/*.yaml` |

**Subamostra para datasets grandes** (`grasp.sample` no YAML): a seleção
roda numa amostra **estratificada** (ex.: 100–150 mil linhas); o
treinamento distribuído usa o dataset completo. Implementado em
`run_grasp()` — reduz o custo de cada avaliação de ~40–90 s para ~1–2 s no
ERENO.

**Ressalvas metodológicas (importantes para a escrita):**

1. A CV interna do GRASP é aleatória → em dados com estrutura de blocos
   (ERENO), o F1 *interno* do GRASP é inflado. Isso afeta apenas o critério
   de seleção; a avaliação final usa o test set íntegro.
2. No caminho sem `test_file` (CICIDS), o GRASP vê o arquivo inteiro —
   incluindo o que depois vira teste (vazamento de seleção, pequeno mas
   real). Com `test_file` (ERENO), o GRASP vê só o train. ✔
3. Saída sempre registrada no log: `Features selecionadas (N): [...]` —
   usada pelo sanity check V5 do plano de validação.

---

## 3. Particionamento (dados → clientes)

Implementado em [fd/dataset.py](../fd/dataset.py) → `load_and_partition()`.
Recebe o dataset **já filtrado pelas features do GRASP** e os rótulos
**multi-classe originais** (a binarização vem depois). Particionadores:

| Nome | Semântica | Argumento |
|---|---|---|
| `iid` | Sorteio uniforme, fatias iguais — baseline homogêneo | — |
| `dirichlet` | Heterogeneidade por classe via Dirichlet(α): α≪1 = cada cliente vê poucas classes; α≫1 ≈ IID | `alpha` (default 0,5) |
| `shard` | Ordena por classe e distribui shards contíguos (não-IID clássico do FedAvg) | `shards_per_client` (2) |
| `exponential` | Volumes exponencialmente desiguais (cliente 0 maior) | — |
| `linear` | Volumes linearmente desiguais | — |
| `attack` | **1 classe de ataque por cliente** + fatia IID dos benignos (sensores especialistas). Exige `num_clients ≤ nº de ataques`; excedentes em round-robin com aviso | — |

Ordem completa das operações em `main_dist.py`:

```
1. GRASP  ──────────────► índices das features vencedoras
2. load_and_partition ──► N partições (rótulos multi-classe)   ← particionador
3. split global:
     sem test_file: 80/20 estratificado do arquivo inteiro (seed do YAML)
     com test_file: train inteiro p/ baseline; teste = arquivo do autor
4. split local por cliente: 80/20 (treino/validação local)
5. binarização (só xgb): normal=0 / ataque=1, em y_tr, y_te e nos splits
6. clientes Flower recebem (X_train, y_train, X_test, y_test) locais
```

**Rastreabilidade cliente↔partição**: sob o simulador Flower, os IDs de
proxy são opacos; o mapeamento para a partição real é aprendido dos
`metrics['cid']` reportados pelos clientes (`fd/strategy/hybrid_strategy.py`
+ `glow_strategy.resolve_node_index`) — é o que garante que
`active_clients: [0, 1]` do schedule selecione as partições de dados 0 e 1.

**Reprodutibilidade**: todos os sorteios (particionador, splits, subamostra
do GRASP, XGBoost) derivam do `seed` do YAML (+ `GRASP_SEED` em
`python/config.py`). Mesma configuração + mesmo seed ⇒ mesmas partições e
métricas (verificado: execução real Flower = mini-simulação em processo,
dígito por dígito).

---

## Pendências conhecidas (backlog)

- Split deduplicado para o CICIDS (remover fluxos duplicados antes do
  split) como análise de robustez;
- Particionar apenas o lado de treino no caminho sem `test_file` (hoje as
  partições vêm do arquivo inteiro — sobreposição treino-cliente × teste
  global de ~16%);
- Variante `benign: shared|bootstrap` no particionador `attack`;
- Eliminar a dupla leitura do ARFF (GRASP e particionamento carregam o
  mesmo arquivo duas vezes — ~15 min extras no ERENO).
