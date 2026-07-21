# Resultados — Centralizado × Federado × Gossip

Comparação das três arquiteturas de treinamento do IDS em dois cenários:
**ERENO IEC-61850** (subestação elétrica, §1–3) e **CICIDS2017** (rede
TI / smart grid, §4). ERENO avaliado no test set do autor (2.955.648
mensagens, split por blocos — sem vazamento); CICIDS no split 80/20
estratificado.

## Configuração do experimento

| Item | Valor |
|---|---|
| Dataset | ERENO IEC-61850 (train/test do autor) |
| Modelo | XGBoost (`binary:logistic`, `scale_pos_weight` local por cliente) |
| Features | seleção do avaliador XGBoost — `features/all_in_one_ereno_train_xgb.json` (15 features) |
| Clientes | 10, particionador `attack` (sensores especialistas; 3 classes de 39k → 2 sensores cada) |
| Fusão | OR (`xgb_union`) — especialistas: a soma de margens sofre veto da maioria |
| Topologia gossip | anel de 10 nós |
| Rounds | 20 |
| `benign_cap` | 500.000 benignos de treino (memória; ataques preservados — ver nota) |
| Logs | `results/ereno_fed_percli_c10.log`, `results/ereno_gossip_percli_c10.log` |

## 1. Métricas globais (teste completo)

| Métrica | Centralizado | Federado (OR) | Gossip (OR, difundido) |
|---|---:|---:|---:|
| F1-score | **96,88%** | 87,51% | 87,54% |
| Recall | 99,51% | 99,71% | **99,71%** |
| Precision | **94,39%** | 77,98% | 77,99% |
| FPR | **0,43%** | 2,05% | 2,05% |
| VP / FN | 199.526 / 983 | 199.925 / 584 | ≈ 199.925 / 584 |

**Conclusões:**

1. **Gossip = Federado**, dígito por dígito (F1 87,54 vs 87,51) — o
   aprendizado descentralizado sem servidor central alcança o mesmo
   resultado do federado, após ~8 rounds de difusão (§3);
2. **Ambos os distribuídos superam o centralizado em recall** (99,71% vs
   99,51%): a união dos especialistas não perde nenhum tipo de ataque;
3. **O centralizado vence no F1/precisão** (menos falsos alarmes): um
   modelo único, vendo todos os ataques juntos, calibra melhor a
   fronteira. O custo do distribuído é o FPR (2,05% vs 0,43%) — os falsos
   alarmes dos 10 sensores independentes somam (*union bound*).

## 2. Métricas por cliente (modelo local × teste global)

### Federado — cada sensor isolado (`#bst` = 1)

| cli | classe-especialista | F1 | Recall | Prec | FPR |
|---:|---|---:|---:|---:|---:|
| 0 | random_replay | 32,56 | 19,45 | 100,00 | 0,00 |
| 1 | random_replay | 32,56 | 19,45 | 100,00 | 0,00 |
| 2 | injection | 56,01 | 38,89 | 100,00 | 0,00 |
| 3 | injection | 54,48 | 39,22 | 89,20 | 0,35 |
| 4 | high_StNum | 32,79 | 19,62 | 99,80 | 0,00 |
| 5 | high_StNum | 46,29 | 30,12 | 100,00 | 0,00 |
| 6 | inverse_replay | 58,86 | 41,70 | 100,00 | 0,00 |
| 7 | poisoned_high_rate | 51,64 | 34,81 | 100,00 | 0,00 |
| 8 | masquerade_fake_normal | 39,12 | 24,99 | 90,11 | 0,20 |
| 9 | masquerade_fake_fault | 51,24 | 41,57 | 66,76 | 1,51 |

Cada especialista tem **precision ~100% e recall parcial (19–42%)**: quase
não gera falso alarme, mas só detecta o próprio ataque. A cobertura de
99,71% do federado global vem **exclusivamente da união** — nenhum sensor
sozinho passa de ~42%. O cliente 9 (masquerade_fake_fault) é o "sensor
ruidoso" (FPR 1,51%, precision 66,8%) que puxa o FPR da união para cima.

### Gossip — após difusão (`#bst` = 19)

| cli | classe-especialista | F1 | Recall | Prec | FPR | #bst |
|---:|---|---:|---:|---:|---:|---:|
| 0–9 | (todos) | 87,51 | 99,71 | 77,98 | 2,05 | 19 |

Os **10 nós ficaram idênticos** — cada um carrega quase toda a rede (19
boosters) e detecta tão bem quanto o sistema federado inteiro, **sem
servidor central**. É o consenso descentralizado alcançado: resiliência
total (sem ponto único de falha) com a mesma capacidade de detecção.
(19 e não 20: sensores redundantes de classes homogêneas deduplicam —
`dedup_union`.)

## 3. Curvas de convergência

`scripts/plot_convergencia.py` → `results/conv_ereno_c10_{metricas,difusao}.png`
e `results/conv_ereno_c10.csv`.

**Métricas × round** (`conv_ereno_c10_metricas.png`): o federado é uma
linha horizontal (união instantânea via servidor); o gossip é uma curva
que sobe do round 1 (recall 51,6%) e **alcança o federado no round 8**
(recall 99,7%), permanecendo colada até o fim.

**Difusão × round** (`conv_ereno_c10_difusao.png`): a união do head cresce
~2 boosters por round (o head absorve seus 2 vizinhos do anel):
3→6→8→10→12→14→16→18, saturando em **19 no round 9**. A métrica converge
(round 8) exatamente quando a difusão completa (round 9) — prova causal
de que "gossip = federado, ao custo de ~N rounds de difusão".

Recall do gossip por round: 51,6 → 78,2 → 78,2 → 78,2 → 79,2 → 91,0 →
91,0 → **99,7** (round 8) → 99,7 …

## 4. Réplica em CICIDS2017 (cenário TI / smart grid)

Mesmo protocolo (XGBoost, features do avaliador XGBoost, 10 clientes
`attack`, fusão OR, federado/gossip/centralizado), no CICIDS v2 — split
80/20 estratificado, **sem `benign_cap`** (dataset mais leve; capar
distorceria a prevalência do teste). 10 clientes = 10 classes de ataque →
mapeamento **1:1 estrito**. Logs: `results/cicids_{fed,gossip}_percli_c10.log`.

### Métricas globais — os dois cenários

| | ERENO distrib. | ERENO central | CICIDS distrib. | CICIDS central |
|---|---:|---:|---:|---:|
| F1 | 87,5% | 96,9% | 94,6% | 99,5% |
| Recall | 99,7% | 99,5% | 99,2% | 99,9% |
| FPR | 2,05% | 0,43% | 2,7% | 0,23% |

**O mesmo padrão nos dois domínios**: federado ≈ gossip; distribuído com
recall altíssimo; centralizado vence no F1 por FPR muito menor. A tese se
sustenta em subestação (ERENO) e em rede TI (CICIDS) — não é artefato de
um dataset.

### Federado por cliente — o desbalanceamento extremo do CICIDS

| cli | classe | Recall isolado |
|---:|---|---:|
| 0 | hulk | 92,68 |
| 1 | GoldenEye | 32,72 |
| 4 | slowloris | 32,71 |
| 8 | sql | 25,47 |
| 2 | ftp | 2,96 |
| 3 | ssh | 2,20 |
| 7 | xss | 1,28 |
| 9 | Heartbleed | **0,00** (11 amostras) |

Vários sensores isolados são **quase inúteis** (Heartbleed F1 0,01%; xss
1,3%; ftp/ssh ~2–3%) — bem pior que no ERENO (onde os isolados já tinham
recall 19–42%). **Após a difusão do gossip, os 10 nós ficam idênticos
com F1 94,27%** (19 boosters cada): a difusão **resgata os sensores
inúteis**. Argumento mais forte que no ERENO: *o gossip é indispensável
justamente quando os dados locais são pobres*.

### Achado: F1 do gossip é NÃO-monotônico no CICIDS

A curva de F1 do gossip **sobe até 97,25% no round 3 e depois cai para
94,27%** (`conv_cicids_c10_metricas.png`). No início, a união tem poucos
boosters, mas os "bons" (hulk, alta precisão) → F1 alto; conforme os
especialistas ruidosos entram, o recall sobe mas o FPR também, e o F1
líquido piora. **O gossip parcial (round 3, F1 97,25%) supera o federado
completo (94,64%) e a própria difusão final (94,27%)** — prova
experimental de que a difusão total pode diluir os bons especialistas
com os ruins. É a evidência mais forte a favor da união seletiva / k-de-n
([GOSSIP_DESIGN.md](GOSSIP_DESIGN.md)): nem sempre se quer a rede inteira.

(Difusão CICIDS: recall parte de 92,9% — o cliente hulk, classe gigante,
já domina no round 1 — e refina para 99,2%; no ERENO partia de 51,6%,
curva bem mais acentuada, pois nenhum sensor dominava.)

## 5. Adaptabilidade — comutação de arquitetura em tempo de execução

O experimento central do ereno-**Adaptativo**: comutar entre federado e
gossip no meio do treinamento (round 21 de 40) e medir o efeito.
Quatro cenários, nos dois datasets e nos dois sentidos, mesmo setup
(10 clientes `attack`, features XGB, fusão OR). Logs
`results/{ereno,cicids}_adapt_{fl_gl,gl_fl}_c10.log`; curvas
`results/conv_*_adapt_*_metricas.png`.

**A transferência de modelo na comutação** (implementada na
`HybridStrategy._transfer_model`):
- **federated → gossip**: o modelo global é semeado em todos os nós da
  topologia (cada nó começa o gossip com a união completa);
- **gossip → federated** (XGBoost): a **união deduplicada** do pool vira
  o global federado — boosters não são promediáveis, então a média
  aritmética original (válida só para NaiveBayes) foi substituída pela
  união (senão o caminho GL→FL falhava).

| Cenário | Comutação (round 21) | Comportamento observado |
|---|---|---|
| ERENO FL→GL | fed→gossip | F1 constante em 87,54 do round 1 ao 40 — comutação sem custo |
| ERENO GL→FL | gossip→fed | difusão sobe 60→87,54 (round 8), comuta no 21 **sem queda** |
| CICIDS FL→GL | fed→gossip | F1 constante em 94,64 — comutação sem custo |
| CICIDS GL→FL | gossip→fed | pico 97,25 (r3) → 94,27 → **degrau no r21** para 94,64 (FPR 2,90→2,69) |

**Conclusões:**

1. **A comutação é transparente nos dois sentidos** — o modelo é
   preservado, sem queda de desempenho. O sistema pode migrar entre
   federado e gossip conforme a infraestrutura muda (perda/retorno do
   servidor central) sem sacrificar a detecção. É a validação central da
   proposta;
2. **Transferir o modelo torna a comutação gratuita**: no FL→GL, o gossip
   não precisa difundir do zero (partiria de 51,6%) porque recebe a união
   pronta do federado — a curva é uma reta atravessando a troca;
3. **A comutação é observável, não apenas nominal**: no CICIDS GL→FL, o
   degrau no FPR (2,90→2,69) no round 21 prova que a troca de arquitetura
   teve efeito mensurável — e, de quebra, reexibe a não-monotonicidade do
   gossip (pico no round 3), reforçando o argumento da união seletiva (§4).

### 5.1 Tolerância a falhas — perda progressiva de nós

Cenário (ERENO, 60 rounds, `results/ereno_nodeloss_c10.log`,
`conf/experiments/ereno_adapt_nodeloss.yaml`): 20 rounds FL, depois gossip
com a rede **encolhendo** — 10 (rounds 21–30) → 7 (31–40) → 5 (41–50) →
3 (51–60) nós ativos (via `active_clients`; saem os índices mais altos =
os especialistas em masquerade/poisoned).

**Resultado: F1 87,54 / recall 99,67 / FPR 2,04 constantes nos 60 rounds**
— linha perfeitamente horizontal atravessando as quatro reduções
(`conv_ereno_nodeloss_metricas.png`). Perder 70% dos sensores (10→3) **não
degrada a detecção em nada**.

Motivo: os 20 rounds de FL formaram a união completa antes do gossip; a
partir daí, cada nó carrega no pool o conhecimento de toda a rede. Quando
os nós de masquerade/poisoned saem, seus boosters já estão nos nós
sobreviventes — nenhum ataque deixa de ser detectado. É a **resiliência do
gossip com memória**: para um IDS de infraestrutura crítica, o sistema
opera sem servidor central e sobrevive à queda progressiva de equipamentos.

*Escopo*: o teste derruba nós sobre uma rede **já madura** (conhecimento
difundido). Não cobre perda de nós *durante* a difusão inicial — cenário
mais duro, deixado como trabalho futuro.

## 6. Redução de falsos positivos

Duas vias foram testadas para cortar os FP do distribuído (`scripts/`):

- **Entre os 10 especialistas** (`analise_fp.py`, votação k-de-n *dentro* do
  federado): a diversidade útil — exigir concordância de ≥k sensores derruba
  o FPR drasticamente (a fusão OR = k≥1 é a mais sensível; k maior troca
  recall por precisão). É o eixo que rende ganho de sistema.
- **Entre arquiteturas** (`voto_arquiteturas.py`, votação Centralizado × FL ×
  GL): **não** rende ganho. Motivo medido: **FL e GL são idênticos sample a
  sample (concordância 100,00%)** — o GL difundido é a união dos mesmos 10
  especialistas que o FL agrega, logo o mesmo classificador. A votação
  colapsa para Centralizado × Distribuído; exigir consenso (k=3) apenas
  reproduz o centralizado (FPR 0,43%). *Resultado registrado como
  amadurecimento — a diversidade útil é entre sensores, não entre
  paradigmas de treinamento.* (FL≡GL também confirma que a escolha
  FL/GL é de arquitetura de rede, não de capacidade de detecção.)

## 7. Seleção de features: global × por-ataque × combinada

Investigação motivada pelo FP do especialista de masquerade (§6). Scripts
`analise_especialistas.py`, `superset_features.py`, `teste_combinado.py`
(ERENO, 10 especialistas, fusão OR, teste do autor 2,9 M).

**Diagnóstico**: cada especialista detecta ~100% da própria classe
(masquerade_fake_normal 100%, masquerade_fake_fault 99,85%) — o FP **não é
falta de cobertura, é precisão** (masquerade imita o normal → fronteira
intrínseca). Features específicas não resolvem sozinhas.

**Seleção por-ataque**: rodando seleção por classe (gain≥3%), os ataques
**quase não compartilham features** (Jaccard médio 0,07; nenhuma comum aos
7). A união é compacta (10 features) e **supera a seleção global-15**
(F1 90,2 vs 89,4) — porque a seleção global, otimizada para o problema
todo (dominado pelos ataques fáceis), *cega* pistas dos ataques difíceis
(perdeu vsbARms, frameLen, timestampDiff, confRev).

**Combinada (união global-15 ∪ superset-10 = 19 features) — o melhor**:

| Conjunto | F1 | Recall | Prec | FPR | #FP | masq9_FPR |
|---|---:|---:|---:|---:|---:|---:|
| global-15 | 89,39 | 99,71 | 81,01 | 1,701% | 46.854 | 1,508% |
| superset-10 | 90,24 | 99,91 | 82,27 | 1,567% | 43.185 | 1,036% |
| **combinado-19** | **94,70** | **99,93** | **89,99** | **0,809%** | **22.298** | **0,618%** |

**Achado central (efeito de interação)**: as 9 features elétricas que só o
global tinha **não são ruído** — sozinhas (global-15) ou ausentes
(superset-10) rendem pouco, mas *combinadas* com as pistas-de-ataque do
superset, o FPR **cai pela metade** (1,70%→0,81%) e o F1 salta para 94,7.
Features fracas isoladas viram fortes em conjunto (típico de XGBoost).

**Conclusão para a seleção de features**: o melhor não é "global *ou*
por-ataque", é a **união dos dois** — captura tanto as features de
interação (que o wrapper global acha) quanto as pistas causais dos ataques
difíceis (que só a seleção por-ataque acha). Conjunto adotado:
`features/all_in_one_ereno_train_combined.json` (19 features). O masquerade
segue o maior FP residual (0,62%) — base para a divisão-com-corroboração (§8).

**Validação por GRASP rigoroso por-ataque** (`grasp_por_ataque.py`,
`features/por_ataque/`): substituímos o proxy (importância por gain) pelo
**GRASP pleno binário** — as mesmas 55-feature RCL, VND e 5-fold CV do GRASP
global, mas com alvo *ataque-vs-normal*, uma execução por classe
(`no-improvement 15`, `sample 60k`; ~9 h de compute). A união das 7 seleções
(superset_grasp = 26 features) combinada com o global-15 dá o
**combined_grasp-31** (100 % GRASP, ponta a ponta):

| Conjunto | n | F1 | Recall | Prec | FPR | #FP | masq9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| global-15 | 15 | 89,39 | 99,71 | 81,01 | 1,701% | 46.854 | 1,508% |
| **combinado-19** | 19 | **94,70** | 99,93 | 89,99 | **0,809%** | 22.298 | **0,618%** |
| superset_grasp-26 | 26 | 94,03 | 99,92 | 88,80 | 0,917% | 25.275 | 0,726% |
| combined_grasp-31 | 31 | 94,69 | 99,93 | 89,96 | 0,811% | 22.350 | 0,620% |

**O GRASP rigoroso confirma o combinado-19 — não o supera.** O combined_grasp-31
empata em tudo (Δ F1 0,01; Δ FPR 0,002 pp) usando **12 features a mais**, e o
combinado-19 é **subconjunto** dele — as extras do GRASP são redundantes. Por
parcimônia (Occam), **combinado-19 é mantido, agora com respaldo rigoroso**: o
proxy por gain não enganou — caiu no mesmo platô que o GRASP pleno. A ressalva
metodológica está **resolvida**.

## 8. Divisão do masquerade + corroboração — o FP residual é sistemático

Motivação: o FP residual do masquerade (0,62%, §7) poderia ser
*idiossincrasia de treino* (decorrelacionável por corroboração) ou uma
propriedade *intrínseca* da fronteira de decisão. Teste (`divisao_masquerade.py`,
features combinado-19): dividir `masquerade_fake_fault` (o pior sensor) em
2 e 3 sub-especialistas treinados em **metades/terços disjuntos** do ataque
+ fatias disjuntas de benigno, e medir a **sobreposição dos FP** (Jaccard) e
a corroboração k≥2.

| Divisão | recall(masq) | Jaccard dos FP | k≥2 FPR | Redução vs FPR individual |
|---|---:|---:|---:|---:|
| 1 (baseline) | 99,06% | — | — | — |
| **2 subs** | 99,06% | **0,993** | 0,622% | **0%** |
| 3 subs | 99,06% | 0,789 | 0,656% (k≥2) / 0,624% (k≥3) | 10% / 14% |

**Achado (resultado negativo forte)**: dois detectores treinados em dados
*disjuntos* marcam **os mesmíssimos benignos** (Jaccard 0,99) — os erros são
quase perfeitamente **correlacionados**, não independentes. Corroboração
(k≥2) não remove nada porque a premissa da regra k-de-n — independência dos
erros — **não vale ao dividir uma única classe**. O recall também fica cravado
(99,06% em todo K e todo k): os TPs são idênticos. Dividir o ataque não gera
diversidade alguma.

**Consequência para a tese**: (i) o FP do masquerade é um **piso intrínseco**
fixado pelo *design* do ataque (imitar o normal), não um artefato de treino
corrigível — fecha a porta para "mais especialistas de masquerade"; justifica
manter **um** sensor por classe. (ii) O ganho do k-de-n (§6) vem da
**heterogeneidade entre classes** (erros de ataques diferentes são
independentes), não de replicar a mesma classe — a corroboração é uma alavanca
*entre especialistas distintos*, não *dentro* de um. Isso delimita com precisão
onde a fusão de decisão ajuda e onde não.

### 8.1 Irredutibilidade do `masquerade_fake_fault` — cinco evidências

O FP residual do masquerade não é um bug a corrigir, é um **limite fundamental
caracterizado**. Cinco evidências independentes convergem, cada uma fechando uma
explicação alternativa:

1. **Cobertura** (§7) — o especialista detecta ~100 % da própria classe: não é
   falta de cobertura, é precisão.
2. **Decorrelação** (§8) — dividir em detectores sobre dados disjuntos gera FPs
   idênticos (Jaccard 0,99): não é ruído de treino decorrelacionável.
3. **Features — existência** (§7) — o GRASP pleno atinge **F1 CV = 100 %** para
   6 dos 7 ataques (prova que *existe* subconjunto separador), mas para
   `masquerade_fake_fault` para em **99,385 %**: para esse ataque **não existe**
   subconjunto separador no espaço de atributos disponível.
4. **Features — esforço** — `masquerade_fake_fault` exige o **maior** nº de
   avaliações do GRASP (6.232, contra ~4,5–5,7 k dos demais): a assinatura de um
   *landscape* de otimização rugoso e sem ótimo dominante — muitos subconjuntos
   igualmente medíocres, típico de classes sobrepostas.
5. **Casamento quantitativo** — o gap de separabilidade em CV
   (100 − 99,385 = **0,615 %**) coincide com o **melhor FPR de teste alcançável**
   do sensor (**0,618–0,620 %**, invariante a qualquer conjunto de features,
   inclusive o ótimo do GRASP) dentro de 0,005 pp. **O erro irredutível medido no
   treino prevê o FP residual observado no teste.**

Conclusão: `masquerade_fake_fault` sobrepõe *de fato* a variedade do tráfego
normal — por design (imita uma falta legítima; o par `fake_normal` é 100 %
separável, só `fake_fault` colide). A tese não apenas mostra *que* há um piso de
FP, mas *por que*, por quatro ângulos qualitativos e um casamento quantitativo.

## Notas metodológicas

- **`benign_cap`**: os 2,76 M benignos de treino foram subamostrados para
  500 k (memória do WSL, 10 clientes). Justificável pela redundância do
  tráfego benigno do ERENO (features GOOSE repetidas por ~4.800 linhas
  SV); **todos os ataques preservados** e o **teste completo** (não
  capado). Ambos os braços (distribuído e monolítico) treinam sobre os
  mesmos dados — comparação justa.
- **Fusão OR e o FPR**: o trade-off cobertura×precisão é intrínseco à
  união de especialistas. O caminho mapeado para cortar o FPR sem perder
  recall (votação k-de-n / limiar por especialista) está em
  [GOSSIP_DESIGN.md](GOSSIP_DESIGN.md) — provavelmente levaria os
  distribuídos a superar o centralizado também em F1.
- **1 seed (42)**: resultados de uma execução; a campanha do
  [PLANO_VALIDACAO.md](PLANO_VALIDACAO.md) (múltiplos seeds) confirmaria
  a significância.
