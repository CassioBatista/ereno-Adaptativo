# Arquitetura ERENO-Adaptive v2 — IDS federado autônomo e auto-autorrecuperativo

> Documento de **projeto** da v2 (branch `ereno-adaptive-v2`). Consolida as decisões
> de arquitetura para automatizar o chaveamento FL↔GL, distribuir o controle e
> adicionar resiliência a nós Byzantine. **Não** altera a v1 (tag `v1.0`).

## 1. Motivação — o furo de consistência da v1

Na v1 o chaveamento FL↔GL é **declarativo** (schedule YAML lido pelo servidor):
o controle mora num **ponto único** (o `FixedArchManager`). Mas se a solução é
resiliente, **o controle não pode morar no servidor** — perdê-lo levaria junto a
capacidade de decidir trocar. **O plano de controle precisa ser tão resiliente
quanto o plano de dados.**

## 2. Separação de planos (a decisão central)

| Plano | O que é | v2 |
|---|---|---|
| **Dados (modo)** | onde a detecção roda | **FL é o padrão** (eficiente); **GL é o fallback** (sob falha/Byzantine) |
| **Controle (política)** | quem decide chavear | **replicado em cada nó** (descentralizado) |

**FL permanece a base eficiente.** O servidor **agrega** (plano de dados) mas **não
controla**: o controlador vive em cada cliente. Perder o servidor = perder o
**agregador**, não o controle. Habilitadores (já existentes ou baratos):

1. **Semeadura** (já existe): o FL transmite a união a cada rodada → todo cliente
   **já segura a união** → opera GL na hora, sem o servidor.
2. **Política replicada**: as regras de chaveamento moram em cada cliente (pequenas).
3. **Detecção local**: cada cliente monitora servidor (timeout) + sinais Byzantine
   (auditoria local, custo zero de comunicação).

Mudança de código: `FixedArchManager` (YAML central) → **`DistributedArchManager`**
(controlador por-nó + política replicada + gossip das decisões).

## 3. Auditoria local — `LocalAudit` (custo zero de comunicação)

Cada nó audita a união recebida contra **os próprios dados rotulados** `D_i`
(reusa o broadcast de semeadura + compute local → **0 bytes extra**). Três camadas:

| Camada | Pega | Determinístico? | k |
|---|---|---|---|
| **Membership** (meu booster está no pool?) | censura direta | **sim** | agnóstico |
| **Corroboração** (≥k boosters disparam no meu ataque?) | quebra de redundância | quase (conta) | **k-aware** |
| **Injeção** (booster outlier no pool) | voto malicioso / FP | não (histerese) | agnóstico |

```
função LocalAudit(U):                       # nos dados locais D_i
  se hash(b_i) ∉ hashes(U):  retorna SUSPECT(censura)          # (1) membership
  para cada x ∈ D_i com rótulo=ataque:                         # (2) corroboração ≥k
      c ← #{ b ∈ U : b dispara em x }
      se c < k E b_i dispara em x:  retorna SUSPECT(redundância)
  para cada b ∈ U:                                             # (3) injeção
      rep[b] ← δ·rep[b] + concordância(b, D_i)
      se rep[b] < τ_rep por ≥ W rodadas:  retorna SUSPECT(injeção)
  retorna OK
```

**Discriminador imune a drift:** a referência é o **próprio especialista** do nó.
Se a união vai pior que `b_i` **nos dados de `b_i`** → servidor corrompeu; se os dois
vão igual → é **dado** (drift/ruído), não o servidor. As camadas (1)–(2) são
**determinísticas** (violação é prova, não estatística) → **zero falso alarme**.

## 4. `TrustedUnion` — a união com filtro de confiança

O detector Byzantine **não** é um módulo à parte: é o `xgb_union` ficando "de
confiança" (rejeita booster outlier/inconsistente). **Mesmo operador, executor
diferente** — espelho da simetria GL=FL:

| Executor | Papel |
|---|---|
| **Servidor (FL)** | detecta **clientes** Byzantine (visão global; par 2/ataque se auto-policia) |
| **Cliente (FL)** | audita o **servidor** (visão local; referência = próprio especialista) |
| **Cada nó (GL)** | filtra os **pares** que recebe (descentralizado) |

Nota Byzantine para modo k≥2: **presença (≥1) não basta** — auditar
**corroboração (≥k disparam no meu ataque)**. E o **par 2/ataque** que o k≥2 já exige
vira detector: os 2 especialistas de um ataque devem concordar; um comprometido
quebra o par. Limite: pega **1** Byzantine/ataque; **2 conluiados** precisam do
cross-check global (a mesma lei da maioria honesta).

## 5. Sinal de confiança unificado (3 drivers)

A reputação de cada nó — que dirige exclusão **e** ranking — tem três entradas:

```
trust(nó j) ← f( disponibilidade,          # crash / timeout
                 concordância-de-modelo,    # auditoria Byzantine (LocalAudit / TrustedUnion)
                 veredito-do-IDS-sobre-j )  # o IDS detecta j como FONTE de ataque (atribuído + corroborado)
```

O **veredito do IDS** fecha o loop: a detecção deixa de ser só saída (alarme) e
**realimenta** a arquitetura → **auto-policiamento**. Mapeamento:

| O IDS/auditoria flagra… | Ação |
|---|---|
| um **cliente** | **exclui o nó** (como Byzantine-client / churn) |
| o **servidor** | **switch FL→GL** (como Byzantine-server) |

⚠️ Cuidados do driver-IDS: **atribuição** (fonte ≠ vítima), **framing/DoS** (forjar
tráfego "de" um nó honesto) → exclusão exige **corroboração por quórum** (`f < q`) +
histerese, e um **mapa** tráfego-fonte → nó-federação.

## 6. Quórum e tolerância

- **Detecção (dados):** k-de-n tolera **`b < k`** boosters maliciosos (corroboração).
- **Controle (decisão):** switch/retorno global exige **quórum `q(N)=⌊N/2⌋+1`** →
  tolera **`f < q`** nós mentindo. Nem força nem bloqueia (maioria honesta).
- **Sybil:** o `xgb_union` deduplica por conteúdo → cópias idênticas colapsam em 1
  voto; precisa de `k` boosters **distintos** para `k` votos.

## 7. Transições — os dois algoritmos

Assimetria de projeto: **"fail fast, recover carefully"**. Sair é reação a um perigo
(rápida, pode ser local); voltar é **conceder confiança** (lenta, coletiva, provada).

### 7.1 FL → GL (saída, *fail fast*)

```
FL_para_GL (nó i, a cada rodada FL r):
  U ← recebe_união_do_servidor(timeout = T_hb)
  se timeout:  anomaly ← CRASH                       # observação compartilhada
  senão:       U_i ← U ;  anomaly ← LocalAudit(U_i)  # servidor Byzantine?
  # AUTO-PROTEÇÃO (local, SEM quórum):
  se anomaly ≠ OK:
      mode ← GL                                       # já opera: segura U_i
      gossip(SUSPEITA{i, anomaly, r})
  # DECISÃO GLOBAL (quórum):
  C ← #{ nós distintos com suspeita na janela }
  se C ≥ q(N_ativo):  commit_global(GL)
```

Crash → todos dão timeout → convergem trivial (rápido). Byzantine equivocante →
quem detecta se auto-protege; vira global só com `≥ q` honestos.

### 7.2 GL → FL (recuperação, *recover carefully*)

```
GL_para_FL (nó i, ao ANÚNCIO de um servidor S):
  # PORTÕES DE HISTERESE (anti-flapping):
  se agora − t_GL < T_dwell:  retorna
  se S expulso E agora − t_exp(S) < T_cool:  retorna
  # TESTE DE ADMISSÃO (determinístico):
  envia b_i a S ;  U' ← pede_agregação(S)
  se U' == TrustedUnion(boosters conhecidos):  voto_i ← CONFIA ; rep[S]↑
  senão:                                       voto_i ← REJEITA ; rep[S]↓ ; retorna
  # CORROBORAÇÃO (quórum, NUNCA unilateral):
  gossip(VOTO_ADMISSÃO{i, S, CONFIA})
  T ← #{ nós distintos votando CONFIA em S }
  se T ≥ q(N_ativo) E rep[S] ≥ τ_rep:
      U_i ← união_atual_do_GL()                # servidor ADOTA a união retida
      envia U_i a S ;  mode ← FL               # eficiência restaurada
```

### 7.3 Nota sobre broadcast

O switch **não** é um broadcast unilateral (isso seria vetor de DoS): **crash** usa
**detecção local** (todos observam o mesmo silêncio → convergem sem avisar);
**Byzantine/assimétrico** usa **gossip de suspeita + quórum**. O modo é decisão
**local** (consistência eventual — todo nó retém a união, split transitório é
tolerado).

## 8. Fronteira de resposta (Nível 1 vs Nível 2)

> **Invariante:** nenhuma ação **automática** cruza da *fábrica de detecção* para o
> *sistema elétrico protegido*. Resposta no grid é **recomendação** com humano.

| Onde a ação incide | Nível | Automático? | Consequência se errar |
|---|---|---|---|
| **Fábrica de detecção** (federação FL/GL, união, nós como detectores) | 1 | **sim** | federação degradada — **reversível** |
| **Sistema protegido** (IEDs, barramento GOOSE/SV, proteção do grid) | 2 | **não** (só recomenda) | risco **físico** — irreversível |

- **Nível 1 (automático):** FL↔GL, exclusão de nó da federação, rejeição de booster
  — sempre com quórum/reputação. Pior caso: nó honesto sai da federação; detecção
  segue pelos demais; readmissão por reputação.
- **Nível 2 (gated):** isolar IED do barramento, desconectar dispositivo — **nunca**
  automático; emite recomendação + evidência a um **operador** (human-in-the-loop).
  Racional: falso positivo num IEC-61850 crítico pode desestabilizar a proteção.

## 7.4 `DistributedArchManager` — troca distribuída e automática (implementado)

O controle **não mora no servidor** (§2): a v2 substitui o schedule estático por
[`DistributedArchManager`](../fd/arch_manager.py) — controle replicado por-nó,
disparado por **detecção local de falha**, sem schedule e sem política central.

- **Modelagem:** N controladores por-nó (`_NodeController`) dentro do manager;
  cada um mantém sua visão local de pares vivos e **se auto-protege** (FL→GL) no
  instante em que um par que julgava vivo fica silencioso — *fail-fast*, sem quórum.
- **Commit global por quórum:** a troca de modo efetiva vira commit quando
  `≥ q` nós votam; `q = ⌊N_ativo/2⌋+1` (§6). Para crash/node-loss todos observam
  o mesmo silêncio → votam unânimes → commit imediato. A estrutura por-nó/quórum
  é a base do caso Byzantine (v3).
- **Recover carefully (§7.2):** GL→FL só com **participação plena** estável por
  `dwell_rounds` **e** fora do `cooldown_rounds` (anti-flapping).
- **Detecção vs ground truth:** as faltas do ambiente (`faults` na conf) são
  *ground truth* usado só para a participação (`get_active_clients`); o controle
  **detecta pela observação** de quem reportou (`observe(round, reported_nodes)`),
  nunca lendo o schedule de faltas. Latência de 1 round (detecta no fim do round
  *r*, comuta no *r+1*).
- **Fiação:** [`HybridStrategy`](../fd/strategy/hybrid_strategy.py) chama
  `observe()` em `aggregate_fit` (quem reportou = quem está vivo) e, na troca,
  lê `consume_switch_reason()` para anotar o **motivo real** (`node_failure` /
  `recovery`) no evento do monitor — não mais inferido.

Config em [`conf/base.yaml`](../conf/base.yaml) (`manager: distributed`,
`initial_mode`, `quorum`, `dwell_rounds`, `cooldown_rounds`, `faults`).
Self-test da máquina de estados: [`scripts/distarch_selftest.py`](../scripts/distarch_selftest.py).

## 7.5 Semeadura do GL pela união retida (fecha o gap de timing da troca)

A troca FL→GL é reativa: a **perda que dispara** a troca acontece *antes* da
detecção (latência irredutível de 1 round). Se o GL fosse semeado do **agregado
do round** — que já perdeu o booster do nó recém-caído —, aquele especialista
sumiria do pool e, sob **k≥2**, seu ataque nunca voltaria a ter 2 votos: a troca
protegeria perdas *futuras*, mas não a que a disparou.

Correção ([`hybrid_strategy.py`](../fd/strategy/hybrid_strategy.py), commit
`3568eae`): o `HybridStrategy` mantém uma **união retida** (`dedup_union`
acumulado de todo booster já agregado) e semeia o GL a partir dela. Como os
especialistas são **determinísticos** (seed/dados fixos), o content-dedup colapsa
as cópias → a união converge para **um booster por nó** e **não cresce**. Assim,
um nó que cai logo antes da troca **ainda tem seu especialista no pool difundido**
— é o modelo fiel de **P4/§2** ("sobreviventes carregam a rede"), agora também na
direção FL→GL (não só no handover GL→FL).

**Evidência (14 nós, k≥2; derruba 1 sensor dos 3 maiores ataques @10/16/22):**

| fase | Estático (piso) | Adaptativo s/ fix | Adaptativo c/ fix |
|---|---|---|---|
| 1–9 (íntegro) | 96.02 | 96.02 | 96.02 |
| 10 (−nó1, FL) | 94.69 | 94.69 | 94.69 |
| 11–36 (GL) | — | 94.69 | **96.02** |
| baseline FL 16→36 | **86.33** | — | — |
| recall final | 82.21 | 97.32 | **99.96** |

O round 10 (94.69) é a **latência irredutível**: a falha ocorre antes de ser
detectada. De 11 em diante o fix recupera **totalmente** (recall 99.96), inclusive
a perda que disparou a troca. Ressalva de escopo: reter o booster de um nó morto
indefinidamente é questão de **staleness/decay** (v3, BitMatcher/BMDecay [`mukam2026byzantine`]);
no horizonte do v2 o especialista permanece válido. Confs
[`ereno_dist_adapt_n14_k2.yaml`](../conf/experiments/ereno_dist_adapt_n14_k2.yaml) /
[`_static_n14_k2`](../conf/experiments/ereno_dist_static_n14_k2.yaml); figura das
três curvas via [`scripts/plot_dist_adapt.py`](../scripts/plot_dist_adapt.py).

> **Nota:** a *análise* deste gap de timing e o fix da união retida são material
> de um **novo artigo** em desenvolvimento; os artefatos pré-fix (`results/*_prefix.*`)
> estão preservados propositalmente.

## 8.1 Monitor externo (observabilidade, read-only)

Um **elemento externo (monitor)** observa a federação **sem influenciá-la**
(estritamente Nível-1). Implementado em [`fd/monitor.py`](../fd/monitor.py),
fiado no [`HybridStrategy`](../fd/strategy/hybrid_strategy.py) (param `monitor`)
e construído da conf (`build_monitor`). Opt-in (desabilitado por padrão → `NullMonitor`,
custo zero). Decisões: **pull** (o monitor consulta; sem webhook, sem supor o
monitor alcançável) + **trilha de auditoria JSONL persistida** (seq monotônico,
contínuo entre runs — replay-friendly). Stdlib apenas (`http.server`).

**Eventos** (exatamente os dois pedidos):

| type | Quando | Campos-chave |
|---|---|---|
| `architecture_change` | troca FL↔GL ([`hybrid_strategy.py`](../fd/strategy/hybrid_strategy.py), `configure_fit`) | `from_mode`, `to_mode`, `reason`, `recent_failed_nodes` |
| `node_failure` | falha de cliente (`aggregate_fit`) | **`failed_nodes`** (índice do nó, via `cid_map`/`resolve_node_index`) |

O `architecture_change` **não** afirma causalidade com a falha — carrega
`failed_nodes` da janela como contexto; a fonte autoritativa de "qual nó falhou"
é o evento `node_failure` separado.

**Endpoints REST (GET, o monitor faz polling):**

```
/health              -> {"status":"ok"}
/status              -> modo corrente, round, active/failed nodes, last_seq
/events?since=<seq>  -> eventos novos (polling incremental)
/events?type=&limit= -> filtros opcionais
```

Config (ver [`conf/base.yaml`](../conf/base.yaml)): `monitor.{enabled,host,port,serve,trail}`.
Self-test ponta a ponta: [`scripts/monitor_selftest.py`](../scripts/monitor_selftest.py).

## 9. Simulação e métricas (verificação)

Duas camadas: **ambiente** (injeta eventos com ground-truth: crash, Byzantine,
churn, retorno) + **nós autônomos** (reagem via Alg. 7.1/7.2). Mede duas famílias:

- **Detecção (dados):** F1 / Recall / FPR por-round, agregado **e por-ataque** —
  verifica a invariância **sob switch autônomo** (não roteirizado).
- **Controle (adaptação):** latência de switch/recuperação, **correção** (matriz de
  confusão TP/FP/FN dos switches vs eventos verdadeiros), downtime, tolerância
  `b<k`/`f<q`, overhead de controle (≈0 normal, pico no switch).

Comparar o **Adaptativo** contra **Oráculo** (switch perfeito/instantâneo — teto) e
**Estático** (não troca — piso): tese = Adaptativo ≈ Oráculo ≫ Estático.

## 10. Propriedades a enunciar (invariantes)

- **P1 (saída segura):** nó sob anomalia se auto-protege sem quórum → nunca preso a
  servidor ruim.
- **P2 (sem sabotagem):** decisão global exige `≥ q` → `f < q` Byzantine não força
  nem bloqueia.
- **P3 (retorno provado):** GL→FL só com teste determinístico + probação → não
  readmite servidor Byzantine.
- **P4 (handover limpo):** reconciliação adota a união **retida** pelo GL.
- **P5 (fronteira física):** resposta automática só na fábrica de detecção; grid é
  advisory.

## 11. Escopo e trabalhos futuros

- **v2 (este desenho):** IDS federado autônomo e auto-curativo — Nível 1.
- **Eixo separado:** backend **LightGBM** (memória/velocidade + generalidade do
  framework; validar GL=FL exato com serialização determinística).
- **Future work com salvaguarda:** Nível 2 (resposta ativa no grid) com
  human-in-the-loop.

## 11.1 Fundamentação externa — peer sampling tolerante a bizantinos (AUPE)

A tese de **Mukam (2026)** [`mukam2026byzantine`] sobre *Byzantine-resilient peer
sampling* dá a maquinaria concreta para os componentes que já projetamos aqui. O
domínio dela é amostragem de pares por **IDs**; o nosso é difusão de **boosters** —
a transferência é por **analogia** (mesmo mecanismo, design/avaliação novos no
contexto GOOSE/SV; **não** herdamos as garantias numéricas dela, ex. "26% de
tolerância", que são do peer sampling).

| Mecanismo do AUPE | Onde entra na v2 |
|---|---|
| **Set Cleaner** — *tracking* (frequência de cada fonte) + *debiasing* (reduz peso de fontes super-representadas; Alg. 1, `p_j = min/Φ_j`) | Endurece o pool difundido do GL: um nó comprometido floodando seu booster poluído é **penalizado por super-representação** antes do voto — vira driver da "concordância-de-modelo" (§5). |
| **Agregação colaborativa de reputação** — nós confiáveis fazem *merge* comutativo/associativo (average; *max* p/ sketches) dos componentes de tracking via gossip | O **sinal de confiança unificado** (§5) deixa de ser só local: os **conselheiros** gossip-agregam reputação por-booster → **k-de-n ponderado por confiabilidade** em vez de voto plano. Fecha o FP do masquerade de forma distribuída. |
| **Bias factor** — métrica que captura super-representação de IDs adversariais vs. corretos (supera erro agregado) | Métrica-alvo p/ *poisoning*: um booster com taxa de **disparo-em-benigno anômala** relativa ao pool é sinal melhor que "acurácia agregada" p/ flagar booster ruim (alimenta `LocalAudit`, §3). |
| **BitMatcher + BMDecay** — contagem adaptativa econômica; *decay* (halving) p/ streams infinitos; *merge* (casa fingerprints, toma max) | IDS roda **indefinidamente** → contadores de stream saturam. Relevante às famílias que **são** estatística de stream (protocol-counter F40–48, temporal). BMDecay mantém frescor em streams de 10M com ~12% da memória; conecta ao parágrafo de **staleness/retenção** (contribuições difundidas devem decair). |

**Dependências/ressalvas:** o *collaborative debiasing* do AUPE usa **TEE + remote
attestation** (handshake de nós confiáveis). Citamos como **opção** para o handshake
dos conselheiros — **não** como requisito de hardware. É escopo de **v2**, não
retrofit da v1: introduz um **modelo de ameaça novo** (bizantino), que a v1 não trata.

**Bullet pronto p/ o future-work do artigo (LaTeX):**

```latex
\item \textbf{Byzantine-resilient gossip fusion (v2).} The current gossip
mode assumes honest peers; a compromised node could poison the diffused
pool, and the union/OR fusion amplifies a single malicious booster into a
false-positive flood. Adapting Byzantine-resilient peer-sampling machinery
\cite{mukam2026byzantine}---a frequency-tracking \emph{Set Cleaner} that
down-weights over-represented sources, and collaborative, gossip-aggregated
reputation among trusted counselors (optionally attested)---would turn the
flat $k$-of-$n$ vote into a reliability-weighted one, extending ReSIDS's
resilience from churn to adversarial nodes. Memory-bounded sketches with
controlled decay (BitMatcher/BMDecay) further address the unbounded-stream
nature of the protocol-counter and temporal features on constrained IEDs.
```

BibTeX em [`docs/refs_v2.bib`](refs_v2.bib).

## 12. Enquadramento (uma frase)

> ERENO-Adaptive v2 é um **IDS federado autônomo e auto-curativo** (*self-healing /
> self-protecting*) para IEC-61850: **FL** é o modo eficiente padrão e **GL** o
> fallback resiliente; o **controle é distribuído** (cada nó decide por política
> local + quórum); a defesa Byzantine é o **`TrustedUnion`** (mesmo operador,
> executor FL/GL); e a **própria detecção realimenta a arquitetura** (ejeta o nó
> comprometido) — estritamente sobre a fábrica de detecção, deixando a resposta no
> grid como recomendação com humano.
