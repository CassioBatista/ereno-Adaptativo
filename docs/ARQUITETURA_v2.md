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

## 12. Enquadramento (uma frase)

> ERENO-Adaptive v2 é um **IDS federado autônomo e auto-curativo** (*self-healing /
> self-protecting*) para IEC-61850: **FL** é o modo eficiente padrão e **GL** o
> fallback resiliente; o **controle é distribuído** (cada nó decide por política
> local + quórum); a defesa Byzantine é o **`TrustedUnion`** (mesmo operador,
> executor FL/GL); e a **própria detecção realimenta a arquitetura** (ejeta o nó
> comprometido) — estritamente sobre a fábrica de detecção, deixando a resposta no
> grid como recomendação com humano.
