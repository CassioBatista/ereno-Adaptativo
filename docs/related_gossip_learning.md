# Related work — aprendizado descentralizado / gossip (comparações)

Comparações do ReSIDS com teses/artigos de **aprendizado descentralizado e
gossip** (bucket distinto do related-work de IDS). Ver critério em
[[related-work-inclusion-criteria]]. BibTeX em [`refs_v2.bib`](refs_v2.bib).

## Legheraba (Sorbonne/LIP6) — `legheraba_p2p_decentralized`

**"Protocoles pair à pair pour un apprentissage décentralisé efficace et
résilient"** (P2P protocols for efficient and resilient decentralized learning).
Duas contribuições:

- **Elevator/Lift** — overlay P2P que se auto-organiza em topologia de **dois
  níveis** por *eleição emergente de hubs* (preferential attachment: cada nó
  conecta aos *h* mais conectados na vizinhança de 2 saltos + *c−h* aleatórios →
  hubs emergem em poucos ciclos). **Lift** = extensão **bizantina** que impede
  manipulação da eleição via PRNG compartilhado semeado pelos IDs dos hubs atuais.
- **HEAL/FLAIR** — FL descentralizado sobre o Elevator, com **agregação
  hierárquica em duas fases** por ciclo (intra-hub + inter-hub) → convergência
  mais rápida que gossip/epidemic plano. **FLAIR** adapta a agregação por hubs a
  redes sem fio via LEACH (avaliado no ns-3). Avaliado em imagem/texto sob
  operação sem falha, falhas estáticas e **churn**.

### Mesmo gênero, espécies diferentes

| Dimensão | Legheraba (HEAL) | ReSIDS |
|---|---|---|
| Domínio | FL descentralizado **genérico** (imagem/texto) | IDS **IEC-61850 GOOSE/SV** |
| Modelo | Redes neurais (**pesos promediáveis**) | **XGBoost** (boosters **não-promediáveis**) |
| Agregação | Merge/média hierárquico (intra/inter-hub) | **União/OR + k-de-n** (nível de decisão) |
| Overlay | **Emergente** (hubs por preferential attachment) | **Fixo** (ring/chain/star/graph) + head round-robin |
| Objetivo do head/hub | Acelerar convergência (hierarquia) | Difundir a união (resiliência) |
| Bizantino | **Lift** (PRNG na eleição) | v3 planejado (TrustedUnion+quórum, AUPE) |
| Resiliência avaliada | Falha estática + **churn** (img/texto) | **Node-loss** (FL destrutivo, GL transparente) IEC-61850 |
| Sem fio/físico | **FLAIR/LEACH** (ns-3) | Subestação (não modelado ainda) |
| Chaveamento FL↔GL | Não (sempre descentralizado) | **Sim** (runtime, distribuído) — distintivo |

### O que distingue o ReSIDS (a defender)

1. **Não-promediabilidade do XGBoost** é a razão de existir do união/OR + k-de-n;
   o HEAL assume merge de pesos (NN) → **não transfere** direto.
2. **Chaveamento FL↔GL em runtime** — Legheraba é sempre-descentralizado; a
   comutabilidade é distintiva do ReSIDS.
3. **Especificidade de domínio** — especialistas por-ataque, fusão em nível de
   decisão, o problema do masquerade FP: ausentes numa tese de FL genérico.

### Convergência do gossip: união/OR ≠ média de pesos (frase de blindagem)

As tabelas da tese (acurácia; ciclos-para-acurácia, 100 nós; msgs/ciclo) mostram
que o **"Gossip Learning"** plano **não converge** a 100 nós (N/A para 0.85/0.90/
0.95 em 1000 ciclos), enquanto o Epidemic converge ao custo de **10× a
comunicação** (1000 vs 100 msg/ciclo) e o HEAL recupera as duas coisas via
hierarquia. **Isso não atinge o ReSIDS**, e vale uma frase explícita no texto para
o revisor não transferir o problema:

> O "Gossip Learning" cuja convergência é lenta/ausente nesses estudos é a média
> estocástica de pesos em *random walk* (Ormándi/Hegedűs): cada nó promedia com um
> par aleatório por ciclo, e a mistura é lenta em redes grandes. O gossip do ReSIDS
> **não** é média de pesos — é **difusão união/OR de boosters inteiros**, um *fold*
> de semilattice **idempotente**. Por isso ele **converge em ~N rounds** (a rampa de
> difusão satura por volta do round 8–9 para N=10), satisfaz **GL = FL exato** sob
> OR (não há mistura a convergir), e é transparente a node-loss. A dificuldade de
> convergência do gossip por média de pesos é uma propriedade daquele operador, não
> do nosso.

Ressalva honesta (eixo de comunicação): o ReSIDS difunde **boosters inteiros**
(payload maior que gradientes); `results/escalabilidade.csv` traz `GL_comm_bytes`
crescendo com N. Qualquer afirmação de *eficiência de comunicação* exige reportar
esse custo — e é aí (não na convergência, que já temos) que a **hierarquia do HEAL**
entra como candidato v3 para escalar/reduzir comms (dialoga com a erosão em N=100).

### O que aproveitar (candidatos a v3, não v1/v2)

- **Elevator (hubs emergentes)** poderia substituir o head round-robin em
  topologia fixa → GL mais escalável (dialoga com a erosão em N=100 do teste de
  escalabilidade, [`plot_escalabilidade.py`](../scripts/plot_escalabilidade.py)).
- **HEAL (agregação hierárquica)** poderia acelerar a difusão gossip (hoje plana,
  sobe em rampa até saturar).
- **Lift (defesa bizantina na eleição)** → v3, ao lado de AUPE
  (`mukam2026byzantine`, ver [`refs_v2.bib`](refs_v2.bib)).
- **FLAIR/LEACH** → deployment sem fio em subestação (future work).

### Enquadramento

Entra em related work de **aprendizado descentralizado/gossip** — **não** no
bucket de IDS (é FL genérico, sem IDS/IEC-61850/XGBoost). Mecanismos (hubs
emergentes, agregação hierárquica, eleição bizantina) são **candidatos a técnica
para v3**.

> ⚠️ **Metadado a verificar:** o PDF do manuscrito traz a data de defesa como
> *placeholder* (`01/01/1970`) e não expõe DOI/HAL na capa — o **ano** da BibTeX
> está pendente de verificação.
