# Gossip Learning no ereno-Adaptativo — desenho e decisões

Consolida a maturação do desenho do gossip (2026-07), motivada pelos
experimentos com o particionador `attack` que revelaram o protocolo
originalmente portado como **sem memória** (curva de métricas periódica,
sem difusão de conhecimento).

## Referência e fidelidade

Referência: **GLow** (Belenguer, Pascual & Navaridas —
[arXiv:2501.10463](https://arxiv.org/abs/2501.10463), JPDC 2026,
[github.com/AitorB16/GLow](https://github.com/AitorB16/GLow)).

No GLow original:

- **o servidor não treina nem decide** — é o orquestrador da simulação;
  a agregação roda na `Strategy` como *stand-in* do head (só combina o
  que o head poderia ver: seu modelo + vizinhos ativos);
- **o cliente ADOTA o que recebe**: `fit()` faz `set_parameters(recebido)`
  e treina a partir dele — é este elo que faz o conhecimento difundir;
- o pool guarda o modelo corrente de cada nó; o head fica com o agregado.

### Decisões (2026-07-10)

1. **Manter a agregação na Strategy**, fiel ao GLow — nossa solução é de
   simulação. Fica registrado que, numa implantação real, a agregação
   deve ocorrer **no cliente head** (o servidor apenas roteia: recebe os
   modelos e os encaminha ao head dentro do round; o Flower não suporta
   cliente↔cliente direto — topologia física estrela, topologia lógica
   definida pelo roteamento).
2. **Corrigir o cliente**: em rounds gossip, o cliente XGBoost passa a
   **adotar o modelo recebido e continuar o boosting** a partir dele
   (`xgb.train(..., xgb_model=recebido)`) — o equivalente para árvores
   do `set_parameters`+train do GLow. Era este o elo quebrado no porte.
3. **Operadores de agregação XGBoost como eixo de pesquisa** — o GLow
   original só agrega pesos promediáveis (redes neurais); árvores exigem
   operadores próprios, enumerados abaixo.

## O espaço de soluções de agregação para XGBoost

Evidência experimental (ERENO, particionador attack, 7 especialistas,
pareado com features fixas — logs `results/ereno_{sum,or}_20_c7_s42.log`):

| Operador | Mecânica | Recall união dos 7 | FPR | Observações medidas |
|---|---|---|---|---|
| `xgb` (merge/sum) | concatena árvores; margens SOMAM | 12,9% | 0,0% | veto da maioria: 6 especialistas "normal" anulam 1 "ataque" — conservador patológico |
| `xgb_union` (OR) | conjunto de boosters; decisão por união | 99,5% | 13,6% | recupera a cobertura; FPR soma (union bound); sensível a especialistas ruidosos (trio bom: F1 81,6 c/ FPR 0) |
| monolítico (referência) | um modelo viu tudo | 90,8% | 2,7% | o alvo a alcançar |

Operadores candidatos a desenvolver/quantificar (com o cliente já
adotando o recebido):

| # | Operador | Modelo do nó | Crescimento | Hipótese |
|---|---|---|---|---|
| A | **merge + warm-start** (fiel ao GLow) | 1 booster que evolui | head multiplica (~3× por agregação) + 10 árvores/round por nó — exige teto de árvores | o boosting local corrige as margens do merge; deve reduzir o veto |
| B | **união com memória** (set-based; protótipo validado em sintético: difusão 3→7 boosters, recall 43→99,6%) | conjunto acumulado (dedup) | limitado pelo nº de nós | cobertura máxima; FPR controlável por k-de-n |
| C | **k-de-n sobre a união** | conjunto + regra "alarme se ≥k dispararem" | idem B | meio-termo sum↔or; k=2 deve cortar o FPR de 13,6% mantendo recall |
| D | **seleção por score** (ScoreAVG do GLow adaptado) | head adota o MELHOR booster da vizinhança (validação local) | constante | evita poluição por especialistas ruidosos |
| E | **destilação** | novo booster treinado com rótulos suaves dos vizinhos | constante | a forma teoricamente correta de "promediar" árvores; custo maior |

## Análise: k-de-n, redundância e o efeito da difusão

O operador C (k-de-n) só faz sentido sob duas condições, ambas ligadas a
decisões já tomadas no projeto:

**1. k-de-n e sensores redundantes são um pacote.** Os "n votantes"
precisam ser testemunhas reais do mesmo fenômeno:

- **Com redundância** (particionador `attack` com N > nº de ataques: 2
  sensores da mesma classe, treinados em metades disjuntas do ataque e
  fatias disjuntas de benignos), k=2 é corroboração entre testemunhas
  independentes — erros de FP tendem a ser independentes, então o alarme
  corroborado tem FPR ≈ produto dos individuais (2% e 2% → ~0,04%,
  contra ~4% da união), com recall preservado para a classe duplicada.
- **No 1:1 estrito** (1 especialista por ataque), k=2 depende de
  generalização cruzada entre especialistas (medida: trios cobrem ~73%,
  bem além das próprias classes — ataques "aparentados" se corroboram),
  mas **suprime as classes sem parentes**. Refinamento natural: **k por
  classe** (k=2 onde há 2 sensores, k=1 onde há 1), que degenera para a
  união seletiva.

**2. A difusão afeta os operadores de forma oposta** ("quando o gossip
difunde, fica tudo igual?"):

| | Operador A (merge) | Operador B (união) |
|---|---|---|
| Após a difusão, os nós... | convergem **misturando** — modelos progressivamente correlacionados (nunca idênticos: dados locais diferem) | convergem **colecionando** — todo nó carrega o MESMO conjunto, mas os membros permanecem os N boosters distintos |
| O que homogeneíza | os próprios votantes | quem carrega os votantes (a diversidade interna é preservada pela deduplicação) |
| k-de-n | perde sentido com o tempo (votos correlacionados não corroboram) | mantém/ganha sentido — após difusão, qualquer nó aplica localmente a votação da rede inteira |

Consequência para o desenho experimental: o k-de-n é um diferencial
**exclusivo do operador B** e casa com o particionador de sensores
redundantes — mais um eixo de comparação A×B, além de acurácia/FPR e
custo de crescimento do modelo.

## Protocolo corrigido (por round gossip)

```
Strategy (stand-in do head):
  configure_fit  → cada nó recebe SEU modelo do pool
  aggregate_fit  → pool[nó] = modelo devolvido (nó treinou A PARTIR do recebido)
                   pool[head] = operador(pool do head + vizinhos ativos)
Cliente:
  ADOTA o modelo recebido (xgb_model=...) e continua o boosting local
Comutação federated→gossip:
  o modelo global é semeado em TODOS os nós da topologia
```

Difusão esperada: o agregado do head vira o modelo dele; nos rounds
seguintes, como vizinho de outros heads, esse conhecimento se propaga —
em anel de N, cobertura completa em ~N−2 agregações (validado em
sintético para o operador B).

## Nota — nosso gossip ≠ "Gossip Learning" por média de pesos

Estudos de FL descentralizado (ex.: Legheraba, ver
[`related_gossip_learning.md`](related_gossip_learning.md)) mostram o **"Gossip
Learning"** plano **falhando em convergir** a 100 nós. Esse "Gossip Learning" é a
**média estocástica de pesos em random walk** (Ormándi/Hegedűs): mistura lenta em
redes grandes. **O nosso gossip não é isso** — é **difusão união/OR de boosters
inteiros**, um *fold* de semilattice **idempotente** → converge em **~N rounds**
(rampa satura ~round 8–9 para N=10), **GL = FL exato** sob OR, transparente a
node-loss. A não-convergência daquele operador é propriedade da média de pesos,
não do nosso. (Blindagem para revisor; detalhes em `related_gossip_learning.md`.)

## Pendências de quantificação

- Crescimento do modelo no operador A (teto de árvores? poda?);
- k ótimo do operador C (varredura k=1..n);
- custo de comunicação por operador (bytes/round — métrica 5.3 do plano);
- comparação A×B×C×D no cenário attack-7 (pareado, features fixas).
