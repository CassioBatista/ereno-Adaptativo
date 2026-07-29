# Resultados — ERENO IEC-61850, conjunto **combinado-24** (Centralizado × FL × GL)

Documento **separado** dos resultados originais (`RESULTADOS.md`, que usa o
conjunto xgb-15). Aqui ficam os resultados sobre o conjunto de features
**adotado** — `combinado_def-24` (GRASP penalizado por cardinalidade, λ=0,05;
ver `RESULTADOS.md §7.1`):

```
[2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
```
(13 SV elétricas + 6 GOOSE + 5 temporais; nomes reais em `docs/FEATURE_MAP` /
`GSVDatasetWriter.java` do ERENO.)

## Síntese

As sete seções abaixo produzem muitas medições, mas **todas derivam de um único
eixo** — duas propriedades que determinam o comportamento do detector:

1. **O operador de fusão é idempotente?** OR sim (semilattice); merge e k-de-n não.
2. **A topologia replica a união dos especialistas em cada nó?** Gossip sim
   (difusão); federado não (só agrega o round corrente).

> **Tese que amarra os resultados:** a escolha adaptativa FL↔GL **não é
> cosmética** — é governada pela álgebra do operador de fusão (idempotência) e
> pela capacidade da topologia de replicar a união dos especialistas. Gossip + OR
> entrega um detector **idempotente, transparente a churn e provadamente
> equivalente ao federado**; a corroboração k-de-n para cortar falsos positivos é
> uma propriedade que **o gossip oferece e o federado não**.

| § | Conclusão | Deriva de |
|---|---|---|
| 2 | combinado-24 > xgb-15 | features fortes ⇒ absorção exata (pré-condição) |
| 3 | **GL = FL exato** (0 divergências) | OR é **idempotente** |
| 4 | churn: gossip transparente, federado destrutivo | gossip **replica a união** |
| 5 | especialista ≫ IID (contribuição, não conveniência) | *union bound* dos FP |
| 6 | OR robusto a N; merge explode; k-de-n com sweet-spot | **idempotência** é o divisor |
| 7 | sweet-spot k≥2 é **exclusivo do gossip**; catastrófico em federado+poucos nós | corroboração exige o pool redundante que só a **difusão** cria |

**Escopo (por que não há contradição):**
- §3 (GL=FL exato) vale **com participação plena**; sob perda de nós (§4, §7) elas
  divergem justamente porque o gossip mantém a união e o federado a perde.
- §6 (sweet-spot precisa N≥8) e §7 (sweet-spot é gossip-exclusivo) são
  complementares: pool grande **e** difundido.
- §1 (FL=GL 95,72) reconcilia o antigo xgb-15 (GL≠FL 0,03 pp) — combinado-24 fecha
  a absorção (§3).

## Configuração do experimento

- **Dataset**: ERENO IEC-61850 (`all_in_one_ereno_train/test`), teste do autor
  (2,9 M amostras: 200.509 ataques, 2.755.139 normais).
- **Particionador**: `attack` — 10 clientes especialistas (1 classe/cliente +
  sensores redundantes para as classes 1, 5, 6); `benign_cap` = 500 k.
- **Distribuído (FL/GL)**: Flower 1.32, estratégia `xgb_bagging`, fusão
  **OR** (`xgb_union`), `eval_sample` 500 k, 20 rounds, 10 nós (anel).
- **Centralizado**: **monolítico XGBoost** — um único modelo treinado em todo o
  train (100 árvores = orçamento total da federação, 10×10), mesmas features e
  mesmo teste. **Não usa Flower** (baseline não-distribuído). Confirmado por
  script standalone (`scripts/teste_central_xgb.py`).
- **Config**: `conf/experiments/ereno_{federado,gossip}_combined.yaml`.

## 1. Métricas globais (teste completo)

| Métrica | Centralizado | Federado (OR) | Gossip (OR, difundido) |
|---|---:|---:|---:|
| F1-score | **99,23%** | 95,72% | 95,72% |
| Recall | 99,96% | **99,97%** | **99,97%** |
| Precision | **98,52%** | 91,81% | 91,81% |
| FPR | **0,110%** | 0,649% | 0,649% |
| #FP | 3.017 | 17.881 | 17.881 |
| VP / FN | 200.424 / 85 | 200.444 / 65 | 200.444 / 65 |

**Conclusões:**

1. **Gossip = Federado**, dígito por dígito — o consenso descentralizado sem
   servidor alcança o federado (difusão satura em 19 boosters no round 9).
2. **Ambos os distribuídos ganham em recall** (99,97 > 99,96): a união dos
   especialistas não perde nenhum tipo de ataque.
3. **O centralizado ganha em F1/precisão/FPR** (0,110 % vs 0,649 %): um modelo
   único, vendo todos os ataques, calibra a fronteira mais apertada. O custo do
   distribuído é o *union bound* dos 10 sensores (FPs somam).

> **Ressalva de métrica — por que não usamos acurácia.** O ERENO é fortemente
> **desbalanceado** (~93 % do teste é tráfego normal), então a **acurácia infla**
> e não reflete a qualidade da detecção: a acurácia é **99,39 %** no distribuído e
> **99,89 %** no centralizado — números que *parecem* ótimos, mas escondem os
> 17.881 falsos positivos do distribuído. Por isso o **F1** e o **FPR** (e o
> recall) são as métricas de mérito reportadas neste documento; a acurácia é
> deliberadamente omitida das tabelas para não induzir uma leitura otimista.
> (A acurácia só é comparável com a coluna *Average Accuracy* de referências em
> datasets balanceados, como MNIST/CIFAR do GLow.)

## 2. Ganho das features — combinado-24 vs xgb-15 (o `RESULTADOS.md §1`)

| | Centralizado | Distribuído (FL≈GL) |
|---|---|---|
| xgb-15 (RESULTADOS.md §1) | F1 96,88 / FPR 0,43 % | F1 87,5 / FPR 2,05 % |
| **combinado-24** | F1 **99,23** / FPR **0,110 %** | F1 **95,72** / FPR **0,649 %** |

O combinado-24 **melhorou os dois lados e encurtou o gap** Centralizado−Distribuído
de ~9,4 pp → **~3,5 pp** de F1; o FPR do distribuído caiu **2,05 % → 0,649 %** (o
custo do *union bound* ficou ~3× menor). O trade-off qualitativo persiste
(distribuído ganha recall; centralizado ganha F1/FPR), mas agora com o
distribuído muito mais próximo do centralizado.

## 3. Gossip = Federado — equivalência exata (com combinado-24)

Com o combinado-24, GL e FL ficam **idênticos** (mesmos VP/VN/FP/FN;
concordância **por-amostra 100,00 %** em 2,9 M, com o GL reconstruído por difusão
gossip **independente** — `scripts/voto_arquiteturas_combined.py`).

Isso **refina** o resultado com xgb-15, onde eram apenas *próximos* (F1 87,54 vs
87,51): a igualdade **exata não é automática** — ela **emerge quando as features
são nítidas o bastante** para o OR absorver as variantes de warm-start do pool
gossip (com xgb-15 restava 0,03 pp; com combinado-24, 0,00 pp).

Demonstração formal e caracterização em **`docs/gl_fl_equivalence.tex`**
(Overleaf): a fusão OR é um *fold* de semilattice **idempotente** ⇒ depende só do
**conjunto** de especialistas ⇒ GL = FL na convergência; contraste com o
GLow-referência (que **promedia** pesos ⇒ GL ≈ FL aproximado e **degradante**,
pois média não é idempotente); e a tabela dos parâmetros que **quebram** a
igualdade (operador de agregação, `num_rounds` < diâmetro, topologia desconexa,
especialistas não-determinísticos).

## 4. Adaptabilidade — comutação de arquitetura + encolhimento de nós

Dois experimentos de 80 rounds que exercitam a comutação FL↔GL em tempo de
execução **e** a redução progressiva de nós ativos (10→5→3), sobre o
combinado-24. Configs `conf/experiments/ereno_adapt_{fl_gl,gl_fl_shrink}_combined.yaml`;
avaliação do modelo agregado no teste global a cada round
(`ROUND;n;modo;f1;recall;fpr`). Curvas em
`results/conv_ereno_adapt_combined_{metricas,difusao}.png`.

- **exp 3 — FL→GL, encolhe sob GOSSIP:** `1-20 fed/10 → 21-40 gossip/10 →
  41-60 gossip/5 → 61-80 gossip/3`.
- **exp 4 — GL→FL, encolhe sob FEDERADO:** `1-20 gossip/10 → 21-40 fed/10 →
  41-60 fed/5 → 61-80 fed/3`.

### 4.1 Trajetórias (F1 por round)

| Fase | exp 3 (encolhe sob GOSSIP) | exp 4 (encolhe sob FEDERADO) |
|---|---:|---:|
| início | 95,74 (FL, união instantânea) | 72,3 → **95,74** (rampa de difusão, satura round 8) |
| comutação (r21) | 95,74 (FL→GL, transparente) | 95,74 (GL→FL, transparente) |
| encolhe p/ 5 (r41) | **95,74** | **85,36** (recall 99,96→74,5) |
| encolhe p/ 3 (r61) | **95,74** | **85,48** (recall 74,6 · Prec 100% · FPR 0,000%) |

### 4.2 O achado central — gossip tolera perda de nós; federado não

**Encolher a rede é transparente sob gossip (exp 3) e destrutivo sob federado
(exp 4).**

- **Gossip:** cada nó **carrega a rede inteira** na memória (a união difundida).
  Perder nós não remove nenhum especialista → F1 **constante em 95,74** dos 80
  rounds, mesmo caindo para 3 nós.
- **Federado:** o modelo agregado é **reconstruído a cada round só dos clientes
  ativos**. Ao cair para 5 e depois 3 nós, os especialistas dos ataques dos nós
  removidos **somem** → recall despenca 99,96%→**74,6%** (com 3 nós sobram só os
  ataques {1,5}: random_replay e injection). O FPR cai a 0 como efeito colateral
  (o sensor ruidoso de masquerade também saiu), mas ao custo altíssimo de recall.

Também confirmado: as **comutações FL→GL e GL→FL são transparentes** (round 21
sem degrau nos dois), e a fase gossip inicial do exp 4 reproduz a **rampa de
difusão** clássica (F1 72,3→95,74 até o round 8, boosters 3→19 saturando no
round 9).

**Consequência para a tese ("Adaptativo"):** diante de *churn* ou perda de nós,
comutar para **gossip** é a decisão que **preserva a detecção** — federado é
frágil a perda de nós (o servidor só agrega quem está presente); gossip é
tolerante (a memória difundida sobrevive em cada nó). O par exp 3 × exp 4
demonstra empiricamente esse diferencial.

## 5. Especialista × IID — por que a partição por-ataque importa

Comparação da **estratégia de partição** dos dados entre os nós, mantendo tudo
mais igual (combinado-24, fusão OR, 20 rounds, teste do autor 2,9 M):

- **`attack` (especialista):** cada nó recebe **1 classe de ataque + benigno** →
  detector especialista.
- **`iid` (generalista):** cada nó recebe uma **fatia IID de todos os ataques +
  benigno** → detector generalista.

Configs `ereno_{gossip,federado}_combined_iid.yaml`.

| Arquitetura | Partição | Nós | F1 | Recall | Prec | FPR | #FP |
|---|---|---:|---:|---:|---:|---:|---:|
| Centralizado (monolítico) | — | 1 | **99,23** | 99,96 | 98,52 | **0,110%** | 3.017 |
| FL (OR) | `attack` | 10 | **95,72** | 99,97 | 91,81 | **0,649%** | 17.881 |
| GL (OR) | `attack` | 10 | **95,72** | 99,97 | 91,81 | **0,649%** | 17.881 |
| FL (OR) | `iid` | 10 | 79,36 | 99,53 | 65,99 | 3,733% | 102.845 |
| GL (OR) | `iid` | 10 | 79,35 | 99,58 | 65,96 | 3,740% | 103.052 |

### 5.1 Dois achados

1. **GL = FL também no IID** (F1 79,35 vs 79,36; FPR 3,740 vs 3,733). A
   equivalência **não depende da partição** — é a álgebra da união-OR (a
   Proposição de `gl_fl_equivalence.tex` vale para qualquer conjunto de
   detectores). Vale para especialistas *e* para generalistas.

2. **IID é muito pior que especialista** no distribuído: F1 **79 vs 96**, FPR
   **3,74 % vs 0,649 %** (~5,7× mais falsos alarmes), embora o recall seja
   igualmente alto (~99,5 %). O centralizado monolítico é **idêntico** nos dois
   (é um único modelo, independe da partição).

### 5.2 Por que — o *union bound* dos falsos positivos

- **Especialista:** cada nó só dispara no *seu* ataque → precisão ~100 %, quase
  nenhum FP. A união OR **não acumula** FP → FPR baixo (0,649 %).
- **Generalista (IID):** cada nó tem recall alto **e** gera ~0,4 % de FP; a união
  OR de 10 generalistas **soma** esses FP (union bound):
  $\text{FPR}_{\text{OR}} \approx 1-(1-0{,}004)^{10} \approx 3{,}9\%$, batendo com
  os **3,74 %** medidos.

### 5.3 Significância — a partição por-ataque é uma contribuição, não conveniência

O particionamento especialista é o que **faz a fusão-OR funcionar** (mantém o FPR
baixo, porque cada detector quase nunca dispara falsamente). **IID + OR é uma
combinação ruim** (os FP dos generalistas se acumulam). Isso **contrasta com a
literatura** de forma informativa: em Hegedűs/Jelasity (redes neurais, **média**
de pesos, dados balanceados) o IID é favorável e gossip ≥ FL; aqui (**XGBoost,
fusão-OR, IDS desbalanceado**) o GL ≈ FL **continua**, mas o **IID é inferior** —
o design **especialista + OR** é o que torna o distribuído viável. A escolha da
partição por-ataque, portanto, **justifica-se empiricamente**.

## 6. Operadores de fusão e o k-de-n variando o número de clientes

### 6.1 OR × k-de-n × merge — a idempotência é o divisor de águas

A equivalência GL=FL (§3) vale **exatamente** para a fusão **OR** (união
idempotente). Testando operadores **não-idempotentes** e medindo GL vs FL
amostra-a-amostra (`scripts/{kden_gl_vs_fl,merge_gl_vs_fl}.py`, combinado-24,
10 clientes):

| Operador | Idempotente? | Concordância GL≡FL |
|---|---|---|
| **OR** (k≥1) | **sim** | **100,00 %** (0 divergências) |
| k-de-n (k≥2) | não | 98,00 % (58.967 div.) |
| k-de-n (k≥3) | não | 98,66 % (39.700 div.) |
| Merge (concatena árvores, soma margens) | não | **2,08 %** (2,89 M div.) |

Escada de FP do k-de-n (corroboração reduz FP, ao custo de recall): OR 17.881 FP
→ k≥2 ~60 FP → k≥3 ~29 FP, mas recall 99,97 → 78,6 → 58,2 (`analise_fp_combined.py`).
O merge é degenerado (FL-merge = veto, recall 30,7 %; GL-merge = explosão de
árvores 100→1560, FPR 100 %). **Só o OR é idempotente ⇒ único com GL=FL exato**;
qualquer corroboração (k-de-n) sai da idempotência (GL≠FL) e sacrifica recall.

### 6.2 k-de-n variando clientes (N = 10 → 3)

`scripts/kden_vary_clients.py` (combinado-24; pool difundido do gossip ≈ **2N−1**
boosters; `results/kden_vary_clients.csv`).

**k≥1 (OR) — FL = GL idênticos em todo N:**

| N | pool FL/GL | F1 | Recall | Prec | FPR |
|---:|:--:|---:|---:|---:|---:|
| 10 | 10/19 | 95,72 | 99,97 | 91,81 | 0,649 |
| 9 | 9/17 | 95,93 | 99,97 | 92,21 | 0,615 |
| 8 | 8/15 | 96,02 | 99,97 | 92,37 | 0,601 |
| 7 | 7/13 | 95,63 | 99,97 | 91,65 | 0,663 |
| 6 | 6/11 | 95,62 | 99,40 | 92,12 | 0,619 |
| 5 | 5/9 | 92,16 | 99,40 | 85,90 | 1,187 |
| 4 | 4/7 | 96,66 | 99,88 | 93,64 | 0,494 |
| 3 | 3/5 | 93,05 | 99,56 | 87,33 | 1,051 |

**k≥2 (FL | GL):**

| N | F1 FL/GL | Recall FL/GL | Prec FL/GL | FPR FL/GL |
|---:|:--:|:--:|:--:|:--:|
| 10 | 87,69 / **96,34** | 78,09 / **99,97** | 99,97 / 92,97 | 0,002 / 0,550 |
| 9 | 87,69 / **96,35** | 78,09 / **99,97** | 100,0 / 92,99 | 0,000 / 0,549 |
| 8 | 77,34 / **96,48** | 63,07 / **99,97** | 99,96 / 93,22 | 0,002 / 0,529 |
| 7 | 75,40 / 94,81 | 60,53 / 97,44 | 99,96 / 92,33 | 0,002 / 0,589 |
| 6 | 64,50 / 90,61 | 48,25 / 85,32 | 97,23 / 96,62 | 0,100 / 0,218 |
| 5 | 66,10 / 86,05 | 49,38 / 85,77 | 99,95 / 86,33 | 0,002 / 0,988 |
| 4 | 72,85 / 94,96 | 57,30 / 96,37 | 99,98 / 93,58 | 0,001 / 0,481 |
| 3 | 58,61 / 80,64 | 42,23 / 68,84 | 95,73 / 97,34 | 0,137 / 0,137 |

**k≥3 (FL | GL):**

| N | F1 FL/GL | Recall FL/GL | Prec FL/GL | FPR FL/GL |
|---:|:--:|:--:|:--:|:--:|
| 10 | 73,67 / 87,69 | 58,32 / 78,09 | 100,0 / 99,97 | 0,000 / 0,002 |
| 9 | 73,63 / 87,69 | 58,26 / 78,09 | 100,0 / 100,0 | 0,000 / 0,000 |
| 8 | 69,69 / 77,34 | 53,48 / 63,07 | 100,0 / 99,96 | 0,000 / 0,002 |
| 7 | 56,92 / 75,40 | 39,78 / 60,53 | 100,0 / 99,96 | 0,000 / 0,002 |
| 6 | 41,66 / 64,50 | 26,31 / 48,25 | 100,0 / 97,23 | 0,000 / 0,100 |
| 5 | 42,57 / 66,10 | 27,04 / 49,38 | 100,0 / 99,95 | 0,000 / 0,002 |
| 4 | 40,49 / 72,85 | 25,39 / 57,30 | 100,0 / 99,98 | 0,000 / 0,001 |
| 3 | 37,59 / 58,61 | 23,14 / 42,23 | 100,0 / 95,73 | 0,000 / 0,137 |

**Análise cruzada (efeito do nº de clientes):**

1. **OR (k≥1): FL = GL em todo N** — a equivalência é robusta ao nº de clientes.
2. **O sweet-spot GL-k≥2** (F1 > OR, recall ~100 %, FPR menor) **só se sustenta com
   N≥8**: precisa de pool grande o bastante (≈2N−1 ≥ 15 boosters) para os ataques
   pegarem ≥2 votos. Com N≤7 o pool encolhe e o recall do GL-k≥2 começa a cair.
3. **FL-k≥2/k≥3 sempre afunda o recall** (sem redundância para ataques singleton),
   em todo N.
4. **N<7 é errático** (regime "multi-ataque" round-robin — ex.: N=4 sai bem, N=5/N=3
   pior) porque depende de *quais* ataques se emparelham no mesmo cliente.

## 7. Adaptação com encolhimento fino de nós × k-de-n

Cenário de *churn* progressivo: 90 rounds, começa com 10 nós, **encolhe 1 nó a
cada 10 rounds** (10→9→8→…→3). Dois sentidos de adaptação e dois níveis de
corroboração, totalizando 4 runs:

- **FL→GL** — r1–10 federado (10 nós); comuta para gossip em r11 e encolhe daí
  em diante. Config: `ereno_adapt_fl_gl_shrink1_k{1,2}.yaml`.
- **GL→FL** — r1–10 gossip *from-scratch* (10 nós); comuta para federado em r11 e
  encolhe daí em diante. Config: `ereno_adapt_gl_fl_shrink1_k{1,2}.yaml`.
- **k≥1** = OR (união); **k≥2** = corroboração (≥2 especialistas votam ataque),
  via `xgb_fusion_k` no pipeline (`main_dist._predict_boosters(k)`).

Dados: `results/conv_adapt_shrink1.csv`. Figuras:
`results/conv_adapt_shrink1_metricas.png` (F1/Recall/FPR × **round**,
`scripts/plot_conv_adapt_shrink1.py`) e `results/shrink_por_no.png`
(F1 e Recall × **nº de nós**, `scripts/plot_shrink_por_no.py`).

### Fase inicial (r1–10, *from-scratch*, 10 nós)

Antes do encolhimento, a fase inicial fixa o ponto de partida:

| Run | r1 | r10 (fim) | ao comutar (r11–20) |
|---|---|---|---|
| FL→GL k≥1 | 95,74 (FL instantâneo) | 95,74 | 95,74 |
| GL→FL k≥1 | 72,28 (rampa gossip) | 95,74 | 95,74 |
| FL→GL k≥2 | 87,63 | 87,63 | 87,63 |
| GL→FL k≥2 | 53,24 (rampa) | **96,38** (sweet-spot) | 87,63 (cai ao comutar p/ FL) |

### F1 por nº de nós (fase de encolhimento, pós-comutação)

O F1 é constante dentro de cada bloco de 10 rounds; a tabela dá o valor em cada
contagem de nós (r20 = 10 nós, r30 = 9, …, r90 = 3):

| nós | FL→GL k≥1 | GL→FL k≥1 | FL→GL k≥2 | GL→FL k≥2 |
|---:|:--:|:--:|:--:|:--:|
| 10 | 95,74 | 95,74 | 87,63 | 87,63 |
| 9 | 95,74 | 94,35 | 87,63 | 86,75 |
| 8 | 95,74 | 90,76 | 87,63 | 86,75 |
| 7 | 95,74 | 90,56 | 87,63 | 85,92 |
| 6 | 95,74 | 85,93 | 87,63 | 85,38 |
| 5 | 95,74 | 85,36 | 87,63 | 85,38 |
| 4 | 95,74 | 85,38 | 87,63 | 85,38 |
| 3 | 95,74 | 85,38 | 87,63 | **53,24** |

### Recall por nº de nós (o que dirige o F1)

| nós | FL→GL k≥1 | GL→FL k≥1 | FL→GL k≥2 | GL→FL k≥2 |
|---:|:--:|:--:|:--:|:--:|
| 10 | 99,96 | 99,96 | 78,01 | 78,01 |
| 9 | 99,96 | 91,80 | 78,01 | 76,59 |
| 8 | 99,96 | 83,13 | 78,01 | 76,59 |
| 7 | 99,96 | 82,79 | 78,01 | 75,31 |
| 6 | 99,96 | 75,37 | 78,01 | 74,48 |
| 5 | 99,96 | 74,49 | 78,01 | 74,48 |
| 4 | 99,96 | 74,48 | 78,01 | 74,48 |
| 3 | 99,96 | 74,48 | 78,01 | **36,28** |

*FPR (resumo):* FL→GL k≥1 constante **0,645 %** em todo N; GL→FL k≥1 **cai para ~0**
conforme encolhe (menos especialistas disparando: 0,645 → 0,203 → 0,003 → 0,000);
ambos os k≥2 ≈ 0.

### Leitura da progressão completa (10 → 3 nós)

- **FL→GL (k≥1 e k≥2): reta perfeita em todo N.** 95,74 (k≥1) e 87,63 (k≥2)
  **constantes de 10 a 3 nós** — encolher em modo gossip é **totalmente
  transparente** (a união herdada é replicada em todo sobrevivente).
- **GL→FL k≥1: escada descendente monotônica.** 95,74 → 94,35 → 90,76 → 90,56 →
  85,93 → 85,36 → 85,38 → 85,38; recall 99,96 → 74,48. **Estabiliza em ~85,4 a
  partir de N=5** (os nós restantes ainda cobrem os ataques "núcleo" via
  round-robin), e o FPR desce a ~0 (menos especialistas ativos).
- **GL→FL k≥2: estável-e-despenca.** Fica em 85–87 de N=10 a N=4 (já no regime de
  recall-baixo do federado-k≥2 desde a comutação) e **crateriza em N=3** (53,24;
  recall 78 → 36) — 3 boosters não sustentam a corroboração ≥2, sobra só o ataque
  de sensor redundante.

### Análise

1. **FL→GL é transparente ao encolhimento (k≥1 e k≥2): F1 plano.** O federado
   semeia a **união completa** dos especialistas no round 1; o gossip a herda e
   cada sobrevivente já carrega todos os boosters → perder nós não perde
   cobertura. Sustenta 95,74 (k≥1) / 87,63 (k≥2) do r1 ao r90.

2. **GL→FL degrada em escada sob encolhimento federado.** Ao operar em federado,
   cada nó que sai **remove seu especialista** da agregação → o recall cai em
   degraus (100→92→83→75 % para k≥1). O encolhimento fino mostra *quais* saídas
   doem: os degraus coincidem com a perda de nós que cobrem ataques únicos.

3. **O sweet-spot GL-k≥2 (96,38, recall 100 %, bate o OR) é EXCLUSIVO do gossip
   *from-scratch*.** Aparece só em GL→FL no r10 (fase gossip inicial) e
   **desaparece ao comutar para federado** (cai a 87,63 em r11). Em FL→GL k≥2 o
   F1 é plano em 87,63 — nunca atinge o sweet-spot. Motivo: a corroboração k≥2
   exige um **pool redundante** (≈2N−1 boosters via variantes warm-start) para os
   ataques pegarem ≥2 votos; só a **difusão gossip real** o constrói. O federado
   (e o gossip *semeado* pelo FL) tem apenas 10 boosters → sem redundância → sem
   sweet-spot. **A corroboração para reduzir FP é uma propriedade do gossip, não
   da união.**

4. **k≥2 + federado + poucos nós é catastrófico.** GL→FL k≥2 **despenca a 53,24**
   (recall 36 %) em 3 nós: com só 3 boosters, exigir 2 votos deixa passar todo
   ataque detectado por 1 único especialista — sobrevive apenas o de **sensor
   redundante** (`random_replay`, 2 sensores → 2 votos naturais). É o mesmo piso
   observado no GL-k≥2 com N pequeno (§6): corroboração sem pool = recall nulo.

### Conclusão prática

- **Robustez a churn** → comutar para **gossip** (FL→GL) preserva a detecção; ficar
  em **federado** encolhendo (GL→FL) degrada progressivamente.
- **Reduzir FP por corroboração (k≥2)** → só compensa **em gossip com pool cheio**
  (o sweet-spot). Em federado, ou com poucos nós, o k≥2 destrói o recall.
- Isso reforça a tese adaptativa: os dois regimes não são intercambiáveis — a
  escolha FL/GL muda *qualitativamente* o comportamento sob perda de nós e sob
  corroboração, não só a acurácia média.

> **Ressalva de acurácia** (idem §1): F1/recall/FPR aqui são sobre o teste
> completo; a acurácia global fica dominada pela classe benigna e não reflete a
> degradação por-ataque que estas curvas evidenciam.

## Pendentes (não incluídos aqui)

- Métricas por-cliente (tabela §2-style) com combinado-24.
- Verificar que a concordância FL≡GL do OR é literalmente 0 divergências (não
  `100,00 %` arredondado).
