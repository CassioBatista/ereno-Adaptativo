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

## Pendentes (não incluídos aqui)

- Métricas por-cliente e curva de convergência com combinado-24 (análogos a
  `RESULTADOS.md §2–§3`).
- Adaptativos com combinado-24: FL→GL e GL→FL com encolhimento de nós
  (`conf/experiments/ereno_adapt_*_combined.yaml`).
