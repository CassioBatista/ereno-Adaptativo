# Resultados — CICIDS2017 (Centralizado × FL × GL)

Documento **separado** (espelha `RESULTADOS_ereno_24feat.md`, que é do ERENO
IEC-61850). Aqui ficam **todos** os resultados do CICIDS2017 — seleção de
features (GRASP) e os experimentos §1–§7 — para não sobrescrever os do ERENO.

## Configuração do experimento

- **Dataset**: CICIDS2017 (`all_in_one_cicids_v2`, formato ARFF, 78 features
  `F1..F78`). ~1,31 M amostras.
- **Classes** (1 benigna + 10 ataques):

  | classe | amostras | classe | amostras |
  |---|---:|---|---:|
  | BENIGN | 1.039.547 | Slowhttptest | 5.499 |
  | hulk | 230.124 | brute | 1.507 |
  | GoldenEye | 10.293 | xss | 652 |
  | ftp | 7.935 | sql | 21 |
  | ssh | 5.897 | Heartbleed | 11 |
  | slowloris | 5.796 | | |

- **Particionador**: `attack` — 10 clientes especialistas (1 classe/cliente).
  Conforme decidido, **mantêm-se as 10 classes** (as minúsculas viram
  especialistas fracos = achado honesto).
- **Distribuído (FL/GL)**: Flower, estratégia `xgb_bagging`, fusão **OR**
  (`xgb_union`). Configs: `conf/experiments/cicids_*.yaml`.
- **Centralizado**: monolítico XGBoost (mesmas features, mesmo teste, sem Flower).

---

## 0. Seleção de features (GRASP)

Mesma metodologia do ERENO: **superset por-ataque ∪ núcleo global penalizado**.

### 0.1 GRASP por-ataque (✅ concluído)

`scripts/grasp_por_ataque_cicids.py` (sample 40k, normal-cap 40k, no-improve 8,
`--min-pos 200`, XGBoost, CV 5-fold; problema binário ataque-vs-BENIGN).

**Pulados** por inviabilidade estatística (< 200 positivos para CV 5-fold):
`Heartbleed` (11) e `sql` (21) — suas features ficam cobertas pelo núcleo global.

| ataque | # feat | features | f1_cv |
|---|---:|---|---:|
| ssh | 4 | `[1, 37, 64, 67]` | 100,00 |
| ftp | 5 | `[1, 64, 65, 66, 67]` | ~99,9 |
| Slowhttptest | 6 | `[1, 2, 29, 39, 53, 67]` | 99,87 |
| slowloris | 6 | `[1, 14, 20, 25, 67, 68]` | 99,84 |
| GoldenEye | 7 | `[1, 43, 66, 67, 68, 70, 75]` | 99,98 |
| xss | 7 | `[1, 25, 29, 38, 64, 67, 68]` | 98,98 |
| hulk | 8 | `[1, 6, 20, 35, 38, 44, 55, 67]` | 99,98 |
| brute | 13 | `[1, 9, 21, 25, 28, 37, 38, 46, 63, 67, 68, 69, 76]` | ~99,5 |

**Superset por-ataque (29 features):**

```
[1, 2, 6, 9, 14, 20, 21, 25, 28, 29, 35, 37, 38, 39, 43, 44, 46, 53, 55,
 63, 64, 65, 66, 67, 68, 69, 70, 75, 76]
```

Observação: **features 1 e 67 aparecem em TODOS os 8 ataques** (núcleo
universal); 68 na maioria. Os ataques do CICIDS separam **trivialmente**
(99,8–100 % de f1 com 4–13 features).

### 0.2 GRASP global penalizado λ=0,05 (✅ concluído)

`scripts/grasp_global_penalizado_cicids.py --lambda 0.05 --sample 150000`.
Núcleo global parcimonioso (regularização L0 de cardinalidade):

**7 features:** `[1, 19, 25, 35, 40, 67, 68]`

Note que **1, 67, 68** (o núcleo universal do por-ataque) reaparecem aqui — o
penalizado acrescenta apenas **19, 25, 35, 40** de novo.

### 0.3 Conjunto combinado adotado (✅ concluído)

`combinado = superset por-ataque (29) ∪ núcleo penalizado (7)` →
`features/all_in_one_cicids_combined.json` (via `scripts/combinar_cicids.py`).

**Conjunto CICIDS adotado — 31 features:**

```
[1, 2, 6, 9, 14, 19, 20, 21, 25, 28, 29, 35, 37, 38, 39, 40,
 43, 44, 46, 53, 55, 63, 64, 65, 66, 67, 68, 69, 70, 75, 76]
```

O penalizado só somou **19 e 40** ao superset (as demais já estavam). Comparável
em espírito ao **combinado-24 do ERENO** (aqui 31 de 78 features). Tempo total do
GRASP (phase A): ~2,2 dias (estágio 1 por-ataque + estágio 2 global penalizado).

---

## 1–7. Experimentos (⏳ pendentes)

Após adotar o conjunto combinado, replicar as seções do ERENO:

1. Métricas globais (Centralizado × FL × GL) + ressalva de acurácia.
2. Ganho do combinado vs o global-xgb existente.
3. Equivalência GL = FL (OR idempotente).
4. Adaptabilidade — comutação FL↔GL + encolhimento de nós.
5. Especialista × IID.
6. Operadores de fusão + k-de-n variando clientes.
7. Adaptação com encolhimento fino × k-de-n.

> **Teste de H6 (validade externa):** se GL = FL exato e o *sweet-spot* k≥2
> gossip-exclusivo **reaparecerem** aqui (domínio TI, muito diferente do
> IEC-61850), a tese passa de "propriedade do domínio" para "**propriedade do
> operador**".
