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

## 1. Métricas globais — Centralizado × FL × GL (OR), N=10

Teste completo (split interno 20%), combinado-31, particionador attack, benign_cap
500k, 20 rounds. Configs `cicids_{federado,gossip}_combined.yaml`.

| Métrica | Centralizado | Federado (OR) | Gossip (OR) |
|---|---|---|---|
| F1 | 99,79% | **96,26%** | **96,26%** |
| Precision | 99,62% | 92,79% | 92,79% |
| Recall | 99,97% | 99,99% | 99,99% |
| FPR | 0,205% | 4,158% | 4,158% |
| #FP | 205 | 4.158 | 4.158 |
| VP / FN | 53.530 / 17 | 53.541 / 6 | 53.541 / 6 |

> **Ressalva de acurácia** (idem ERENO): a acurácia global é dominada pela classe
> benigna; use F1/FPR por-classe. O **FPR distribuído é alto (4,16 %)** — bem acima
> do ERENO (0,649 %) — porque os especialistas do CICIDS "sujam" mais (disparam em
> benignos) e a união-OR acumula os FP (*union bound*). É um achado real do dataset.

## 3. Equivalência GL = FL — **H6 CONFIRMADO** ✅

**Federado e Gossip produzem métricas byte-a-byte idênticas** (96,2569 % / 92,7936 %
/ 99,9888 % / 4,1580 %; VP=53.541, VN=95.842, FP=4.158, FN=6 nos dois). A
equivalência **GL = FL exato** vale também no CICIDS2017 — um domínio (TI) totalmente
diferente do IEC-61850.

> **Significado:** a equivalência é **propriedade algébrica do operador OR**
> (idempotente), **não** do domínio ERENO. Gap Central→distribuído = **3,53 pp**,
> quase igual ao ERENO (3,51 pp). É a validade externa da tese.

## 5. Especialista × IID — achado **INVERTIDO** vs ERENO

Configs `cicids_{federado,gossip}_combined_iid.yaml`.

| Partição | Arq | F1 | Prec | Recall | FPR | #FP |
|---|---|---|---|---|---|---|
| attack (especialista) | FL = GL | 96,26 | 92,79 | 99,99 | 4,158 | 4.158 |
| iid (generalista) | FL | 97,48 | 96,99 | 97,98 | 1,630 | 1.630 |
| iid (generalista) | GL | 97,62 | 95,72 | 99,60 | 2,382 | 2.382 |

**No CICIDS o IID SUPERA o especialista** (F1 97,5 vs 96,26; FPR menor) — o **oposto**
do ERENO (onde especialista 96 ≫ IID 79). Motivo: no ERENO cada especialista tinha
precisão ~100 % (união-OR limpa); no CICIDS os especialistas são "sujos" (FPR alto
individual), então a união-OR acumula FP, enquanto os generalistas IID, mais
calibrados no benigno, têm FPR menor.

> **RQ2 tem resposta dataset-dependente:** "especialista > IID" **não** é universal —
> depende de os especialistas serem limpos. É um achado honesto que qualifica a
> contribuição (e evita generalização indevida). *Obs.:* no IID, FL e GL divergem
> levemente (97,48 vs 97,62) — as variantes warm-start do gossip são generalistas
> distintos (não redundantes como no caso especialista), então a união-OR difere.

## 6. k-de-n (k≥2) — também diferente do ERENO

Configs `cicids_{federado,gossip}_combined_k2.yaml`.

| Regra | Arq | F1 | Recall | Prec | FPR | #FP |
|---|---|---|---|---|---|---|
| OR (k≥1) | FL = GL | 96,26 | 99,99 | 92,79 | 4,158 | 4.158 |
| k≥2 | FL | 97,18 | 96,10 | 98,27 | 0,904 | 904 |
| k≥2 | GL | **97,87** | 98,91 | 96,85 | 1,725 | 1.725 |

No CICIDS, **k≥2 melhora AMBOS** (FL não colapsa — recall 96 %, não 78 % como no
ERENO): corta muito os FP (4.158 → 904/1.725) mantendo recall alto. O **GL-k≥2 ainda
vence** (97,87). A diferença vs ERENO: o FPR do CICIDS é alto o bastante para a
corroboração compensar mesmo no federado.

## 4 e 7. Adaptabilidade (comutação FL↔GL + encolhimento)

Runs concluídos (configs `cicids_adapt_*`): coarse (80r, 10→5→3) e fine-shrink (90r,
10→3, k1/k2), FL→GL e GL→FL. **Trajetórias por-round a extrair/plotar** (dados nos
logs `~/cicids_batch/B_cicids_adapt_*.log`) — pendente de tabela/figura.

## Síntese cross-dataset (ERENO × CICIDS)

| Achado | ERENO | CICIDS | Conclusão |
|---|---|---|---|
| **GL = FL exato (OR)** | sim | **sim** | **propriedade do operador** (H6 ✓) |
| Especialista × IID | espec. ≫ IID | **IID > espec.** | dataset-dependente (limpeza dos especialistas) |
| k≥2 | só ajuda GL | ajuda **ambos** | dataset-dependente (nível de FPR) |
| Gap Central→dist. | 3,51 pp | 3,53 pp | consistente |

A equivalência **generaliza**; os trade-offs específicos **não** — o que fortalece o
paper (validade externa + nuance honesta, evitando *overclaim*).
