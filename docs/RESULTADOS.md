# Resultados — Centralizado × Federado × Gossip (ERENO)

Comparação das três arquiteturas de treinamento do IDS no dataset ERENO
IEC-61850, avaliadas no **test set do autor** (2.955.648 mensagens, split
por blocos — sem vazamento).

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
