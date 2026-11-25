# Perda estática de nós — evaluation-v1, seção 5 (2026-11-20)

Script: `scripts/ev1_nodeloss.py` (sem treino; lê os scores do passo 1, seed 42, em
`results/.cache/ev1_step1_<domínio>_s42.npz`). Saídas: `results/ev1_nodeloss_<domínio>.{txt,json}`,
figura `results/ev1_nodeloss.{pdf,png}`. Change log do protocolo: entrada de 2026-11-20.

- Ordem de remoção (`default_rng(42).permutation`): 7, 8, 10, 4, 1, 12, 14, 11, 6, 3, 5, 13, 2, 9
  (sobram 2, 9 e 13 com N = 3).
- **Retenção:** cada sobrevivente mantém os especialistas recebidos; o detector é o do passo 1
  em todo N (constante por construção, supõe troca completa antes da perda).
- **Re-formado:** o agregado só tem os especialistas dos nós presentes (limite em que o
  conhecimento de um nó perdido sai com ele).
- Faixa: percentis 5–95 sobre 1.000 outras ordens (`default_rng(2026)`).

## Resultado (ordem do protocolo, re-formado)

| Domínio | Retenção (recall / FPR %) | Re-formado, N = 3 (recall / FPR %) | Tipos sem especialista em N = 3 | Recall N = 3, p5–p95 sobre 1.000 ordens |
|---|---|---|---|---|
| ERENO | 99.97 / 0.626 | 70.89 / 0.438 | 4 de 7 | 51.9–83.8 |
| CICIDS2017 | 99.70 / 2.632 | 51.58 / 0.210 | 22 de 26 | 2.9–93.6 |
| Morris-4 | 99.81 / 42.003 | 56.69 / 28.442 | 27 de 35 | 41.4–84.0 |
| CICIoT2023 | 99.97 / 28.569 | 99.80 / 16.184 | 24 de 30 | 98.1–99.97 |

## Leitura

- **ERENO:** cada tipo tem dois especialistas; um tipo só some quando os dois nós caem.
  inverse replay (nós 7 e 8) cai para 67,3 % em N = 12 — os outros especialistas ainda pegam
  2/3 dele.
- **CICIDS2017:** um especialista por tipo. Perder o nó 1 (Portscan) em N = 9, depois do nó 4
  (Infiltration Portscan), deixa 46,7 % das linhas de ataque sem especialista; recall 54,1 %.
  A FPR cai de 2,63 para 0,40 % em N = 7, quando sai o nó 14 (especialista compartilhado dos
  tipos pequenos, FPR local 2,34 %).
- **Morris-4:** queda gradual até 56,7 % em N = 3; a FPR também cai (42,0 → 28,4 %).
- **CICIoT2023:** o recall não cai (≥ 99,8 %) mesmo com 24 de 30 tipos sem especialista. Os
  especialistas sobreviventes (nós 2, 9, 13) disparam em 72,3–99,6 % das linhas dos tipos
  perdidos; o de DDoS TCP Flood dispara em 96,3 % delas com 0,03 % dos benignos. No CICIoT2023
  um especialista separa ataque de benigno, não o seu tipo dos outros (coerente com o nó 11
  sozinho a 99,4 % na 5.1). 20 dos 24 tipos perdidos ficam com recall ≥ 90 %; o pior, 40,3 %.
  A FPR cai de 28,57 para 16,36 % ao perder os nós 12 e 14 (web e força bruta).
- **Troca:** a retenção mantém todos os tipos ao custo de manter todas as fontes de falso
  positivo; o re-formado perde tipos e falsos positivos juntos.
- A ordem decide quais tipos caem: a faixa sobre 1.000 ordens é larga (CICIDS2017 em N = 3:
  2,9–93,6 %).

## Limites declarados

- Supõe a troca completa antes da perda; perda durante a difusão = replay T2 (seção 9).
- O re-formado é o limite (conhecimento sai de uma vez), não o FL típico: FedAvg mantém o
  modelo global e a contribuição de um cliente perdido se apaga aos poucos; o bagging de
  XGBoost do Flower acumula árvores.
- Mesma seed (42) e mesmos especialistas do passo 1; sem dispersão sobre seeds.
