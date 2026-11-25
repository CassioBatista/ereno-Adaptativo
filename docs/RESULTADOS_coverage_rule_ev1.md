# Regra de cobertura dos especialistas — exploratória (2026-11-20)

Script: `scripts/ev1_coverage_rule.py [--seed S]`; resumo das seeds:
`scripts/ev1_coverage_rule_seeds.py` → `results/ev1_coverage_rule_seeds.txt`. Por seed:
`results/ev1_coverage_rule[_s<seed>].{txt,json}`; cobertura de 50 % (seed 42):
`results/ev1_coverage_rule_c0.5_p0.01.txt`. Change log do protocolo: 2026-11-20.

## Regra (igual nos quatro domínios, decidida só com dados de treino)

- Cada nó aplica os especialistas recebidos às suas linhas de treino:
  - linhas de ataque do seu tipo → cobertura C[t, s] (cobre se ≥ 90 %);
  - linhas benignas → FPR dos especialistas de outros nós (preciso se ≤ 1 %), fora da amostra.
- Cobertura gulosa: **dois especialistas precisos em nós distintos por tipo** (cover-2), quando
  existem; tipos que nenhum preciso cobre mantêm o próprio especialista.
- *precise-2*: cover-2 sem os imprecisos. O teste só pontua.

## Validação nas seeds 42–46

- **cover-1 (um cobridor) falha no CICIDS2017** em 3 de 5 seeds: o especialista de DDoS
  cobre 91–95 % das linhas de treino do DoS Hulk, substitui o especialista dele, e o DoS Hulk
  cai para 91–94 % no teste (recall global até −2,67 pontos). Nos outros três domínios,
  cover-1 não perde nada em nenhuma seed.
- **cover-2 não perde nenhum tipo (≥ 5 pontos) em nenhuma seed nem domínio**; recall dentro de
  0,01 ponto; FPR igual ou menor. É a variante reportada.

| Domínio | cover-2: mantidos (seeds 42–46) | precise-2: mantidos | precise-2: recall % | precise-2: FPR % | precise-2: F1 a 6,8 % |
|---|---|---|---|---|---|
| ERENO | 10 de 14 | 10 | 99,96 | 0,63–0,67 | 95,6–95,9 |
| CICIDS2017 | 16–17 de 17 | 15–16 | 99,55–99,73 | 0,50–1,02 (all: 2,15–3,00) | 93,3–96,5 |
| Morris-4 | 28–29 de 35 | 14 | 65,4–67,4 | 0,90–1,72 | 70,6–73,6 |
| CICIoT2023 | 13–14 de 29 | 3–4 | 99,36–99,54 | 0,21–0,76 (all: 28,6–31,2) | 94,8–98,3 |

- precise-2 perde, em todas as seeds: CICIDS2017 6–7 tipos com < 100 linhas de teste e parte
  de Slowloris/Slowhttptest; CICIoT2023 os 11 tipos rotulados por sessão (0,8 % das linhas)
  para 0–72 %; Morris-4 os 15 tipos só cobertos por imprecisos.
- precise-1 (um cobridor) no CICIoT2023 deixa um único especialista (DDoS SlowLoris): 99,16–99,37 %
  / 0,012–0,024 % — mas a regra de um cobridor falha no CICIDS2017, então não é reportada como
  resultado.
- Limiar de cobertura 50 %: perde recall (ERENO inverse replay → 60 %).
- A poda é aplicada após a troca; não muda o que a retenção guarda na perda de nós.
