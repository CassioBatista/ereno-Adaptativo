# Grade do replay de trust level — evaluation-v1, seção 9 (2026-11-20/21)

> **SUPERADO (2026-11-23/24) pelo replay fiel** (`results/ev1_tl/faithful`,
> `scripts/ev1_tl_faithful_summary.py`, `results/ev1_tl_faithful_summary.txt`): enlaces do
> cenário (os dois lados se respondem) e rounds em 60 r + 30 s. Resultados reportados:
> ReSIDS nunca o pior em 12 células informativas (média 1,6, máx. 6,0 pontos atrás do melhor);
> FL fixo pior em 7 (19,0 / 82,0), sempre com o agregador fora; GL fixo pior em 5 (8,4 / 48,1),
> por não usar o agregador quando ele responde. Perguntas do scenario2.md: o especialista de
> 3–4 sai da região sob ReSIDS e GL fixo, nunca sob FL fixo; o cliente 5 não recebe o de
> 11–12 antes dos 45 min em nenhuma; o de 9–10 chega a 1 e 8 aos 75 min nas três. Decisões:
> 4 trocas por nó no T1 com settle 5 s (5–7 sem settle; 3–8 pela rede direta); nenhum round
> misturando F e G. O conteúdo abaixo é o registro da grade anterior.

- 48 execuções: 6 configurações × T1/T2 × 4 domínios; Δ = 60 s; particionador `specialists`;
  seed 42; chegada dos nós 3, 4, 11, 12 aos 30 min e 9, 10 aos 60 min; `monolithic: false`.
  Rodada de 17:41 a 20:53 (3 h 12 min), uma por vez, sem falhas.
- Execuções: `results/ev1_tl/grid_d60/<run>/node_eval.{csv,txt}`; runner
  `~/.audit_tmp/validate_flgl/run_tl_grid.sh` + `tl_grid_template.yaml`.
- Consolidação: `scripts/ev1_tl_grid_summary.py` → `results/ev1_tl_grid_summary.{txt,csv}`,
  figura `results/ev1_tl_grid.{pdf,png}`.
- **Protegido:** recall nos tipos do bloco ≥ 95 % do recall do detector com todos os especialistas.

## Linha do tempo (vista pelo ReSIDS)

- T1: 0–15 min local (TL subindo); 16–30 FL; 31–45 local: agregador perdido e anel partido —
  12 nós ficam com UM vizinho (TL = 15, abaixo da faixa de gossip, que exige os dois) e os nós
  1 e 14 com nenhum; o gossip fixo continua trocando pelo vizinho que restou; 45–60 gossip;
  61+ FL. (Corrigido em 2026-11-23: antes dizia "TL cai com os probes respondendo"; o TL é a
  soma dos alvos que respondem — `~/.audit_tmp/t1_pernode_check.py`.)
- T2: 16–30 FL; 31 gossip (agregador perdido); 46 nós 3–6 isolados; 61 nós 1–8 isolados;
  76 FL para nós 1 e 8–14; nós 2–7 isolados até o fim.

## Resultados principais (ERENO e CICIDS2017; Morris-4 e CICIoT2023 têm detecção cruzada)

| | Bloco 30 min | Bloco 60 min |
|---|---|---|
| T1 | ReSIDS: 13/14 já no round dos 30 min (exposição 3,9 %); FL fixo espera até 60 min (56,6 % ERENO, 52,6 % CICIDS); gossip fixo: mediana 26 / 18 min | todos em 1–2 rounds; gossip fixo 12 min (CICIDS) |
| T2 | ReSIDS = FL fixo (round FL dos 30 min); gossip fixo 12 / 8 min; ERENO: 4 nós nunca recebem | não chega aos 6 nós isolados; CICIDS: ReSIDS 4 min (52,4 %), FL fixo 16 min (63,9 %), gossip fixo 4 min (49,6 %, alcança nós 1 e 8 antes) |

- Re-formado, T2: aos 75 min perde os especialistas dos nós 2–7; só os 6 isolados ficam
  protegidos contra o bloco dos 30 min; recall médio final 91,4 % (ERENO) e 72,8 % (CICIDS),
  mínimo 84,9 / 53,1 %, contra 99,96 / 99,47 % com retenção. T1: igual à retenção.

## Argumentos (b), (c), (d) — `scripts/ev1_tl_arguments.py` → `results/ev1_tl_arguments.txt`

ReSIDS × GL fixo × FL fixo (todos com retenção), ERENO e CICIDS2017, T1 e T2; só logs e traces.

- **(b) Exposição evitável** (só nó-rounds de nós que alcançam o agregador ou um vizinho), média
  das 8 células domínio × trace × bloco: **ReSIDS 3,8 %**, GL fixo 23,7 %, FL fixo 17,9 %
  (todas as nó-rounds: 8,0 / 29,6 / 21,6 %). Sem as duas células do bloco dos 30 min no T1
  (fronteira): ReSIDS 3,7 %, GL 18,8 %, FL 6,3 %. CICIDS2017/T2 bloco 60, alcançáveis:
  ReSIDS 17,9 %, GL 13,0 %, FL 37,7 %.
- **(c) Comunicação:** NÃO é argumento a favor do ReSIDS.
  - Volume que precisa circular (delta, cada booster uma vez por nó): igual nas três políticas,
    1,5–1,7 MB (ERENO) e 2,5–2,8 MB (CICIDS2017).
  - Como o pipeline troca (estado completo): ReSIDS ≈ FL fixo (51–105 MB, 772–1.063 mensagens),
    GL fixo muito menos (10–24 MB, 133–153 mensagens; um head por round).
  - Leitura: a rapidez do ReSIDS custa o mesmo que o FL; o excesso vem do reenvio do agregado
    completo a cada round FL, que uma troca por delta eliminaria (nota de implementação).
- **(d) Estabilidade e coerência das decisões** (grade de 1 s):
  - T1: trocas de modo por nó com TL e settle 5 s = **4 em todos os nós**; com settle 0 = 5–7;
    guiado pelos próprios probes = 3–10 (média 6,1). T2: 3–4 / 3–5 / 3–6.
  - Modo do TL ≠ modo dos probes: 15,9 % do tempo no T1 (sobretudo 30–45 min: nós com um só
    vizinho ficam com TL = 15 → local pela faixa de gossip, enquanto o modo dos probes é gossip
    com qualquer vizinho) e 4,6 % no T2 (nós 1 e 8 entre 60 e 75 min, mesmo motivo).
  - Rounds com federado e gossip misturados nas execuções do ReSIDS: 1 no T1, 0 no T2.

## Chegadas deslocadas (+5 e +20 min) e arrependimento — exploratório, após os resultados

24 execuções (21/11, 06:48–08:49), `results/ev1_tl/arrival_offsets/`, resumo
`results/ev1_tl_offset_summary.txt`, arrependimento `scripts/ev1_tl_regret.py` →
`results/ev1_tl_regret.txt`.

- 16 células (domínio × trace × chegada × bloco) em que as políticas diferem > 1 ponto.
  Arrependimento = exposição − melhor das três.
  - **ReSIDS: nunca o pior (0/16)**; melhor ou a 1 ponto em 9/16; arrependimento médio 1,6,
    máximo 8,3 pontos.
  - FL fixo: pior em 8/16; médio 12,5, máx. 52,7. GL fixo: pior em 8/16; médio 22,3, máx. 85,7.
  - Só chegadas deslocadas: ReSIDS médio 1,9 (máx. 8,3); GL 17,7 (85,7); FL 8,4 (38,8).
- **Tolerância à falha do agregador, caso limpo:** ERENO/T2, bloco 1 aos 35 min (agregador fora
  30–75): FL fixo nunca difunde os especialistas dos nós 3 e 4 (isolados aos 45) → 0/14,
  exposição 100 %; ReSIDS 10/14 (61,2 %); GL fixo 9/14 (66,1 %).
- **Limites:** durante a queda o ReSIDS não é mais rápido que o GL fixo (mesmo gossip, um nó por
  round); no T1 entre 30 e 45 min os nós ficam com um só vizinho (TL = 15) → local pela faixa de
  gossip (≥ 30, decisão de construção) → GL fixo melhor (CICIDS +5: 26,5 vs 34,8 %, o maior
  arrependimento do ReSIDS). Com a faixa começando em 15, esses nós fariam gossip. Com +20 min no T2 nenhum
  protege o bloco 1 (nós 3 e 4 já isolados).
- O caso de fronteira do T1 (registrado) está resolvido: com +5 min o ReSIDS espera o gossip
  dos 45 min, como previsto, e fica igual ou melhor que os modos fixos no ERENO.

## Ressalva (importante)

- **T1, bloco dos 30 min:** a vantagem do ReSIDS sobre o FL fixo vem de UM round na fronteira da
  queda: o TL sai da banda federada 5 s depois da mudança (settle), então o round dos 30 min
  ainda é federado; o FL fixo, guiado pelos probes, já está local nesse round. Um bloco que
  chegasse um round depois esperaria o gossip dos 45 min. As chegadas (30/60 min) coincidem com
  fronteiras de fase dos dois traces.
- Proposta: verificação com chegadas deslocadas (p. ex. 35 e 65 min), ao menos ReSIDS, gossip
  fixo e FL fixo, em ERENO e CICIDS2017 (exploratória; o protocolo registra 30/60 min).
- Pendente pré-registrado: sensibilidade a Δ (5–600 s).
