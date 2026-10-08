# Resultados — CICIDS2017 corrigido (perfil `it-flow`, rótulos `cicids2017c-7`)

Protocolo do ERENO aplicado sem alteração (`scripts/cross_domain_results.py`,
relatório `results/cross_domain_cicids2017c.{txt,csv}`; análise de janelas
`scripts/_cicids_window_analysis.py` → `results/cicids2017c_window_analysis.txt`).

## Configuração

- 2.099.965 fluxos, 5 dias; condições de uso da auditoria: Heartbleed excluído,
  "Infiltration - Portscan" → PortScan, Attempted → benigno, identificadores
  (IPs, portas, Timestamp, Flow ID) e `FWD/Bwd Init Win Bytes` fora das features
  → 80 features (sem seleção GRASP).
- 14 especialistas XGBoost (2 por categoria; depth 4, eta 0.1, 10 rounds,
  scale_pos_weight, seed 42); cada um treina na sua metade da categoria + 1/14 do benigno.
- Divisão temporal: em cada dia, os últimos 30 % do intervalo de cada episódio de
  ataque são TESTE (todos os fluxos dentro, inclusive benignos); segunda-feira:
  primeiros 70 % = calibração das janelas (fora do treino), últimos 30 % = teste benigno.
  Teste: 179.104 fluxos de ataque, 275.258 benignos.

## Encolhimento 14 → 3 nós (amostra)

| | F1 | recall | FPR | F1 @ 13,75:1 (prevalência ERENO) |
|---|---:|---:|---:|---:|
| GL, união dos 14, k≥2 (constante 14→3) | **98,87** | **98,36** | **0,388 %** | **96,58** |
| GL, k≥1 | 98,66 | 98,63 | 0,845 % | 93,82 |
| FL k≥2 com 9 nós | 60,36 | 43,45 | 0,341 % | — |
| FL k≥2 com ≤7 nós | 27,5 | 15,9 | 0,005 % | — |

Mesmo padrão do ERENO: o GL retém a união e não perde nada; o FL k≥2 perde a
corroboração quando o par de cada categoria some. Recall por categoria (GL, k≥2):
DoS 95,6 %, DDoS 99,9 %, PortScan 99,4 %, BruteForce 98,4 %, Bot 100 %,
Web 51,6 % (31 fluxos), Infiltration 63,6 % (11 fluxos).

## Triagem por janela global — NÃO transfere (ver a seção seguinte)

Política do ERENO (calibração só em janelas benignas da segunda-feira, margem ×1,25,
volume OU fração):

| janela | T | janelas de ataque detectadas | FPR janelas benignas |
|---|---:|---:|---:|
| 1 s | 11 | 684/2.544 (26,9 %) | 6/14.728 (0,04 %) |
| 10 s | 35 | 163/630 (25,9 %) | 3/1.819 (0,16 %) |

Por quê: os ataques de baixa taxa (BruteForce, Bot, Web, Infiltration, parte do
PortScan) têm **mediana de 1 fluxo marcado por segundo** — o mesmo volume dos
extremos benignos (máx. 8 fluxos marcados/s na calibração). A regra de fração fica
inutilizável (janelas benignas com 1 fluxo marcado → fração 1,0). Só os ataques
volumétricos passam (DDoS 99 %, DoS 56 % a 1 s / 94 % a 10 s). Com T=2 a 1 s:
61 % das janelas de ataque a 0,39 % de FPR — sem calibração a zero falso alarme.

**Leitura:** a triagem por volume/fração funciona quando o ataque é volumétrico em
relação à taxa de falsos alarmes benignos (ERENO: GOOSE/SV de alta taxa); em fluxos
corporativos, ataques lentos não se separam por volume.

## Triagem por especialista (2026-10-07) — resolve os ataques lentos

`scripts/_cicids_source_triage.py` → `results/cicids2017c_source_triage.txt`.
Contagem por janela **por especialista**: para cada categoria c, fluxos marcados pela
fusão (k≥2) em que um especialista de c votou; limiar por categoria
`T_c = max(ceil(1,25 · máx benigno) + 1, 2)` (calibração: segunda-feira, só benigno).
Um ataque lento acumula no seu especialista; falsos alarmes benignos se espalham.

| janela | global (ERENO) | por origem | por especialista | global OU especialista | FA benignos |
|---|---:|---:|---:|---:|---:|
| 1 s | 26,9 % | 26,9 % | 30,8 % | 30,8 % | 6 (0,041 %) |
| 10 s | 25,9 % | 25,9 % | 73,5 % | **73,5 %** | 3 (0,165 %) |
| 60 s | 49,0 % | 34,7 % | 74,8 % | 75,5 % | 1 (0,370 %) |

Por categoria a 10 s (global → especialista): BruteForce 0 → 90 %, Bot 0 → 60 %,
PortScan 37 → 54 %; DoS 94 % e DDoS 100 % iguais; Web e Infiltration continuam 0 %.
O alarme nomeia a categoria certa (categoria que mais excede o próprio limiar).

**Por origem (Src IP) foi descartado:** o gateway 172.16.0.1 (NAT) é a origem de 72,5 %
dos fluxos de ataque do teste e de 0 % da calibração — a chave aprenderia a topologia da
captura; e o limiar por origem fica igual ao global (falsos alarmes concentrados num host).
Os 3 falsos alarmes benignos a 10 s vêm do gateway em todas as regras.

**Generaliza para o ERENO** (`scripts/oos_specialist_triage.py` →
`results/oos_specialist_triage_ereno.txt`, blocked-5, margem ×1,25): global 276/282
(97,9 %) → global OU especialista **280/282 (99,3 %)**, mesmo 1/724 (0,14 %) benigno.
Com uma só chave/sem atribuição a regra global continua valendo; a chave por
especialista existe em todo perfil.

Também vale no Electra Modbus: ver `docs/RESULTADOS_electra.md`.

## Ablação de perfil (2026-10-07) — perfil sem tempo e sem atribuição (RQ5)

`scripts/_cicids_profile_ablation.py` → `results/cicids2017c_profile_ablation.txt`.
Mesmos dados, especialistas e divisão; só o perfil muda. `it-flow-blind`: tick a cada M
fluxos em ordem de chegada, atribuição `none` (alarme sem emissor → sem isolamento).
Detecção por fluxo idêntica (recall 98,36 %, FPR 0,388 %).

| perfil | duração mediana | rede inteira | rede inteira OU especialista | FA benignos |
|---|---:|---:|---:|---:|
| `it-flow`, tempo 10 s | 8,7 s | 25,9 % | 73,5 % | 3 / 1.819 (0,165 %) |
| `it-flow-blind`, M = 100 | 1,2 s | 78,2 % | **85,6 %** | 3 / 2.160 (0,139 %) |
| `it-flow-blind`, M = 300 | 5,0 s | 77,2 % | 87,2 % | 2 / 632 (0,316 %) |
| `it-flow-blind`, M = 1000 | 34 s | 84,2 % | 90,7 % | 1 / 164 (0,610 %) |
| `it-flow-blind`, M = 3000 | 146 s | 88,5 % | 92,3 % | 1 / 48 (2,08 %) |

Leitura: com janelas de contagem o denominador é fixo e a **regra de fração volta a
funcionar** (em janelas de tempo, uma janela benigna com 1 fluxo marcado tem fração 1,0).
As colunas não são diretamente comparáveis (a unidade de janela muda: 100 fluxos ≈ 1 s
durante ataque); com M grande há poucas janelas benignas e 1 FA vira 2 %.
