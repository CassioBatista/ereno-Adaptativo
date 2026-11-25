# Matriz de detecção cruzada dos especialistas — evaluation-v1, passo 1 (2026-11-20)

Script: `scripts/ev1_crossdetect.py` (sem treino; scores do passo 1, seed 42). Saídas:
`results/ev1_crossdetect_<domínio>.{csv,txt}`, figura `results/ev1_crossdetect.{pdf,png}`.
Análise post hoc, não pré-registrada.

- Célula (tipo t, especialista s): fração das linhas de teste de t que s marca (score ≥ 0,5).
- **Cobre:** marca ≥ 50 % das linhas do tipo. **Preciso:** FPR benigna do especialista ≤ 1 %.
  Os dois cortes são escolhas desta análise.
- Grupos: componentes conexas ligando tipos cobertos por um mesmo especialista preciso
  (ligação simples: dois tipos do mesmo grupo não precisam se cobrir diretamente).
  Especialistas ruidosos (FPR > 1 %) marcam quase tudo e não ligam comportamento.

## Por que existe detecção cruzada

Cada especialista é treinado com o seu tipo contra os benignos do seu nó; aprende como o tipo
se afasta do benigno. Tipos que se afastam do mesmo jeito são marcados pelos especialistas uns
dos outros. Ex.: DDoS TCP Flood (CICIoT2023) aprende `Header_Length < 248` (benigno: 99 %
acima de 925); sem Header_Length passa a usar `rst_count` e marca os mesmos floods
(`~/.audit_tmp/ciciot_tcpflood_rule.py`, `ciciot_drop_features.py`).

## Resultado

| Domínio | Tipos | Linhas cobertas por especialista preciso de outro nó | Blocos de comportamento comum | Tipos só com o próprio especialista |
|---|---|---|---|---|
| ERENO | 7 | 28,7 % (high_StNum, poisoned_high_rate) | cadeia via high_StNum/poisoned | random replay, injection, inverse replay, masquerade fault, masquerade normal (cada um em 2 nós) |
| CICIDS2017 | 26 | 97,4 % | DoS Hulk + DDoS + GoldenEye (50,5 % das linhas, nós 2 e 3); Portscan ↔ Infiltration Portscan | FTP-Patator, SSH-Patator, DoS Slowloris, DoS Slowhttptest, Botnet; pequenos via especialista compartilhado (FPR 2,33 %) |
| Morris-4 | 35 | 63,0 % (90,4 % contando ruidosos) | 8 tipos NMRI/CMRI (nó 4 cobre todos); 6 tipos MFCI/Recon (nós 8 e 14) | ~15 tipos MPCI/MSCI, cobertos só por especialistas ruidosos (FPR 4–21 %) |
| CICIoT2023 | 30 | 99,4 % | **21 tipos, 99,4 % das linhas**: todos os floods DoS/DDoS/Mirai + PortScan, PingSweep, VulnerabilityScan; o nó 11 (DDoS HTTP Flood, SlowLoris) cobre todos | 9 tipos de reconhecimento/web/força bruta, cobertos só por especialistas ruidosos (FPR 6–16 %) |

## Leitura

- O número de rótulos supera o número de comportamentos distintos, sobretudo no CICIoT2023
  (um bloco volumétrico) e no CICIDS2017 (um bloco DoS e um de varredura). No ERENO, os
  especialistas são quase todos específicos.
- A redundância dentro dos blocos é o que mantém o recall na perda de nós (CICIoT2023:
  ≥ 99,8 % com 3 nós; `RESULTADOS_nodeloss_ev1.md`). Os tipos fora de blocos são os que caem
  quando o nó sai (CICIDS2017: Portscan, FTP/SSH-Patator) e os que o k = 2 veta.
- Os tipos cobertos só por especialistas ruidosos são a fonte dos falsos positivos (Morris-4,
  CICIoT2023).
