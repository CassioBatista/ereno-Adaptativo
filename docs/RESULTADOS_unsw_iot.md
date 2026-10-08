# Resultados — UNSW IoT attack traces (perfil `iot-device-minute`, rótulos `unsw-iot-3`)

Fonte: Hamza et al., ACM SOSR 2019, iotanalytics.unsw.edu.au/attack-data (MIT-0):
`flowdata.zip` (6,3 MB), `annotations.zip`, `attackinfo.xlsx` (MD5 em
`results/unsw_iot_attack_md5sums.txt`). Script: `scripts/inspect_unsw_iot.py` →
`results/unsw_iot_audit.txt`.

## Dados

10 dispositivos IoT de consumo (câmeras Samsung/Netatmo, tomadas TP-Link/WeMo, sensor,
etc.), 32–49 dias cada, **uma linha por dispositivo e minuto** (570.460 dispositivo-minutos)
com contadores de pacotes/bytes dos fluxos do perfil MUD do dispositivo. Features comuns:
contadores somados por direção (From/To) × alcance (Local/Internet) × protocolo (Udp, Tcp,
Icmp, Arp, outros) × (pacotes, bytes) + NoOfFlows = 41. 234 ataques anotados de 10 min
(início/fim, fluxos afetados, tipo, taxa 1/10/100 pkt/s, origem local ou internet),
espalhados em 9 dos 83 dias, intercalados com tráfego normal (55.596 minutos normais nos
dias de ataque). Prevalência: 0,40 % dos dispositivo-minutos.

## Auditoria

- **Rótulos verificados pelo tráfego:** nos intervalos anotados, os contadores do protocolo
  afetado sobem vs a hora anterior em **100 %** dos intervalos de ARP spoofing, ping of death
  e Smurf (inclusive a 1 pkt/s; razão mediana 8–53×), mas só em 58–86 % dos de TCP/UDP
  (SYN flood, SYN reflection, UDP flood, SSDP/SNMP reflection).
- **Probe:** aleatório 0,85 → divisão temporal com intervalos inteiros 0,37 (ARP 0,91 se
  mantém; TCP/UDP caem).
- **Atribuição:** o contador é do dispositivo-**vítima**, não do emissor → perfil com
  atribuição `none` (isolar a vítima seria errado).

## Protocolo (divisão temporal, intervalos de ataque inteiros)

| | recall (minuto) | FPR (minuto) | F1 @ 13,75 | ataques detectados | FA / dispositivo-dia |
|---|---:|---:|---:|---:|---:|
| 7 categorias, GL k≥2 (14 esp.) | 76,8 % | **5,43 %** | 61,1 | 63/70 | 78,2 |
| **`unsw-iot-3`, GL k≥2 (6 esp.)** | **71,1 %** | **0,026 %** | **82,9** | **18/18** | **0,37** |
| `unsw-iot-3`, GL k≥1 | 87,8 % | 4,69 % | 69,6 | — | — |
| `unsw-iot-3`, FL k≥2, 3 nós | 23,3 % (ARP 0 %) | 0,007 % | — | — | — |
| `unsw-iot-3`, FL k≥2, 2 nós | 0 % | 0 % | — | — | — |

Por taxa (unsw-iot-3, minuto): 1 pkt/s 64 %, 10 pkt/s 70 %, 100 pkt/s 77 %. Variação com a
partição do benigno entre especialistas: num teste rápido com outra partição, 75,0 % /
0,034 %.

**Por que 7 categorias falham:** os falsos positivos vêm dos especialistas de SYN
reflection, SYN flood e UDP reflection (~20 mil disparos), espalhados (não perto de
ataques: FPR 7–10 % perto vs 5 % longe), metade num dispositivo (f4f5d88f0a3c): na
resolução de 1 min o ataque TCP/UDP soma centenas de pacotes, o mesmo que a variação
normal de um dispositivo movimentado. Os rótulos dessas categorias **não estão errados**;
é limite da representação. Normalização por patamar do dispositivo (log, mediana/IQR do
treino normal) foi testada nas 7 categorias e **piorou** (FPR 16,0 %) — não usada.

**Critério de `unsw-iot-3`:** categorias cujo ataque aparece nos contadores em 100 % dos
intervalos (seção ii da auditoria). Declarado: o critério foi aplicado depois de ver o
protocolo com 7 categorias falhar.

**k-de-n:** k≥1 4,69 % → k≥2 0,026 % de FPR (a corroboração derruba o falso alarme em 180×).
**ARP spoofing** (ataque de contexto) detectado: o contador por minuto já carrega o contexto.
