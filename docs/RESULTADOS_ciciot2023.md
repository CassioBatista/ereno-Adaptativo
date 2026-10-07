# Resultados — CICIoT2023 (perfil `iot-flow`, rótulos `ciciot2023-7`)

Protocolo do ERENO aplicado sem alteração (`scripts/cross_domain_results.py`,
relatório `results/cross_domain_ciciot2023.{txt,csv}`; sensibilidade
`scripts/_ciciot_benign_sensitivity.py` → `results/ciciot2023_benign_sensitivity.txt`).

## Configuração

- Amostra auditada: 10 dos 169 arquivos do release CSV; part-00000..06 = treino
  (1.665.103 linhas), part-00007..09 = teste (701.853 linhas; 16.665 benignas = 2,4 %).
- IAT excluída (relógio de captura); 45 features; sem tempo → sem triagem por janela.
- 14 especialistas XGBoost, 2 por categoria, mesmos hiperparâmetros do ERENO.

## Encolhimento 14 → 3 nós (amostra)

| | recall | FPR benigno | F1 natural | F1 @ 13,75:1 |
|---|---:|---:|---:|---:|
| GL, união dos 14, k≥2 (constante 14→3) | **99,69** | **23,4 %** | 99,56 | **38,2** |
| GL, k≥1 | 99,77 | 30,3 % | 99,52 | 32,4 |
| FL k≥2 com 3 nós | 97,14 | 0,05 % | 98,55 | — |

O GL mantém a detecção sob perda de nós, mas o **FPR benigno é alto**: o F1
"natural" (99,6) é inflado porque 97,6 % do teste é ataque; na prevalência do ERENO
a precisão cai para ~24 %.

## Por que o FPR é alto (sensibilidade)

| variante | k≥2 recall | k≥2 FPR | F1 @ 13,75 |
|---|---:|---:|---:|
| protocolo (1/14 do benigno, 10 rounds, depth 4) | 99,71 | 24,2 % | 37,5 |
| 1/14 do benigno, 100 rounds, depth 6 | 99,78 | 16,8 % | 46,4 |
| todo o benigno por especialista, 10 rounds | 99,70 | 23,6 % | 38,0 |
| todo o benigno, 100 rounds, depth 6 | 99,72 | 11,7 % | 55,3 |

O FPR vem dos especialistas de **Recon, Web, BruteForce e Spoofing** (8–16 % de
FPR benigno cada; DDoS/DoS/Mirai ≈ 0 %).

## Limiares calibrados por especialista (`_ciciot_threshold_calibration.py`)

Limiar de cada especialista ajustado em benigno separado do treino (30 % do benigno de
treino) a um orçamento de FPR. Com todo o benigno, 100 rounds, depth 6 e orçamento
0,1 %: FPR fundido k≥2 **0,17 %**, recall total 99,05 % — mas o recall total é
dominado por DDoS/DoS/Mirai; as categorias de baixo volume caem para Recon 51,5 %,
Web 30,9 %, BruteForce 30,6 %, Spoofing 50,9 %. O FPR não se resolve sem sacrificar
justamente essas categorias.

## Causa: rótulos por sessão nas categorias de baixo volume (`_ciciot_label_protocol_check.py`)

As linhas trazem indicadores de protocolo; nas classes de baixo volume, o protocolo do
próprio ataque quase não aparece:

| rótulo | linhas com o protocolo do ataque |
|---|---:|
| MITM-ArpSpoofing (ARP) | 0,1 % |
| DNS_Spoofing (DNS) | 0,2 % |
| Recon-PingSweep (ICMP) | 0,0 % |
| Recon-HostDiscovery (ARP/ICMP) | 0,5 % |
| ataques web (HTTP/HTTPS) | 28–50 % (benigno: 75 %) |
| DictionaryBruteForce (HTTP/HTTPS/Telnet/SSH) | 51 % |
| DDoS/DoS ICMP e UDP floods | 99,5–99,9 % |

A mistura de protocolos dessas classes é a do benigno (ex.: MITM-ArpSpoofing 67 %
HTTPS / 71 % TCP / 23 % UDP vs benigno 71 % / 86 % / 7 %), e 34–66 % das suas linhas
estão mais perto de uma linha benigna do que de qualquer linha da própria classe
(DDoS/DoS/Mirai: 0 %). Conclusão: **nas categorias Recon, Web, BruteForce e Spoofing
os rótulos marcam a sessão de captura, não o ataque** — o mesmo defeito do
CIC-BCCC-NRC-IoMT-2024 (mesmo laboratório e metodologia). Os especialistas dessas
classes aprendem tráfego benigno como ataque; o FPR de 23 % é consequência dos
rótulos, não da arquitetura. (Ressalva: se o extrator ignora pacotes não-IP, a
ausência de ARP pode ser do extrator; a conclusão — as linhas não contêm o ataque —
não muda.)

## Subconjunto com rótulos consistentes (`_ciciot_volumetric_subset.py`)

Só DDoS, DoS e Mirai (6 especialistas): GL k≥2 recall **99,64 %**, FPR **0,042 %**,
F1 @ 13,75:1 **99,53**. Mas o encolhimento quase não pesa: com 3 nós (um por
categoria) o FL k≥2 ainda tem 99,25 % de recall — as três categorias são inundações
parecidas e um especialista detecta as outras. Subconjunto fácil, pouco informativo
sobre recuperação.
