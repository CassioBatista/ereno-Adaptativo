# Resultados — Electra Modbus (perfil industrial, rótulos `electra-4`)

Fonte: perception.inf.um.es/ICS-datasets (`electra_modbus.zip`, 56 MB; CSV 1,5 GB,
16.289.277 pacotes; MD5 em `results/electra_md5sums.txt`). Auditoria:
`scripts/inspect_electra.py` → `results/electra_audit.txt`.

## Condição de uso (auditoria)

Três rótulos marcam o **caminho** (host MitM 08:00:27:79:b0:4a), não o pacote:
`MITM_UNALTERED` (83 % com conteúdo de pacote normal), `REPLAY_ATTACK` (97 %),
`READ_ATTACK` (66 %); o MAC sozinho prevê 95 % dos rótulos. **Excluídos do fluxo.**
`electra-4` = RECOGNITION, WRITE, FORCE_ERROR, RESPONSE (nunca compartilham vetor com o
normal; probe F1 0,97–1,00 também na divisão temporal). Features: só conteúdo
(`request, fc, error, address, data`); MAC/IP/tempo vão para o perfil. Atribuição: todo
pacote de ataque tem o host MitM como origem ou destino. Prevalência: 0,41 % de ataques.

## Protocolo ReSIDS (`scripts/cross_domain_electra.py` → `results/cross_domain_electra.txt`)

8 especialistas (2 por categoria), divisão temporal global 70/30.

| | recall | FPR (teste) |
|---|---:|---:|
| GL k≥2, união dos 8 | **99,94 %** | **0 FP em 4,17 M normais** |
| FL k≥2, 4 nós | 51,7 % (RESPONSE 0 %) | 0 % |
| FL k≥2, 3 nós | 42,8 % | 0 % |
| clean5 (+READ), GL k≥2 | 38,1 % (READ 33,9 %) | 0 % |

**Transiente de processo:** no período de treino, os especialistas de RESPONSE marcam
50.114 pacotes normais (0,36 % do normal da captura), todos nos minutos 30–36, respostas do
PLC de:9c ao SCADA com valores de registrador ≈ 18.000, sem o host MitM. No teste não há
transiente → 0 FP. (`~/.audit_tmp/electra_fp_check.py`.)

## Triagem por janela (`scripts/_electra_window_triage.py` → `results/electra_window_triage.txt`)

Calibração nas janelas sem ataque do período de treino (margem ×1,25; piso 2 por
especialista); teste no período seguinte.

| janela | rede inteira | rede inteira OU especialista | FA benignos |
|---|---:|---:|---:|
| 1 s | 0,0 % | **88,7 %** (3.218/3.630) | 0 / 11.469 |
| 10 s | 0,0 % | **85,7 %** (414/483) | 0 / 1.028 |
| 60 s | 0,0 % | **89,2 %** (116/130) | 0 / 122 |

Limiar global = 241 pacotes/s (o transiente está na calibração) → a regra de rede inteira
não detecta nada; a por especialista isola o transiente no especialista de RESPONSE
(T = 241) e mantém os outros três no piso (2). Por categoria (1 s / 10 s / 60 s):
RECOGNITION e WRITE 100 %; FORCE_ERROR 13 / 18 / 27 % (1–2 pacotes por janela);
RESPONSE 4 / 70 / 93 %.
