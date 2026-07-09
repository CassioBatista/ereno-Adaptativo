# Datasets — proveniência e verificação

## CICIDS (all_in_one_cicids*.csv)

Ambos os arquivos derivam do **CICIDS2017** (Canadian Institute for
Cybersecurity / University of New Brunswick), formato *MachineLearningCVE*
(CICFlowMeter). Citação acadêmica: Sharafaldin, Lashkari & Ghorbani,
*"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization"*, ICISSP 2018. Página oficial:
<https://www.unb.ca/cic/datasets/ids-2017.html>.

### Proveniência verificada (2026-07)

- As 78 features do `all_in_one_cicids.csv` são **as mesmas, na mesma
  ordem**, do MachineLearningCVE (F1 = Destination Port … F78 = Idle Min).
  Prova: para as 5 classes preservadas 100% (GoldenEye, slowloris, ssh,
  ftp, xss), as somas por coluna são idênticas (390/390 comparações,
  rtol 1e-9) entre o all_in_one e os CSVs originais.
- Origem das linhas: 3 dias do CICIDS2017 — Tuesday (FTP/SSH-Patator),
  Wednesday (DoS ×4 + Heartbleed), Thursday-Morning (ataques web).

### v1 — `all_in_one_cicids.csv` (57.613 amostras, versionado)

Recorte com **teto de ~10 mil amostras por classe**, aplicado de fato só a
BENIGN (1.040.291 → 9.999) e hulk (231.073 → 10.000); as demais 9 classes
estão íntegras. Consequência: prevalência invertida (82,6% ataques), o que
infla métricas — o F1 do classificador trivial "sempre ataque" já é 90,5%.

### v2 — `all_in_one_cicids_v2.csv` (1.307.282 amostras, NÃO versionado)

Regenerado dos originais **sem teto por classe** (higiene: 1.696 linhas com
NaN/Infinity removidas, 0,13%). Prevalência realista: **79,5% BENIGN**.
O arquivo tem 392 MB (acima do limite do GitHub); para gerá-lo:

```bash
# baixar os 3 CSVs originais (oficial via formulário, ou espelho):
#   https://huggingface.co/datasets/c01dsnap/CIC-IDS2017
python scripts/regenerate_cicids_v2.py <dir_dos_csvs> all_in_one_cicids_v2.csv
```

| classe | v1 | v2 |
|---|---:|---:|
| BENIGN | 9.999 (17,4%) | 1.039.547 (79,5%) |
| hulk | 10.000 | 230.124 |
| GoldenEye | 10.293 | 10.293 |
| ftp | 7.938 | 7.935 |
| ssh | 5.897 | 5.897 |
| slowloris | 5.796 | 5.796 |
| Slowhttptest | 5.499 | 5.499 |
| brute | 1.507 | 1.507 |
| xss | 652 | 652 |
| sql | 21 | 21 |
| Heartbleed | 11 | 11 |

(pequenas diferenças v1→v2 em ftp/hulk vêm da higiene de NaN/Inf.)

### Hashes de verificação (MD5)

| arquivo | md5 |
|---|---|
| Tuesday-WorkingHours.pcap_ISCX.csv | `df16dccfd59a4ee126690fd6b71ee0a4` |
| Wednesday-workingHours.pcap_ISCX.csv | `bf0dd7e9d991987df4e13ea58a1b409c` |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | `13e1c70d2b380bf5d90f82e60e7befb1` |
| all_in_one_cicids.csv (v1, LF) | `913fed94c7a80713be13214acd3ee6f3` |
| all_in_one_cicids_v2.csv | `3e3fc367e07f666eb2da3e9354598dd2` |

### Implicações de uso do v2

- **Métricas**: com 79,5% de benignos, o piso do F1 trivial cai de 90,5%
  para ~34%, e o FPR passa a custar caro — cenário de avaliação honesto.
- **Custo**: 23× mais linhas. GRASP `I-G-VND` viável (avaliações de CV
  ~15–30 s); `GR-G-VND` sobe para horas — considerar `max_iterations`
  menor ou GRASP em subamostra estratificada.
- **Uso**: `python ereno.py distributed I-G-VND 2 all_in_one_cicids_v2 ...`
  (o sufixo contém "cicid", então os subconjuntos de features do CICIDS
  são selecionados automaticamente).

## Outros datasets versionados

| arquivo | amostras | origem |
|---|---:|---|
| all_in_one_kdd.csv | 148.517 | NSL-KDD/KDD-Cup (verificação de proveniência pendente) |
| all_in_one_wsn.csv | 374.661 | WSN-DS (verificação de proveniência pendente) |
