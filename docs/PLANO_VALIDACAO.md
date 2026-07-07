# Plano de Validação e Testes — ereno-Adaptativo

Comparação experimental entre quatro arquiteturas de treinamento do IDS —
**monolítica (centralizada)**, **federada**, **gossip** e **mista
(federated → gossip)** — variando o número de clientes.

## 1. Objetivo e hipóteses

Medir o custo (ou ganho) de distribuir o treinamento sob condições idênticas
de dataset, features e partição de dados, e validar que a comutação de
arquitetura em tempo de execução não degrada o modelo.

| # | Hipótese | Como será verificada |
|---|----------|----------------------|
| H1 | Com dados IID, o federado se aproxima do monolítico (ΔF1 pequeno) | E01–E03 vs E00 |
| H2 | O gossip puro rende F1 ≤ federado com mesmo nº de rounds (agregação local, visão parcial da rede) | E04–E06 vs E01–E03 |
| H3 | No misto, a comutação federated→gossip não causa queda abrupta: F1 final próximo do federado puro | E07–E09 vs E01–E03 |
| H4 | Aumentar clientes reduz dados por cliente e tende a reduzir o F1 distribuído; o monolítico é invariante | tendência ao longo de 3→5→10 |

## 2. Fatores

**Fixos em toda a campanha** (mudar qualquer um invalida as comparações):

| Fator | Valor | Onde |
|---|---|---|
| Dataset | `all_in_one_cicids` | CLI |
| Seleção de features | GRASP `I-G-VND`, avaliador J48 (`clf_idx 2`) | CLI |
| Estratégia federada | `xgb_bagging` | YAML/CLI |
| Agregação gossip | `xgb` (merge de boosters) | YAML |
| Particionador | `iid` | YAML |
| Topologia gossip | `ring` | YAML |
| Rounds | 20 | YAML |
| GRASP `max_iterations` | 10 | YAML |
| Seeds (repetições) | 42, 43, 44 | YAML (`seed`) |

**Variáveis:**

| Fator | Níveis |
|---|---|
| Arquitetura | monolítica, federada, gossip, mista |
| Nº de clientes | 3, 5, 10 |

## 3. Matriz de experimentos

| ID | Arquitetura | Clientes | Schedule (rounds 1–20) |
|----|-------------|----------|------------------------|
| E00 | Monolítica | — | não se aplica (treino único com todos os dados) |
| E01 | Federada | 3 | 1–20 federated |
| E02 | Federada | 5 | 1–20 federated |
| E03 | Federada | 10 | 1–20 federated |
| E04 | Gossip | 3 | 1–20 gossip |
| E05 | Gossip | 5 | 1–20 gossip |
| E06 | Gossip | 10 | 1–20 gossip |
| E07 | Mista | 3 | 1–10 federated, 11–20 gossip |
| E08 | Mista | 5 | 1–10 federated, 11–20 gossip |
| E09 | Mista | 10 | 1–10 federated, 11–20 gossip |
| E10 *(opcional)* | Mista c/ perda de nós | 10→5 | 1–10 federated (10 nós), 11–20 gossip com `active_clients: [0,1,2,3,4]` |

Total: 10 células × 3 seeds = **30 execuções** (+3 do E10 opcional).
E00 é 1 execução por seed (não depende de clientes).

### Configurações YAML

Um arquivo por arquitetura em `conf/experiments/` (só o bloco `architecture`
muda; o restante é idêntico ao `conf/base.yaml`):

```yaml
# conf/experiments/federado.yaml — trecho architecture
architecture:
  manager: fixed
  default_mode: federated
  aggregation: xgb
  schedule:
    - {from: 1, to: 20, mode: federated}
```

```yaml
# conf/experiments/gossip.yaml — trecho architecture
architecture:
  manager: fixed
  default_mode: gossip
  aggregation: xgb
  schedule:
    - {from: 1, to: 20, mode: gossip}
```

```yaml
# conf/experiments/misto.yaml — trecho architecture
architecture:
  manager: fixed
  default_mode: federated
  aggregation: xgb
  schedule:
    - {from: 1,  to: 10, mode: federated}
    - {from: 11, to: 20, mode: gossip}
```

### Comandos

```bash
mkdir -p results
# exemplo: E08 (mista, 5 clientes, seed 42)
python ereno.py distributed I-G-VND 2 all_in_one_cicids \
    --conf conf/experiments/misto.yaml --clients 5 \
    2>&1 | tee results/E08_misto_c5_s42.log
```

Convenção de nome de log: `results/<ID>_<arquitetura>_c<clientes>_s<seed>.log`.
Para trocar o seed entre repetições, editar `seed:` no YAML do experimento
(ver pré-requisito P3).

## 4. Pré-requisitos (corrigir ANTES de iniciar a campanha)

| # | Gap | Impacto | Correção |
|---|-----|---------|----------|
| P1 **(bloqueante)** | `main_dist.py` só imprime métricas finais para estratégias com `majority_vote_predict` (só `ensemble`). Com `xgb_bagging`, a simulação termina **sem reportar F1** | Sem P1 não há dados para coletar | Avaliar via `strategy.get_global_model()` + `evaluate_predictions`, como o `main_fd.py` já faz |
| P2 **(bloqueante p/ E00)** | Não existe baseline monolítico no `main_dist.py`. O `ereno.py central` usa cross-validation, **não comparável** com o holdout 80/20 do distribuído | E00 incomparável | Treinar um XGBoost centralizado com as mesmas features e o mesmo split 80/20 dentro do `main_dist.py` (espelhar `main_fd.py:311-323`) |
| P3 | `seed` não tem flag CLI; repetições exigem editar YAML | Fricção operacional, risco de erro | Adicionar `--seed` ao `main_dist.py` (ou aceitar a edição manual) |
| P4 | GRASP re-executa a cada rodada e é estocástico: features podem variar entre execuções | Confunde a fonte de variação (features × arquitetura) | `I-G-VND` mitiga (espaço restrito a 11 features); **verificar V5**. Solução definitiva: cache/warm-start de features (projetado, não implementado) |
| P5 *(desejável)* | Não há métrica por round exportada | Sem curvas de convergência; H2/H3 avaliadas só pelo F1 final | Logar F1 do modelo agregado a cada round (via `aggregate_evaluate`) |

## 5. Métricas coletadas

**Primárias** (do teste global, 20% estratificado): F1-score, accuracy,
precision, recall, VP/VN/FP/FN.

**Secundárias**: tempo do GRASP, tempo total da simulação, features
selecionadas (quantidade e índices), nº de árvores do booster final
(bagging cresce o modelo — comparar tamanho), logs de comutação.

## 6. Procedimento

1. `git pull && pip install -r requirements.txt` no ambiente WSL (`venv-ereno314`).
2. Aplicar P1 e P2 (bloqueantes); opcionalmente P3 e P5.
3. Executar a matriz em ordem crescente de ID, seeds 42→43→44, salvando logs.
4. Extrair métricas dos logs para um CSV consolidado
   (`ID, arquitetura, clientes, seed, f1, acc, prec, rec, tempo_grasp, tempo_sim`).
5. Analisar: média ± desvio por célula; deltas vs E00; tendência por nº de
   clientes; efeito da comutação (E07–E09 vs E01–E06).

## 7. Critérios de validação (sanity checks por execução)

| # | Checagem | Onde olhar no log |
|---|----------|--------------------|
| V1 | O modo de cada round obedece ao schedule | linhas `[ArchManager] round=N mode=...` |
| V2 | No misto, a comutação ocorre exatamente 1× (round 11) com transferência de modelo | `[Hybrid] mode switch: federated → gossip at round 11` |
| V3 | Nenhum cliente ficou sem dados após a partição | log do particionamento |
| V4 | F1 monolítico ≥ distribuído − tolerância; distribuído > monolítico +2pp exige investigação (suspeita de vazamento) | comparação E00 × Exx |
| V5 | Mesmas features em todas as execuções comparadas entre si | linha `Features selecionadas (N): [...]` |
| V6 | Mesma config + mesmo seed ⇒ mesmas métricas (reprodutibilidade) | repetir 1 célula e comparar |

## 8. Ameaças à validade

- **Estocasticidade do GRASP** entre execuções (mitigada por P4/V5).
- **Baseline incomparável**: nunca usar `ereno.py central` (CV) como E00 —
  usar o baseline holdout do P2.
- **Crescimento do modelo no bagging**: com mais rounds/clientes o booster
  global acumula mais árvores; parte do ganho pode vir de capacidade, não de
  arquitetura — reportar nº de árvores junto com F1.
- **Gossip em anel com 10 nós**: cada head vê só 2 vizinhos por round; 20
  rounds podem ser insuficientes para difundir informação pelo anel —
  se H2 falhar feio com 10 clientes, rodar sensibilidade com `--rounds 40`.
- **Paralelismo do Flower/Ray**: pode introduzir não-determinismo mesmo com
  seed fixo (V6 detecta).

## 9. Estimativa de esforço

GRASP `I-G-VND`: minutos por execução; simulação de 20 rounds com XGBoost:
~1–5 min. Estimativa por execução: ~5–15 min ⇒ campanha completa
(30 execuções): **~3 a 7 h de máquina**, paralelizável por célula se a RAM
permitir (cada simulação Flower sobe seus próprios processos).
