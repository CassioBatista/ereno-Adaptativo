#!/usr/bin/env bash
# Retomada IDEMPOTENTE das seleções GRASP definitivas.
#
# Roda apenas as seleções cujo JSON ainda não existe (interrupções por
# desligamento/queda são toleradas: o que terminou fica; o que faltou
# recomeça). Ordem: J48 (canônicas) → XGBoost (estabilidade).
#
# Uso (de dentro de ~/ereno-Adaptativo, com o venv ativo):
#   nohup bash scripts/resume_grasp_selections.sh > results/resume.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

run_if_missing() {
    local out="$1"; shift
    if [ -f "$out" ]; then
        echo "[resume] OK, já existe: $out"
    else
        echo "[resume] executando → $out"
        "$@"
        echo "[resume] exit=$? para $out"
    fi
}

mkdir -p results features

run_if_missing features/all_in_one_ereno_train.json \
    python ereno.py grasp GR-G-VND 2 all_in_one_ereno_train \
        --sample 150000 --no-improvement 15 --max-iterations 300 \
        --out features/all_in_one_ereno_train.json

run_if_missing features/all_in_one_cicids_v2.json \
    python ereno.py grasp GR-G-VND 2 all_in_one_cicids_v2 \
        --sample 150000 --no-improvement 15 --max-iterations 300 \
        --out features/all_in_one_cicids_v2.json

run_if_missing features/all_in_one_ereno_train_xgb.json \
    python ereno.py grasp GR-G-VND 6 all_in_one_ereno_train \
        --sample 50000 --no-improvement 10 --max-iterations 200 \
        --out features/all_in_one_ereno_train_xgb.json

run_if_missing features/all_in_one_cicids_v2_xgb.json \
    python ereno.py grasp GR-G-VND 6 all_in_one_cicids_v2 \
        --sample 50000 --no-improvement 10 --max-iterations 200 \
        --out features/all_in_one_cicids_v2_xgb.json

echo "[resume] todas as 4 seleções presentes em features/"
