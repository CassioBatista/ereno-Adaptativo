#!/bin/bash
# Phase A offline: GRASP CICIDS2017 (por-ataque v2 -> global penalizado -> combinar).
# Roda em WSL, loga cada estagio, escreve marcadores em 00_status.log.
cd ~/ereno-Adaptativo || exit 2
PY=~/venv-ereno314/bin/python
LOG=~/cicids_batch
mkdir -p "$LOG"
export PYTHONUNBUFFERED=1
ST="$LOG/00_status.log"

echo "==== ORQUESTRADOR GRASP CICIDS ====" | tee "$ST"
echo "[orq] INICIO $(date)" | tee -a "$ST"

echo "[orq] ESTAGIO 1: GRASP por-ataque (v2, sample 40k, no-improve 8, resumivel) $(date)" | tee -a "$ST"
$PY scripts/grasp_por_ataque_cicids.py --sample 40000 --normal-cap 40000 \
    --no-improvement 8 --min-pos 200 --tag cicids >> "$LOG/01_grasp_por_ataque.log" 2>&1
rc1=$?
echo "[orq] ESTAGIO 1 FIM rc=$rc1 $(date)" | tee -a "$ST"
[ $rc1 -ne 0 ] && { echo "[orq] ABORTA: estagio 1 falhou" | tee -a "$ST"; exit 1; }

echo "[orq] ESTAGIO 2: global penalizado (lambda=0.05, sample 150k) $(date)" | tee -a "$ST"
$PY scripts/grasp_global_penalizado_cicids.py --lambda 0.05 --sample 150000 \
    --no-improvement 15 > "$LOG/02_grasp_penalizado.log" 2>&1
rc2=$?
echo "[orq] ESTAGIO 2 FIM rc=$rc2 $(date)" | tee -a "$ST"
[ $rc2 -ne 0 ] && { echo "[orq] ABORTA: estagio 2 falhou" | tee -a "$ST"; exit 1; }

echo "[orq] ESTAGIO 3: combinar $(date)" | tee -a "$ST"
$PY scripts/combinar_cicids.py > "$LOG/03_combinar.log" 2>&1
rc3=$?
echo "[orq] ESTAGIO 3 FIM rc=$rc3 $(date)" | tee -a "$ST"
[ $rc3 -ne 0 ] && { echo "[orq] ABORTA: estagio 3 falhou" | tee -a "$ST"; exit 1; }

echo "[orq] ===== GRASP COMPLETO (phase A) $(date) =====" | tee -a "$ST"
echo "[orq] conjunto combinado:" | tee -a "$ST"
grep -E '"n_features"|"features"|"n_superset"|"n_global_penalizado"|"skipped' \
    features/all_in_one_cicids_combined.json | tee -a "$ST"
echo "[orq] FIM_TOTAL_PHASE_A" | tee -a "$ST"
