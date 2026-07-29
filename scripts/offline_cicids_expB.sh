#!/bin/bash
# Phase B offline: experimentos §1-§7 do CICIDS (combinado-31), ordem de prioridade.
# Resumivel: pula run cujo log ja tem o bloco final (MONOL). Roda em WSL, destacado.
cd ~/ereno-Adaptativo || exit 2
PY=~/venv-ereno314/bin/python
LOG=~/cicids_batch
mkdir -p "$LOG"
export PYTHONUNBUFFERED=1
ST="$LOG/00_status_B.log"

run() {
  local cfg="$1"; local clients="${2:-10}"
  local out="$LOG/B_${cfg}.log"
  if [ -f "$out" ] && grep -q "MONOL" "$out"; then
    echo "[orqB] JA FEITO $cfg (pula, resume)" | tee -a "$ST"; return 0
  fi
  echo "[orqB] INICIO $cfg (clients=$clients) $(date)" | tee -a "$ST"
  $PY ereno.py distributed GR-G-VND 6 all_in_one_cicids_v2 \
     --conf conf/experiments/${cfg}.yaml --clients "$clients" > "$out" 2>&1
  echo "[orqB] FIM $cfg rc=$? $(date)" | tee -a "$ST"
}

echo "==== ORQUESTRADOR EXPERIMENTOS CICIDS (phase B) ====" | tee "$ST"
echo "[orqB] INICIO $(date)" | tee -a "$ST"

# §1/§3 — Central x FL x GL + equivalencia (PRIORIDADE: teste de H6)
run cicids_federado_combined
run cicids_gossip_combined
# §5 — especialista x IID
run cicids_federado_combined_iid
run cicids_gossip_combined_iid
# §6 — k-de-n (k>=2) em N=10
run cicids_federado_combined_k2
run cicids_gossip_combined_k2
# §4 — adaptabilidade coarse (comutacao + encolhimento)
run cicids_adapt_fl_gl_combined
run cicids_adapt_gl_fl_combined
# §7 — adaptacao fine-shrink x k-de-n
run cicids_adapt_fl_gl_shrink1_k1
run cicids_adapt_gl_fl_shrink1_k1
run cicids_adapt_fl_gl_shrink1_k2
run cicids_adapt_gl_fl_shrink1_k2

echo "[orqB] ===== EXPERIMENTOS COMPLETOS (phase B) $(date) =====" | tee -a "$ST"
echo "[orqB] FIM_TOTAL_PHASE_B" | tee -a "$ST"
