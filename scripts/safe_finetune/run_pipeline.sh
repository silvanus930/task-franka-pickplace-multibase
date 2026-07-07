#!/usr/bin/env bash
# Full pipeline: train → eval checkpoints → gate (one strategy).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

STRATEGY="${1:?Usage: $0 <s1_safe_grip|s2_safe_smooth|s3_safe_shallow|s4_safe_combo|s5_safe_disp>}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/strategies/${STRATEGY}.sh"

mkdir -p "${SCRIPT_DIR}/results"

echo ""
echo "################################################################"
echo "# Pipeline: ${STRATEGY_ID}"
echo "# ${STRATEGY_LABEL}"
echo "################################################################"
echo ""

# Optional: baseline eval for reference (skip if SKIP_BASELINE_EVAL=1)
if [[ "${SKIP_BASELINE_EVAL:-0}" != "1" ]]; then
  echo "--- Step 0: baseline reference ($(basename "${BASELINE_CKPT}")) ---"
  "${SCRIPT_DIR}/eval.sh" "${BASELINE_CKPT}" "baseline_ref" || true
fi

echo "--- Step 1: train (${TRAIN_ITERS} extra iters from $(basename "${BASELINE_CKPT}")) ---"
"${SCRIPT_DIR}/train.sh" "${STRATEGY}"

RUN_FILE="${SCRIPT_DIR}/.last_run_${STRATEGY_ID}"
RUN_DIR="$(cat "${RUN_FILE}")"
mapfile -t CKPTS < <(ls -1 "${RUN_DIR}"/model_*.pt 2>/dev/null | sort -V)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[ERROR] No checkpoints in ${RUN_DIR}" >&2
  exit 1
fi

BEST_CKPT=""
BEST_SCORE="-1"
GATE_LOG="${SCRIPT_DIR}/results/${STRATEGY_ID}_gate_summary.txt"
: > "${GATE_LOG}"

echo "--- Step 2: official eval on each saved checkpoint ---" | tee -a "${GATE_LOG}"
for CKPT in "${CKPTS[@]}"; do
  TAG="${STRATEGY_ID}_$(basename "${CKPT}" .pt)"
  echo "" | tee -a "${GATE_LOG}"
  echo "Evaluating ${CKPT}" | tee -a "${GATE_LOG}"
  if "${SCRIPT_DIR}/eval.sh" "${CKPT}" "${TAG}"; then
    RESULT_JSON="${PROJECT_ROOT}/evaluation_result_${TAG}_$(basename "${CKPT}" .pt).json"
    SCORE="$(python -c "import json; print(json.load(open('${RESULT_JSON}'))['score'])")"
    echo "PASS ${CKPT} score=${SCORE}" | tee -a "${GATE_LOG}"
    if python -c "import sys; sys.exit(0 if float('${SCORE}') > float('${BEST_SCORE}') else 1)"; then
      BEST_SCORE="${SCORE}"
      BEST_CKPT="${CKPT}"
    fi
  else
    echo "FAIL ${CKPT} (below baseline ${BASELINE_SCORE})" | tee -a "${GATE_LOG}"
  fi
done

echo "" | tee -a "${GATE_LOG}"
echo "============================================================" | tee -a "${GATE_LOG}"
if [[ -n "${BEST_CKPT}" ]]; then
  echo "BEST in ${STRATEGY_ID}: ${BEST_CKPT} (score=${BEST_SCORE})" | tee -a "${GATE_LOG}"
  echo "${BEST_CKPT}" > "${SCRIPT_DIR}/results/${STRATEGY_ID}_best_ckpt.txt"
  echo "${BEST_SCORE}" > "${SCRIPT_DIR}/results/${STRATEGY_ID}_best_score.txt"
else
  echo "No checkpoint beat baseline ${BASELINE_SCORE}. Keep $(basename "${BASELINE_CKPT}")" | tee -a "${GATE_LOG}"
fi
echo "Gate log: ${GATE_LOG}" | tee -a "${GATE_LOG}"
echo "============================================================" | tee -a "${GATE_LOG}"
