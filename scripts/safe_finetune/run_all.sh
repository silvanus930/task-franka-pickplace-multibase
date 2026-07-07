#!/usr/bin/env bash
# Run all three safe strategies sequentially; pick global best that beats baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${SCRIPT_DIR}/results"
GLOBAL_BEST_CKPT=""
GLOBAL_BEST_SCORE="-1"
SUMMARY="${SCRIPT_DIR}/results/all_strategies_summary.txt"
: > "${SUMMARY}"

echo "Safe finetune campaign — baseline gate: ${BASELINE_SCORE} (${BASELINE_SUCCESSES}/90)" | tee "${SUMMARY}"
echo "Started: $(date -u)" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

for STRATEGY in s1_safe_grip s2_safe_smooth s3_safe_shallow s4_safe_combo s5_safe_disp; do
  echo ">>> Running ${STRATEGY}" | tee -a "${SUMMARY}"
  SKIP_BASELINE_EVAL=1 "${SCRIPT_DIR}/run_pipeline.sh" "${STRATEGY}" || true

  BEST_FILE="${SCRIPT_DIR}/results/${STRATEGY}_best_ckpt.txt"
  SCORE_FILE="${SCRIPT_DIR}/results/${STRATEGY}_best_score.txt"
  if [[ -f "${BEST_FILE}" && -f "${SCORE_FILE}" ]]; then
    CKPT="$(cat "${BEST_FILE}")"
    SCORE="$(cat "${SCORE_FILE}")"
    echo "  ${STRATEGY}: ${SCORE} — ${CKPT}" | tee -a "${SUMMARY}"
    if python -c "import sys; sys.exit(0 if float('${SCORE}') > float('${GLOBAL_BEST_SCORE}') else 1)"; then
      GLOBAL_BEST_SCORE="${SCORE}"
      GLOBAL_BEST_CKPT="${CKPT}"
    fi
  else
    echo "  ${STRATEGY}: no improvement" | tee -a "${SUMMARY}"
  fi
  echo "" | tee -a "${SUMMARY}"
done

echo "============================================================" | tee -a "${SUMMARY}"
if [[ -n "${GLOBAL_BEST_CKPT}" ]]; then
  echo "GLOBAL BEST: ${GLOBAL_BEST_CKPT} score=${GLOBAL_BEST_SCORE}" | tee -a "${SUMMARY}"
  echo "Promote this checkpoint for submission." | tee -a "${SUMMARY}"
else
  echo "No strategy beat baseline. Keep:" | tee -a "${SUMMARY}"
  echo "  ${BASELINE_CKPT}" | tee -a "${SUMMARY}"
fi
echo "Finished: $(date -u)" | tee -a "${SUMMARY}"
echo "============================================================" | tee -a "${SUMMARY}"
