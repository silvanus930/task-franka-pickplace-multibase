#!/usr/bin/env bash
# A/B eval: standard HL vs RetryGrasp HL (max_retries=5, grasp_hold_s=1.0).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${SCRIPT_DIR}/results"
SUMMARY="${SCRIPT_DIR}/results/hl_retry_grasp_ab_summary.txt"
CKPT_NAME="$(basename "${POLICY_CKPT}" .pt)"

require_policy_ckpt
require_nepher_env

echo "HL retry/grasp — A/B eval (no training)" | tee "${SUMMARY}"
echo "Policy: ${POLICY_CKPT}" | tee -a "${SUMMARY}"
echo "Tweaks: max_retries=5, grasp_hold_s=1.0" | tee -a "${SUMMARY}"
echo "Started: $(date -u)" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

echo ">>> A: baseline HL (EnvhubPlay)" | tee -a "${SUMMARY}"
"${SCRIPT_DIR}/eval.sh" baseline "${POLICY_CKPT}" || true
BASE_JSON="${PROJECT_ROOT}/evaluation_result_hl_retry_grasp_baseline_${CKPT_NAME}.json"
BASE_SCORE="$(python -c "import json; print(json.load(open('${BASE_JSON}'))['score'])")"
echo "  baseline score=${BASE_SCORE}" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

echo ">>> B: RetryGrasp HL (max_retries=5, grasp_hold_s=1.0)" | tee -a "${SUMMARY}"
"${SCRIPT_DIR}/eval.sh" retry_grasp "${POLICY_CKPT}" || true
RG_JSON="${PROJECT_ROOT}/evaluation_result_hl_retry_grasp_${CKPT_NAME}.json"
RG_SCORE="$(python -c "import json; print(json.load(open('${RG_JSON}'))['score'])")"
echo "  retry_grasp score=${RG_SCORE}" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

echo "============================================================" | tee -a "${SUMMARY}"
if python -c "import sys; sys.exit(0 if float('${RG_SCORE}') > float('${BASE_SCORE}') else 1)"; then
  echo "WINNER: RetryGrasp HL (${RG_SCORE} > ${BASE_SCORE})" | tee -a "${SUMMARY}"
  echo "Consider adopting RetryGrasp for submission eval." | tee -a "${SUMMARY}"
elif python -c "import sys; sys.exit(0 if float('${RG_SCORE}') == float('${BASE_SCORE}') else 1)"; then
  echo "TIE: no change (${RG_SCORE})" | tee -a "${SUMMARY}"
else
  echo "KEEP standard HL (${BASE_SCORE} >= ${RG_SCORE})" | tee -a "${SUMMARY}"
fi
echo "Summary: ${SUMMARY}" | tee -a "${SUMMARY}"
echo "============================================================" | tee -a "${SUMMARY}"
