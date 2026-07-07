#!/usr/bin/env bash
# A/B eval: standard HL vs TimingSafe HL, same frozen S2 model_5500 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${SCRIPT_DIR}/results"
SUMMARY="${SCRIPT_DIR}/results/hl_timing_ab_summary.txt"

require_policy_ckpt
require_nepher_env

echo "HL mustard timing safe — A/B eval (no training)" | tee "${SUMMARY}"
echo "Policy: ${POLICY_CKPT}" | tee -a "${SUMMARY}"
echo "Started: $(date -u)" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

echo ">>> A: baseline HL (EnvhubPlay)" | tee -a "${SUMMARY}"
if "${SCRIPT_DIR}/eval.sh" baseline "${POLICY_CKPT}"; then
  BASE_PASS=1
else
  BASE_PASS=0
fi
BASE_JSON="${PROJECT_ROOT}/evaluation_result_hl_timing_baseline_model_5500.json"
BASE_SCORE="$(python -c "import json; print(json.load(open('${BASE_JSON}'))['score'])")"
echo "  baseline score=${BASE_SCORE}" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

echo ">>> B: TimingSafe HL (grasp_hold_s=0.55, mustard z=-0.045)" | tee -a "${SUMMARY}"
if "${SCRIPT_DIR}/eval.sh" timing_safe "${POLICY_CKPT}"; then
  SAFE_PASS=1
else
  SAFE_PASS=0
fi
SAFE_JSON="${PROJECT_ROOT}/evaluation_result_hl_timing_safe_model_5500.json"
SAFE_SCORE="$(python -c "import json; print(json.load(open('${SAFE_JSON}'))['score'])")"
echo "  timing_safe score=${SAFE_SCORE}" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

echo "============================================================" | tee -a "${SUMMARY}"
if python -c "import sys; sys.exit(0 if float('${SAFE_SCORE}') > float('${BASE_SCORE}') else 1)"; then
  echo "WINNER: TimingSafe HL (${SAFE_SCORE} > ${BASE_SCORE})" | tee -a "${SUMMARY}"
  echo "Consider adopting TimingSafe for submission eval." | tee -a "${SUMMARY}"
elif python -c "import sys; sys.exit(0 if float('${SAFE_SCORE}') == float('${BASE_SCORE}') else 1)"; then
  echo "TIE: no change (${SAFE_SCORE})" | tee -a "${SUMMARY}"
else
  echo "KEEP standard HL (${BASE_SCORE} >= ${SAFE_SCORE})" | tee -a "${SUMMARY}"
fi
echo "Summary: ${SUMMARY}" | tee -a "${SUMMARY}"
echo "============================================================" | tee -a "${SUMMARY}"
