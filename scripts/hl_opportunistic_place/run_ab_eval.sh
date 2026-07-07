#!/usr/bin/env bash
# A/B eval: baseline EnvhubPlay vs OpportunisticPlace with frozen champion checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

CHECKPOINT="${CHECKPOINT:-${BASELINE_CHECKPOINT}}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="${PROJECT_ROOT}/${CHECKPOINT}"
fi

echo "=== OpportunisticPlace A/B eval ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Baseline gate: score=${GATE_MIN_SCORE} successes=${GATE_MIN_SUCCESSES}/90"
echo

if [[ "${SKIP_BASELINE_EVAL:-0}" != "1" ]]; then
  echo "--- Baseline (EnvhubPlay) ---"
  "${SCRIPT_DIR}/eval.sh" "${CHECKPOINT}" baseline
  BASELINE_RUN="$(ls -td "${EVAL_NAV_ROOT}/logs/franka-pickplace-multibase-hl"/eval_run_* 2>/dev/null | head -1)"
  if [[ -n "${BASELINE_RUN}" && -f "${BASELINE_RUN}/summary.txt" ]]; then
  echo "Baseline summary: ${BASELINE_RUN}/summary.txt"
  cat "${BASELINE_RUN}/summary.txt"
  fi
  echo
fi

echo "--- Variant (OpportunisticPlace) ---"
"${SCRIPT_DIR}/eval.sh" "${CHECKPOINT}" variant
VARIANT_RUN="$(ls -td "${VARIANT_LOG_DIR}"/eval_run_* 2>/dev/null | head -1)"
if [[ -n "${VARIANT_RUN}" && -f "${VARIANT_RUN}/summary.txt" ]]; then
  echo "Variant summary: ${VARIANT_RUN}/summary.txt"
  cat "${VARIANT_RUN}/summary.txt"
  echo
  "${SCRIPT_DIR}/gate.sh" "${VARIANT_RUN}/summary.txt"
fi
