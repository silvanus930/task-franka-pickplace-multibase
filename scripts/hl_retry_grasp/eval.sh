#!/usr/bin/env bash
# Run official eval-nav with baseline or RetryGrasp HL task.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

VARIANT="${1:?Usage: $0 <baseline|retry_grasp> [checkpoint]}"
CHECKPOINT="${2:-${POLICY_CKPT}}"

case "${VARIANT}" in
  baseline) CONFIG="${EVAL_BASELINE_CONFIG}"; TAG="hl_retry_grasp_baseline" ;;
  retry_grasp) CONFIG="${EVAL_RETRY_GRASP_CONFIG}"; TAG="hl_retry_grasp" ;;
  *)
    echo "[ERROR] VARIANT must be baseline or retry_grasp" >&2
    exit 1
    ;;
esac

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

require_nepher_env
activate_venv

CKPT_NAME="$(basename "${CHECKPOINT}" .pt)"
RESULT_JSON="${PROJECT_ROOT}/evaluation_result_${TAG}_${CKPT_NAME}.json"

echo "============================================================"
echo "HL retry/grasp A/B eval: ${VARIANT}"
echo "  Config      : ${CONFIG}"
echo "  Checkpoint  : ${CHECKPOINT}"
echo "  Result JSON : ${RESULT_JSON}"
echo "============================================================"

cd "${EVAL_NAV}"
NEPHER_EVAL_IN_PROCESS=1 python scripts/evaluate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --headless \
  --result-path "${RESULT_JSON}"

mkdir -p "${SCRIPT_DIR}/results"
python "${SCRIPT_DIR}/gate.py" "${RESULT_JSON}" \
  --baseline-score "${BASELINE_SCORE}" \
  --baseline-successes "${BASELINE_SUCCESSES}" \
  --label "${VARIANT}"
echo "[INFO] Result: ${RESULT_JSON}"
