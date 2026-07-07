#!/usr/bin/env bash
# Official eval-nav benchmark for an HL-in-loop checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

CHECKPOINT="${1:?Usage: $0 <checkpoint.pt> [tag]}"
TAG="${2:-hl_in_loop}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

require_nepher_env
activate_venv

CKPT_NAME="$(basename "${CHECKPOINT}" .pt)"
RESULT_JSON="${PROJECT_ROOT}/evaluation_result_${TAG}_${CKPT_NAME}.json"
FULL_LOG="${EVAL_NAV}/logs/franka-pickplace-multibase-hl/final_${CKPT_NAME}.txt"

echo "============================================================"
echo "Official eval: ${CHECKPOINT}"
echo "  Log tag     : ${TAG}"
echo "  Result JSON : ${RESULT_JSON}"
echo "============================================================"

cd "${EVAL_NAV}"
NEPHER_EVAL_IN_PROCESS=1 python scripts/evaluate.py \
  --config "${EVAL_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --headless \
  --result-path "${RESULT_JSON}"

mkdir -p "${SCRIPT_DIR}/results"
if [[ -f "${FULL_LOG}" ]]; then
  cp -f "${FULL_LOG}" "${SCRIPT_DIR}/results/${TAG}_${CKPT_NAME}.log"
fi

python "${SCRIPT_DIR}/gate.py" "${RESULT_JSON}" \
  --baseline-score "${BASELINE_SCORE}" \
  --baseline-successes "${BASELINE_SUCCESSES}" \
  ${GATE_MIN_SUCCESSES:+--min-successes "${GATE_MIN_SUCCESSES}"}
echo "[INFO] Result: ${RESULT_JSON}"
