#!/usr/bin/env bash
# Train one safe finetune strategy (sources strategies/<name>.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

STRATEGY="${1:?Usage: $0 <s1_safe_grip|s2_safe_smooth|s3_safe_shallow|s4_safe_combo|s5_safe_disp>}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/strategies/${STRATEGY}.sh"

require_baseline_ckpt
activate_venv
cd "${PROJECT_ROOT}/scripts/rsl_rl"

echo "============================================================"
echo "Train: ${STRATEGY_LABEL}"
echo "  Task        : ${TASK}"
echo "  Base ckpt   : ${BASELINE_CKPT}"
echo "  Extra iters : ${TRAIN_ITERS}"
echo "  Target fail : ${TARGET_FAILURE}"
echo "============================================================"

python train.py \
  --task="${TASK}" \
  --headless \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${TRAIN_ITERS}" \
  --run_name "${RUN_NAME}" \
  --resume \
  --checkpoint "${BASELINE_CKPT}"

LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/franka_ll_ee_tracking"
LATEST_RUN="$(ls -td "${LOG_ROOT}"/*_"${RUN_NAME}" 2>/dev/null | head -1)"
if [[ -z "${LATEST_RUN}" ]]; then
  echo "[ERROR] Could not find run directory for ${RUN_NAME}" >&2
  exit 1
fi

echo "${LATEST_RUN}" > "${SCRIPT_DIR}/.last_run_${STRATEGY_ID}"
echo "[INFO] Run directory: ${LATEST_RUN}"
ls -1 "${LATEST_RUN}"/model_*.pt 2>/dev/null | tail -5
