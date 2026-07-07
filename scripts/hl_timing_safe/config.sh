#!/usr/bin/env bash
# HL-only mustard timing safe tweak — eval with frozen S2 checkpoint (no training).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STACK_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
EVAL_NAV="${STACK_ROOT}/eval-nav"
VENV="${STACK_ROOT}/.venv"

# Current top model (HL-in-loop v3 peak).
POLICY_CKPT="${PROJECT_ROOT}/logs/rsl_rl/franka_ll_ee_tracking/2026-07-07_19-01-58_hl_in_loop_v3_finetune/model_5510.pt"
BASELINE_SCORE="0.141"
BASELINE_SUCCESSES="14"

EVAL_BASELINE_CONFIG="${EVAL_NAV}/configs/task-franka-pickplace-multibase.yaml"
EVAL_TIMING_CONFIG="${EVAL_NAV}/configs/task-franka-pickplace-multibase-timing-safe.yaml"

activate_venv() {
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
}

require_policy_ckpt() {
  if [[ ! -f "${POLICY_CKPT}" ]]; then
    echo "[ERROR] Policy checkpoint not found: ${POLICY_CKPT}" >&2
    exit 1
  fi
}

require_nepher_env() {
  if ! python -c "from nepher.loader.registry import load_env; load_env('franka-pickplace-multibase-sample', category='manipulation')" 2>/dev/null; then
    echo "[WARN] Nepher env not cached. Run:" >&2
    echo "  nepher download franka-pickplace-multibase-sample" >&2
  fi
}
