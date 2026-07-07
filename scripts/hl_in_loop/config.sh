#!/usr/bin/env bash
# Shared paths and gates for HL-in-loop LL finetune (Wave 3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STACK_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
EVAL_NAV="${STACK_ROOT}/eval-nav"
VENV="${STACK_ROOT}/.venv"

# HL_VERSION: v4 (Option B) | v3 | v2 (legacy 80-iter run)
HL_VERSION="${HL_VERSION:-v4}"

case "${HL_VERSION}" in
  v2)
    TASK="Nepher-Franka-PickPlace-HL-LL-Finetune-v0"
    RUN_NAME="hl_in_loop_finetune"
    NUM_ENVS="${NUM_ENVS:-1024}"
    TRAIN_ITERS="${TRAIN_ITERS:-80}"
    GATE_MIN_SUCCESSES="${GATE_MIN_SUCCESSES:-13}"
    ;;
  v3)
    TASK="Nepher-Franka-PickPlace-HL-LL-FinetuneV3-v0"
    RUN_NAME="hl_in_loop_v3_finetune"
    NUM_ENVS="${NUM_ENVS:-512}"
    TRAIN_ITERS="${TRAIN_ITERS:-40}"
    GATE_MIN_SUCCESSES="${GATE_MIN_SUCCESSES:-14}"
    ;;
  v4)
    TASK="Nepher-Franka-PickPlace-HL-LL-FinetuneV4-v0"
    RUN_NAME="hl_in_loop_v4_finetune"
    NUM_ENVS="${NUM_ENVS:-512}"
    TRAIN_ITERS="${TRAIN_ITERS:-20}"
    GATE_MIN_SUCCESSES="${GATE_MIN_SUCCESSES:-15}"
    ;;
  *)
    echo "[ERROR] Unknown HL_VERSION=${HL_VERSION} (use v2, v3, or v4)" >&2
    exit 1
    ;;
esac

# Current top checkpoint (HL-in-loop v3 peak).
BASELINE_CKPT="${PROJECT_ROOT}/logs/rsl_rl/franka_ll_ee_tracking/2026-07-07_19-01-58_hl_in_loop_v3_finetune/model_5510.pt"
BASELINE_SCORE="0.141"
BASELINE_SUCCESSES="14"

EVAL_CONFIG="${EVAL_NAV}/configs/task-franka-pickplace-multibase.yaml"
RESULTS_TAG="hl_in_loop_${HL_VERSION}"

activate_venv() {
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
}

require_baseline_ckpt() {
  if [[ ! -f "${BASELINE_CKPT}" ]]; then
    echo "[ERROR] Baseline checkpoint not found: ${BASELINE_CKPT}" >&2
    exit 1
  fi
}

require_nepher_env() {
  if ! python -c "from nepher.loader.registry import load_env; load_env('franka-pickplace-multibase-sample', category='manipulation')" 2>/dev/null; then
    echo "[WARN] Nepher env not cached. Run:" >&2
    echo "  nepher download franka-pickplace-multibase-sample" >&2
  fi
}
