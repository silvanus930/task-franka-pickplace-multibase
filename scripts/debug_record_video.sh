#!/usr/bin/env bash
# Record a short debug video of HL+LL pick-place (1 env, SafePlay).
#
# Requirements for video on headless Linux:
#   - libs from install_isaac_headless_libs.sh
#   - Working NVIDIA Vulkan (vulkaninfo --summary must succeed)
#   - --enable_cameras (Isaac offscreen rendering kit)
#
# If Vulkan fails (ERROR_INCOMPATIBLE_DRIVER), this script will still run but
# may hang or produce empty video — use log-based debug instead.
#
# Usage:
#   ./scripts/debug_record_video.sh
#   CHECKPOINT=path/to/model_5400.pt ./scripts/debug_record_video.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STACK_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
VENV="${STACK_ROOT}/.venv"

CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/logs/rsl_rl/franka_ll_ee_tracking/2026-07-06_22-44-55_v2_grip_finetune/model_5400.pt}"
TASK="${TASK:-Nepher-Franka-PickPlace-HL-Multibase-EnvhubSafePlay-v0}"
VIDEO_LEN="${VIDEO_LEN:-300}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if command -v vulkaninfo >/dev/null 2>&1; then
  echo "--- Vulkan probe ---"
  if ! vulkaninfo --summary 2>&1 | head -8; then
    echo ""
    echo "[WARN] Vulkan not available — video may fail on this OS/container."
    echo "       Training/eval without --video still works."
    read -r -p "Continue anyway? [y/N] " ans
    [[ "${ans,,}" == "y" ]] || exit 1
  fi
fi

cd "${PROJECT_ROOT}/scripts/rsl_rl"

echo "Recording ${VIDEO_LEN} steps: ${TASK}"
echo "Checkpoint: ${CHECKPOINT}"

python play.py \
  --task="${TASK}" \
  --headless \
  --enable_cameras \
  --video \
  --video_length "${VIDEO_LEN}" \
  --num_envs 1 \
  --max_episodes 1 \
  --checkpoint "${CHECKPOINT}"

echo "Video saved under: ${PROJECT_ROOT}/best_policy/videos/play/"
